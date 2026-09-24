// bayes_matting_nocv.cpp
// Dependency-free C++ rewrite of the provided Bayes matting code.
// - No OpenCV, no Eigen
// - Supports BMP (24bpp color for observation, 8bpp grayscale for trimap) input
// - Writes alpha as 8-bit PGM
// - Fixed 3x3 linear algebra + Schur complement for the 6x6 block solve
// - Uses features normalized to [0,1]
// Build:  g++ -O3 -std=c++11 -Wall -Wextra -o bayes_matting_nocv bayes_matting_nocv.cpp

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <cfloat>
#include <cassert>
#include <list>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <chrono>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ------------------------------
   Tiny image struct
   ------------------------------ */
struct ImageU8 {
    int w = 0, h = 0, c = 0;          // width, height, channels (1 or 3)
    std::vector<unsigned char> data;  // row-major, interleaved for c=3

    bool empty() const { return data.empty() || w<=0 || h<=0 || c<=0; }
    unsigned char* ptr() { return data.empty() ? NULL : &data[0]; }
    const unsigned char* ptr() const { return data.empty() ? NULL : &data[0]; }

    // Safe pixel accessors (no bounds check):
    inline unsigned char* row_ptr(int y) { return &data[y * w * c]; }
    inline const unsigned char* row_ptr(int y) const { return &data[y * w * c]; }
};

/* ------------------------------
   BMP (24bpp, 8bpp) Loader
   ------------------------------ */
#pragma pack(push,1)
struct BMPFileHeader {
    uint16_t bfType;      // 'BM'
    uint32_t bfSize;
    uint16_t bfReserved1;
    uint16_t bfReserved2;
    uint32_t bfOffBits;
};
struct BMPInfoHeader {
    uint32_t biSize;      // 40
    int32_t  biWidth;
    int32_t  biHeight;
    uint16_t biPlanes;    // 1
    uint16_t biBitCount;  // 24 or 8
    uint32_t biCompression; // 0 = BI_RGB
    uint32_t biSizeImage; // may be 0 for BI_RGB
    int32_t  biXPelsPerMeter;
    int32_t  biYPelsPerMeter;
    uint32_t biClrUsed;     // palette count
    uint32_t biClrImportant;
};
#pragma pack(pop)

// Loads 24-bit color BMP or 8-bit grayscale BMP.
// For 8-bit, we read palette (if present) and convert to grayscale using palette luminance.
static bool load_bmp(const std::string& path, ImageU8& out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if(!f) { std::perror(("open " + path).c_str()); return false; }

    BMPFileHeader fh;
    BMPInfoHeader ih;
    if(std::fread(&fh, sizeof(fh), 1, f) != 1) { std::fclose(f); return false; }
    if(std::fread(&ih, sizeof(ih), 1, f) != 1) { std::fclose(f); return false; }

    if(fh.bfType != 0x4D42 || ih.biPlanes != 1 || ih.biCompression != 0) {
        std::cerr << "Unsupported BMP: " << path << "\n";
        std::fclose(f);
        return false;
    }
    if(ih.biBitCount != 24 && ih.biBitCount != 8) {
        std::cerr << "Only 24bpp color or 8bpp grayscale BMP supported: " << path << "\n";
        std::fclose(f);
        return false;
    }
    const int W = ih.biWidth;
    const int H = std::abs(ih.biHeight);
    const bool bottom_up = (ih.biHeight > 0);

    long palette_entries = 0;
    if(ih.biBitCount == 8) {
        palette_entries = ih.biClrUsed ? ih.biClrUsed : 256;
        // Seek to palette start (immediately after headers)
        long palette_pos = sizeof(BMPFileHeader) + ih.biSize;
        std::fseek(f, palette_pos, SEEK_SET);
    }

    // Seek to pixel array
    std::fseek(f, fh.bfOffBits, SEEK_SET);

    out.w = W; out.h = H; out.c = (ih.biBitCount == 24) ? 3 : 1;
    out.data.assign(W * H * out.c, 0);

    if(ih.biBitCount == 24) {
        // Each row is padded to 4-byte boundary
        int row_bytes = ((W * 3 + 3) / 4) * 4;
        std::vector<unsigned char> row(row_bytes);
        for(int y = 0; y < H; ++y) {
            int yy = bottom_up ? (H-1-y) : y;
            if((int)std::fread(&row[0], 1, row_bytes, f) != row_bytes) { std::fclose(f); return false; }
            unsigned char* dst = out.row_ptr(yy);
            // BMP stores BGR; keep as BGR or convert to RGB as needed (we treat generically)
            for(int x=0;x<W;++x){
                dst[3*x+0] = row[3*x+0]; // B
                dst[3*x+1] = row[3*x+1]; // G
                dst[3*x+2] = row[3*x+2]; // R
            }
        }
    } else {
        // 8-bit, read palette then indices; convert to grayscale
        struct RGBQuad { unsigned char b,g,r,zero; };
        std::vector<RGBQuad> palette((size_t)palette_entries);
        long cur = std::ftell(f);
        long palette_start = sizeof(BMPFileHeader) + ih.biSize;
        if(cur != palette_start) std::fseek(f, palette_start, SEEK_SET);
        if(std::fread(&palette[0], sizeof(RGBQuad), (size_t)palette_entries, f) != (size_t)palette_entries) {
            std::fclose(f); return false;
        }
        // Now at pixel array
        std::fseek(f, fh.bfOffBits, SEEK_SET);
        int row_bytes = ((W + 3) / 4) * 4;
        std::vector<unsigned char> row(row_bytes);
        for(int y=0;y<H;++y){
            int yy = bottom_up ? (H-1-y) : y;
            if((int)std::fread(&row[0],1,row_bytes,f) != row_bytes){ std::fclose(f); return false; }
            unsigned char* dst = out.row_ptr(yy);
            for(int x=0;x<W;++x){
                unsigned char idx = row[x];
                RGBQuad c = palette[idx];
                // luminance
                double gray = 0.114 * c.b + 0.587 * c.g + 0.299 * c.r;
                if(gray < 0) gray = 0; if(gray > 255) gray = 255;
                dst[x] = (unsigned char)(gray + 0.5);
            }
        }
    }

    std::fclose(f);
    return true;
}

/* ------------------------------
   Save 8-bit PGM
   ------------------------------ */
static bool save_pgm_u8(const std::string& path, int W, int H, const unsigned char* data) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if(!f){ std::perror(("open " + path).c_str()); return false; }
    std::fprintf(f, "P5\n%d %d\n255\n", W, H);
    size_t n = (size_t)W * (size_t)H;
    if(std::fwrite(data, 1, n, f) != n){ std::fclose(f); return false; }
    std::fclose(f);
    return true;
}

/* ------------------------------
   General helpers
   ------------------------------ */
template<typename T> inline T sqr(T x){ return x*x; }
inline float norm255(unsigned char v){ return (float)v * (1.0f/255.0f); }
inline float clip01(float x){ if(x<0.f) return 0.f; if(x>1.f) return 1.f; return x; }

/* ------------------------------
   3×3 Linear Algebra (row-major)
   ------------------------------ */
struct Mat3 {
    // m[0..8] = [m00 m01 m02; m10 m11 m12; m20 m21 m22]
    double m[9];
    Mat3(){ for(int i=0;i<9;++i) m[i]=0.0; }
    static Mat3 Identity(){ Mat3 I; I.m[0]=I.m[4]=I.m[8]=1.0; return I; }
};
struct Vec3 { double v[3]; };

static Mat3 mat3_add(const Mat3& A, const Mat3& B){
    Mat3 C; for(int i=0;i<9;++i) C.m[i]=A.m[i]+B.m[i]; return C;
}
static Mat3 mat3_scale(const Mat3& A, double s){
    Mat3 C; for(int i=0;i<9;++i) C.m[i]=A.m[i]*s; return C;
}
static void mat3_mul_vec(const Mat3& A, const Vec3& x, Vec3& y){
    y.v[0] = A.m[0]*x.v[0]+A.m[1]*x.v[1]+A.m[2]*x.v[2];
    y.v[1] = A.m[3]*x.v[0]+A.m[4]*x.v[1]+A.m[5]*x.v[2];
    y.v[2] = A.m[6]*x.v[0]+A.m[7]*x.v[1]+A.m[8]*x.v[2];
}
static Mat3 mat3_mul(const Mat3& A, const Mat3& B){
    Mat3 C;
    for(int i=0;i<3;++i){
        for(int j=0;j<3;++j){
            double s=0.0;
            for(int k=0;k<3;++k) s += A.m[3*i+k]*B.m[3*k+j];
            C.m[3*i+j] = s;
        }
    }
    return C;
}
static int mat3_invert(const Mat3& A, Mat3& invA, double* det_out){
    double a00=A.m[0],a01=A.m[1],a02=A.m[2];
    double a10=A.m[3],a11=A.m[4],a12=A.m[5];
    double a20=A.m[6],a21=A.m[7],a22=A.m[8];
    double c00=(a11*a22 - a12*a21);
    double c01=-(a10*a22 - a12*a20);
    double c02=(a10*a21 - a11*a20);
    double c10=-(a01*a22 - a02*a21);
    double c11=(a00*a22 - a02*a20);
    double c12=-(a00*a21 - a01*a20);
    double c20=(a01*a12 - a02*a11);
    double c21=-(a00*a12 - a02*a10);
    double c22=(a00*a11 - a01*a10);
    double det=a00*c00 + a01*c01 + a02*c02;
    if(det_out) *det_out = det;
    if(std::fabs(det) < 1e-15) return 1;
    invA.m[0]=c00/det; invA.m[1]=c10/det; invA.m[2]=c20/det;
    invA.m[3]=c01/det; invA.m[4]=c11/det; invA.m[5]=c21/det;
    invA.m[6]=c02/det; invA.m[7]=c12/det; invA.m[8]=c22/det;
    return 0;
}

// Symmetric eigendecomposition via Jacobi (A overwritten)
static void mat3_eig_sym(Mat3 A, Mat3& V, double d[3]){
    // V = I
    V = Mat3::Identity();
    for(int iter=0; iter<50; ++iter){
        // find largest off-diagonal
        double a01 = std::fabs(A.m[1]);
        double a02 = std::fabs(A.m[2]);
        double a12 = std::fabs(A.m[5]);
        int p=0,q=1;
        if(a02>a01 && a02>=a12){ p=0; q=2; }
        else if(a12>=a01 && a12>=a02){ p=1; q=2; }
        int pq = 3*p+q;
        if(std::fabs(A.m[pq]) < 1e-12) break;

        double app = A.m[3*p+p];
        double aqq = A.m[3*q+q];
        double apq = A.m[pq];
        double c,s;
        if(std::fabs(aqq-app)<1e-30){
            c = 1.0/std::sqrt(2.0);
            s = (apq>=0)? c : -c;
        }else{
            double tau = (aqq-app)/(2.0*apq);
            double t = (tau>=0)? 1.0/(tau + std::sqrt(1.0+tau*tau))
                               : -1.0/(-tau + std::sqrt(1.0+tau*tau));
            c = 1.0/std::sqrt(1.0 + t*t);
            s = t*c;
        }
        // A <- J^T A J
        for(int k=0;k<3;++k){
            double Akp = A.m[3*k+p], Akq = A.m[3*k+q];
            A.m[3*k+p] = c*Akp - s*Akq;
            A.m[3*k+q] = s*Akp + c*Akq;
        }
        for(int k=0;k<3;++k){
            double Apk = A.m[3*p+k], Aqk = A.m[3*q+k];
            A.m[3*p+k] = c*Apk - s*Aqk;
            A.m[3*q+k] = s*Apk + c*Aqk;
        }
        A.m[3*p+q] = A.m[3*q+p] = 0.0;

        // V <- V J
        for(int k=0;k<3;++k){
            double Vkp = V.m[3*k+p], Vkq = V.m[3*k+q];
            V.m[3*k+p] = c*Vkp - s*Vkq;
            V.m[3*k+q] = s*Vkp + c*Vkq;
        }
    }
    d[0] = A.m[0]; d[1] = A.m[4]; d[2] = A.m[8];
}

/* ------------------------------
   Padding (constant=0)
   ------------------------------ */
static ImageU8 pad_image(const ImageU8& src, int padH, int padW){
    ImageU8 dst; dst.w = src.w + 2*padW; dst.h = src.h + 2*padH; dst.c = src.c;
    dst.data.assign(dst.w * dst.h * dst.c, 0);
    for(int y=0;y<src.h;++y){
        const unsigned char* srow = src.row_ptr(y);
        unsigned char* drow = &dst.data[(y+padH)*dst.w*dst.c + padW*dst.c];
        std::memcpy(drow, srow, (size_t)src.w * src.c);
    }
    return dst;
}

/* Copy src into dst wherever mask==255; elsewhere zero. dst must be allocated */
static void copy_with_mask(const ImageU8& src, const ImageU8& mask1ch, ImageU8& dst){
    assert(src.w==mask1ch.w && src.h==mask1ch.h && mask1ch.c==1);
    dst.w=src.w; dst.h=src.h; dst.c=src.c; dst.data.assign(src.w*src.h*src.c,0);
    for(int y=0;y<src.h;++y){
        const unsigned char* sp = src.row_ptr(y);
        const unsigned char* mp = mask1ch.row_ptr(y);
        unsigned char* dp = dst.row_ptr(y);
        for(int x=0;x<src.w;++x){
            if(mp[x]==255){
                if(src.c==3){ dp[3*x+0]=sp[3*x+0]; dp[3*x+1]=sp[3*x+1]; dp[3*x+2]=sp[3*x+2]; }
                else dp[x]=sp[x];
            }
        }
    }
}

/* ------------------------------
   Unknown-edge (8-neighborhood) detection
   cur_unknown: 1ch image where 255=unknown not yet solved; this function marks the boundary ring pixels as 0
   um_ptr: the original unknown (trimap ambiguous) mask (255 where truly unknown)
   Returns a list of linear indices (y*w + x) for current ring to solve.
   ------------------------------ */
static std::vector<int> find_unknown_edge(ImageU8& cur_unknown, const int H, const int W, const unsigned char* um_ptr){
    ImageU8 padded = pad_image(cur_unknown, 1, 1);
    const int W2 = padded.w;
    // mark boundary ring (set to 0) if any 8-neighbor is 0
    for(int i=0;i<H;++i){
        unsigned char* row = &padded.data[(i+1)*W2 + 1];
        unsigned char* res = &cur_unknown.data[i*W + 0];
        for(int j=0;j<W;++j){
            if(row[j]==255){
                if(row[j-1]==0 || row[j+1]==0 ||
                   row[j-1-W2]==0 || row[j-W2]==0 || row[j+1-W2]==0 ||
                   row[j-1+W2]==0 || row[j+W2]==0 || row[j+1+W2]==0){
                    res[j]=0;
                }
            }
        }
    }
    // collect ring ∩ original-unknown
    std::vector<int> edge;
    edge.reserve(W*H/10);
    for(int i=0;i<H*W;++i){
        if(cur_unknown.data[i]==0 && um_ptr[i]==255){
            edge.push_back(i);
        }
    }
    return edge;
}

/* ------------------------------
   Clustering (Orchard-Bouman like) in RGB space
   ------------------------------ */
struct Cluster {
    // immutable
    const unsigned char* img_ptr; // points to image->data (BGR or RGB), 3 channels
    const std::vector<int> pixels;
    const std::vector<float> W;
    const int pixel_num;
    // statistics
    Mat3 cov;
    Vec3 mean;
    Vec3 eigen_vec;
    float threshold;
    float max_eigen_value;

    Cluster(const unsigned char* _img_ptr, std::vector<int> _pixels, std::vector<float> _W)
        : img_ptr(_img_ptr), pixels(_pixels), W(_W), pixel_num((int)_pixels.size()) {
        // weighted mean
        double mu[3] = {0,0,0};
        double weight_sum = 0.0;
        for(int i=0;i<pixel_num;++i){
            int pos = 3 * pixels[i];
            double w = W[i];
            mu[0] += norm255(img_ptr[pos+0]) * w;
            mu[1] += norm255(img_ptr[pos+1]) * w;
            mu[2] += norm255(img_ptr[pos+2]) * w;
            weight_sum += w;
        }
        if(weight_sum <= 0.0) weight_sum = 1.0;
        mu[0]/=weight_sum; mu[1]/=weight_sum; mu[2]/=weight_sum;
        mean.v[0]=mu[0]; mean.v[1]=mu[1]; mean.v[2]=mu[2];

        // weighted covariance = (1/W) Sum_i w_i (x_i - mu)(x_i - mu)^T  + 1e-5 I
        Mat3 C; // zero
        for(int i=0;i<pixel_num;++i){
            int pos = 3 * pixels[i];
            double w = W[i];
            double dx0 = norm255(img_ptr[pos+0]) - mu[0];
            double dx1 = norm255(img_ptr[pos+1]) - mu[1];
            double dx2 = norm255(img_ptr[pos+2]) - mu[2];
            C.m[0]+=w*dx0*dx0; C.m[1]+=w*dx0*dx1; C.m[2]+=w*dx0*dx2;
            C.m[3]+=w*dx1*dx0; C.m[4]+=w*dx1*dx1; C.m[5]+=w*dx1*dx2;
            C.m[6]+=w*dx2*dx0; C.m[7]+=w*dx2*dx1; C.m[8]+=w*dx2*dx2;
        }
        for(int k=0;k<9;++k) C.m[k] /= weight_sum;
        C.m[0]+=1e-5; C.m[4]+=1e-5; C.m[8]+=1e-5;
        cov = C;

        // eigen (symmetric)
        Mat3 V;
        double evals[3];
        mat3_eig_sym(C, V, evals);

        // pick largest |lambda|
        int imax=0; double lam = std::fabs(evals[0]);
        for(int i=1;i<3;++i){
            double a = std::fabs(evals[i]);
            if(a>lam){ lam=a; imax=i; }
        }
        max_eigen_value = (float)lam;
        // eigen_vec = column imax of V
        eigen_vec.v[0] = V.m[0+3*imax];
        eigen_vec.v[1] = V.m[3+3*imax];
        eigen_vec.v[2] = V.m[6+3*imax];
        // threshold = mean^T * eigen_vec
        threshold = (float)( mean.v[0]*eigen_vec.v[0] + mean.v[1]*eigen_vec.v[1] + mean.v[2]*eigen_vec.v[2] );
    }

    bool operator<(const Cluster& rhs) const { return max_eigen_value < rhs.max_eigen_value; }
};

static std::vector< std::pair<Vec3, Mat3> >
make_clusters(const unsigned char* img_ptr, std::vector<int>& pixels, std::vector<float>& W, bool single=false)
{
    std::list<Cluster> nodes; nodes.push_back(Cluster(img_ptr, pixels, W));
    std::vector<int> lhs, rhs;
    std::vector<float> lhs_w, rhs_w;

    if(!single){
        while(true){
            // find node with max eigenvalue
            std::list<Cluster>::iterator it_max = nodes.begin();
            for(std::list<Cluster>::iterator it=nodes.begin(); it!=nodes.end(); ++it){
                if(it_max->max_eigen_value < it->max_eigen_value) it_max = it;
            }
            if(it_max->max_eigen_value > 0.05f){
                lhs.clear(); rhs.clear(); lhs_w.clear(); rhs_w.clear();
                for(int i=0;i<it_max->pixel_num;++i){
                    int pos = 3*it_max->pixels[i];
                    double proj = (norm255(it_max->img_ptr[pos+0]) * it_max->eigen_vec.v[0]
                                 + norm255(it_max->img_ptr[pos+1]) * it_max->eigen_vec.v[1]
                                 + norm255(it_max->img_ptr[pos+2]) * it_max->eigen_vec.v[2]);
                    if(proj <= it_max->threshold){
                        lhs.push_back(it_max->pixels[i]);
                        lhs_w.push_back(it_max->W[i]);
                    }else{
                        rhs.push_back(it_max->pixels[i]);
                        rhs_w.push_back(it_max->W[i]);
                    }
                }
                if(lhs.empty() || rhs.empty()) break;
                nodes.push_back(Cluster(it_max->img_ptr, lhs, lhs_w));
                nodes.push_back(Cluster(it_max->img_ptr, rhs, rhs_w));
                nodes.erase(it_max);
            }else{
                break;
            }
        }
    }

    std::vector< std::pair<Vec3,Mat3> > res;
    res.reserve(nodes.size());
    for(std::list<Cluster>::const_iterator it=nodes.begin(); it!=nodes.end(); ++it){
        res.push_back(std::make_pair(it->mean, it->cov));
    }
    return res;
}

/* ------------------------------
   The Bayes Matting (dependency-free)
   ------------------------------ */
static ImageU8 bayers_matting(
    const ImageU8& _observation,   // 3ch
    const ImageU8& _trimap,        // 1ch: 0=background, 255=foreground, else unknown
    bool single=false,
    int radius=12,
    float sigma=8.0f,
    int min_cluster_num=8,
    int max_iterations=37,
    float sigma_c = 0.01f)
{
    assert(_observation.c==3 && _trimap.c==1);
    assert(_observation.w==_trimap.w && _observation.h==_trimap.h);

    // padding
    ImageU8 observation = pad_image(_observation, radius, radius);
    ImageU8 trimap = pad_image(_trimap, radius, radius);

    const int H = observation.h;
    const int W = observation.w;
    const int length = H * W;
    const unsigned char* trimap_ptr = trimap.ptr();

    // masks
    ImageU8 fore_mask; fore_mask.w=W; fore_mask.h=H; fore_mask.c=1; fore_mask.data.assign(length,0);
    ImageU8 back_mask; back_mask.w=W; back_mask.h=H; back_mask.c=1; back_mask.data.assign(length,0);
    ImageU8 unknown_mask; unknown_mask.w=W; unknown_mask.h=H; unknown_mask.c=1; unknown_mask.data.assign(length,0);

    for(int i=0;i<length;++i){
        if(trimap_ptr[i]==255) fore_mask.data[i]=255;
        else if(trimap_ptr[i]==0) back_mask.data[i]=255;
        else unknown_mask.data[i]=255;
    }

    ImageU8 foreground, background;
    copy_with_mask(observation, fore_mask, foreground);
    copy_with_mask(observation, back_mask, background);

    // Gaussian window for spatial weighting
    const int filter_size = (2*radius+1)*(2*radius+1);
    std::vector<float> gauss_filter(filter_size, 0.0f);
    std::vector<int> gauss_offset(filter_size, 0);
    int off=0;
    const float sigma_inv = 1.0f/(2.0f*sigma*sigma);
    for(int di=-radius; di<=radius; ++di){
        for(int dj=-radius; dj<=radius; ++dj){
            gauss_filter[off] = std::exp(-(di*di+dj*dj)*sigma_inv);
            gauss_offset[off] = di*W + dj;
            ++off;
        }
    }
    float wsum=0.f; for(int i=0;i<filter_size;++i) wsum+=gauss_filter[i];
    for(int i=0;i<filter_size;++i) gauss_filter[i] /= (wsum>0.f? wsum : 1.f);

    // initial alpha: foreground mask / 255; unknown -> -1.0 (marker)
    std::vector<float> alpha(length, 0.f);
    for(int i=0;i<length;++i){
        if(fore_mask.data[i]==255) alpha[i] = 1.0f;
        else if(unknown_mask.data[i]==255) alpha[i] = -1.0f;
        else alpha[i] = 0.0f;
    }

    int targets_num = 0;
    for(int i=0;i<length;++i) if(unknown_mask.data[i]==255) ++targets_num;
    std::cout << "Unknown pixels to solve: " << targets_num << "\n";

    int obtained = 0;
    int last_ring = 0;
    ImageU8 cur_unknown = unknown_mask; // copy
    unsigned char* um_ptr = &unknown_mask.data[0];

    const float sigma_c_inv = 1.f/(sigma_c*sigma_c);

    auto t0 = std::chrono::steady_clock::now();

    while(obtained < targets_num){
        std::vector<int> unknown_edge = find_unknown_edge(cur_unknown, H, W, um_ptr);
        int ring_sz = (int)unknown_edge.size();
        std::cout << obtained << "/" << targets_num << "   to solve " << ring_sz << "\n";
        if(ring_sz == last_ring) break;
        last_ring = ring_sz;

        for(int u=0; u<ring_sz; ++u){
            int pos = unknown_edge[u];

            // collect known neighbors in window
            int valid_cnt=0;
            float init_alpha=0.f;
            std::vector<int> fore_pixels; fore_pixels.reserve(filter_size);
            std::vector<int> back_pixels; back_pixels.reserve(filter_size);
            std::vector<float> fore_weights; fore_weights.reserve(filter_size);
            std::vector<float> back_weights; back_weights.reserve(filter_size);

            for(int k=0;k<filter_size;++k){
                int idx = pos + gauss_offset[k];
                float a = alpha[idx];
                if(a >= 0.f){
                    float wf = a*a * gauss_filter[k];
                    if(wf>0.f){ fore_pixels.push_back(idx); fore_weights.push_back(wf); }
                    float wb = (1.f - a)*(1.f - a) * gauss_filter[k];
                    if(wb>0.f){ back_pixels.push_back(idx); back_weights.push_back(wb); }
                    init_alpha += a;
                    ++valid_cnt;
                }
            }
            if(valid_cnt>0) init_alpha /= (float)valid_cnt;
            else continue;

            if((int)fore_weights.size() < min_cluster_num || (int)back_weights.size() < min_cluster_num)
                continue;

            // clusters
            std::vector<int> fpx = fore_pixels; std::vector<float> fw = fore_weights;
            std::vector<int> bpx = back_pixels; std::vector<float> bw = back_weights;
            const unsigned char* fimg = &foreground.data[0];
            const unsigned char* bimg = &background.data[0];
            const unsigned char* oimg = &observation.data[0];

            std::vector< std::pair<Vec3,Mat3> > fore_clusters = make_clusters(fimg, fpx, fw, single);
            std::vector< std::pair<Vec3,Mat3> > back_clusters = make_clusters(bimg, bpx, bw, single);

            // current observed color C (normalized)
            Vec3 C; {
                int p3 = 3*pos;
                C.v[0] = norm255(oimg[p3+0]);
                C.v[1] = norm255(oimg[p3+1]);
                C.v[2] = norm255(oimg[p3+2]);
            }

            // iterate over all fore/back cluster pairs
            float best_alpha = 1.0f;
            Vec3 best_F; best_F.v[0]=1; best_F.v[1]=1; best_F.v[2]=1;
            Vec3 best_B; best_B.v[0]=0; best_B.v[1]=0; best_B.v[2]=0;
            float max_like = -FLT_MAX;

            const Mat3 I = Mat3::Identity();
            const int FN = (int)fore_clusters.size();
            const int BN = (int)back_clusters.size();

            for(int i=0;i<FN;++i){
                // fore stats
                const Vec3& muF = fore_clusters[i].first;
                Mat3   covF = fore_clusters[i].second;
                Mat3   invF; double detF=0.0;
                if(mat3_invert(covF, invF, &detF)!=0){ continue; } // safeguard

                for(int j=0;j<BN;++j){
                    // back stats
                    const Vec3& muB = back_clusters[j].first;
                    Mat3   covB = back_clusters[j].second;
                    Mat3   invB; double detB=0.0;
                    if(mat3_invert(covB, invB, &detB)!=0){ continue; }

                    float cur_alpha = init_alpha;
                    int iterations = 0;
                    float last_like = -FLT_MAX;

                    while(true){
                        // Block 6x6 system in 3x3 blocks:
                        // A11 = invF + I * (a^2) * s ; A12 = I * a(1-a) * s ; A22 = invB + I * (1-a)^2 * s
                        // b1  = invF*muF + C * a * s  ; b2  = invB*muB + C * (1-a) * s
                        // Solve with Schur complement
                        float a = cur_alpha;
                        float t1 = a*a * sigma_c_inv;
                        float t2 = a*(1.f-a) * sigma_c_inv;
                        float t3 = (1.f-a)*(1.f-a) * sigma_c_inv;

                        Mat3 A11 = mat3_add(invF, mat3_scale(I, t1));
                        Mat3 A12 = mat3_scale(I, t2);
                        Mat3 A22 = mat3_add(invB, mat3_scale(I, t3));

                        // b1, b2
                        Vec3 b1, b2, tmpv;
                        // invF*muF
                        mat3_mul_vec(invF, muF, tmpv);
                        b1.v[0]=tmpv.v[0]+C.v[0]*a*sigma_c_inv;
                        b1.v[1]=tmpv.v[1]+C.v[1]*a*sigma_c_inv;
                        b1.v[2]=tmpv.v[2]+C.v[2]*a*sigma_c_inv;
                        // invB*muB
                        mat3_mul_vec(invB, muB, tmpv);
                        b2.v[0]=tmpv.v[0]+C.v[0]*(1.f-a)*sigma_c_inv;
                        b2.v[1]=tmpv.v[1]+C.v[1]*(1.f-a)*sigma_c_inv;
                        b2.v[2]=tmpv.v[2]+C.v[2]*(1.f-a)*sigma_c_inv;

                        // inv(A11)
                        Mat3 invA11; double detA11=0.0;
                        if(mat3_invert(A11, invA11, &detA11)!=0){
                            // jitter diagonal and retry once
                            A11.m[0]+=1e-8; A11.m[4]+=1e-8; A11.m[8]+=1e-8;
                            if(mat3_invert(A11, invA11, &detA11)!=0) break;
                        }

                        // Schur: M = A22 - A12 * invA11 * A12
                        Mat3 A12invA11 = mat3_mul(A12, invA11);
                        Mat3 M = mat3_add(A22, mat3_scale(mat3_mul(A12invA11, A12), -1.0));

                        // rhs = b2 - A12 * invA11 * b1
                        Vec3 invA11b1, A12invA11b1;
                        mat3_mul_vec(invA11, b1, invA11b1);
                        mat3_mul_vec(A12, invA11b1, A12invA11b1);
                        Vec3 rhs; rhs.v[0] = b2.v[0] - A12invA11b1.v[0];
                        rhs.v[1] = b2.v[1] - A12invA11b1.v[1];
                        rhs.v[2] = b2.v[2] - A12invA11b1.v[2];

                        // Solve M * B = rhs (via inversion)
                        Mat3 invM; double detM=0.0;
                        if(mat3_invert(M, invM, &detM)!=0){
                            // jitter
                            M.m[0]+=1e-8; M.m[4]+=1e-8; M.m[8]+=1e-8;
                            if(mat3_invert(M, invM, &detM)!=0) break;
                        }
                        Vec3 B; mat3_mul_vec(invM, rhs, B);

                        // F = invA11 * (b1 - A12 * B)
                        Vec3 A12B, b1_minus;
                        mat3_mul_vec(A12, B, A12B);
                        b1_minus.v[0]=b1.v[0]-A12B.v[0];
                        b1_minus.v[1]=b1.v[1]-A12B.v[1];
                        b1_minus.v[2]=b1.v[2]-A12B.v[2];
                        Vec3 F; mat3_mul_vec(invA11, b1_minus, F);

                        // clamp F,B to [0,1]
                        for(int t=0;t<3;++t){ F.v[t]=clip01((float)F.v[t]); B.v[t]=clip01((float)B.v[t]); }

                        // update alpha
                        double FB0 = F.v[0]-B.v[0], FB1 = F.v[1]-B.v[1], FB2 = F.v[2]-B.v[2];
                        double num = (C.v[0]-B.v[0])*FB0 + (C.v[1]-B.v[1])*FB1 + (C.v[2]-B.v[2])*FB2;
                        double den = FB0*FB0 + FB1*FB1 + FB2*FB2;
                        if(den < 1e-12) break;
                        cur_alpha = (float)(num/den);
                        if(cur_alpha < 0.f) cur_alpha=0.f; else if(cur_alpha>1.f) cur_alpha=1.f;

                        // likelihood (unnormalized): LC + LF + LB
                        double dxC0 = C.v[0] - (cur_alpha*F.v[0] + (1.f-cur_alpha)*B.v[0]);
                        double dxC1 = C.v[1] - (cur_alpha*F.v[1] + (1.f-cur_alpha)*B.v[1]);
                        double dxC2 = C.v[2] - (cur_alpha*F.v[2] + (1.f-cur_alpha)*B.v[2]);
                        double LC = -(dxC0*dxC0 + dxC1*dxC1 + dxC2*dxC2) * sigma_c_inv;

                        // (F - muF)^T invF (F - muF) / 2
                        Vec3 dF; dF.v[0]=F.v[0]-muF.v[0]; dF.v[1]=F.v[1]-muF.v[1]; dF.v[2]=F.v[2]-muF.v[2];
                        Vec3 invF_dF; mat3_mul_vec(invF, dF, invF_dF);
                        double LF = -(dF.v[0]*invF_dF.v[0] + dF.v[1]*invF_dF.v[1] + dF.v[2]*invF_dF.v[2]) * 0.5;

                        Vec3 dB; dB.v[0]=B.v[0]-muB.v[0]; dB.v[1]=B.v[1]-muB.v[1]; dB.v[2]=B.v[2]-muB.v[2];
                        Vec3 invB_dB; mat3_mul_vec(invB, dB, invB_dB);
                        double LB = -(dB.v[0]*invB_dB.v[0] + dB.v[1]*invB_dB.v[1] + dB.v[2]*invB_dB.v[2]) * 0.5;

                        float like = (float)(LC + LF + LB);
                        if(like > max_like){
                            max_like = like;
                            best_alpha = cur_alpha;
                            best_F = F; best_B=B;
                        }

                        if(std::fabs(like - last_like) < 1e-5f || ++iterations >= max_iterations) break;
                        last_like = like;
                    } // while
                } // back clusters
            } // fore clusters

            // write results to foreground/background (optional) and alpha
            int idx3 = 3*pos;
            foreground.data[idx3+0] = (unsigned char)std::max(0, std::min(255, (int)std::round(255.0 * best_F.v[0])));
            foreground.data[idx3+1] = (unsigned char)std::max(0, std::min(255, (int)std::round(255.0 * best_F.v[1])));
            foreground.data[idx3+2] = (unsigned char)std::max(0, std::min(255, (int)std::round(255.0 * best_F.v[2])));

            background.data[idx3+0] = (unsigned char)std::max(0, std::min(255, (int)std::round(255.0 * best_B.v[0])));
            background.data[idx3+1] = (unsigned char)std::max(0, std::min(255, (int)std::round(255.0 * best_B.v[1])));
            background.data[idx3+2] = (unsigned char)std::max(0, std::min(255, (int)std::round(255.0 * best_B.v[2])));

            alpha[pos] = best_alpha;

            // mark solved
            um_ptr[pos] = 0;
            ++obtained;
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "bayers_matting took " << ms << " ms\n";

    // Convert alpha to 8-bit image and crop padding
    ImageU8 alpha_img; alpha_img.w=W; alpha_img.h=H; alpha_img.c=1;
    alpha_img.data.assign(length, 0);
    for(int i=0;i<length;++i){
        float a = alpha[i]; if(a<0.f) a=0.f; if(a>1.f) a=1.f;
        alpha_img.data[i] = (unsigned char)std::max(0, std::min(255, (int)std::round(255.0 * a)));
    }

    // Crop back to original size
    ImageU8 cropped; cropped.c=1;
    cropped.w = _observation.w; cropped.h=_observation.h;
    cropped.data.assign(cropped.w*cropped.h, 0);
    for(int y=0;y<cropped.h;++y){
        const unsigned char* src = &alpha_img.data[(y+radius)*W + radius];
        unsigned char* dst = &cropped.data[y*cropped.w];
        std::memcpy(dst, src, (size_t)cropped.w);
    }
    return cropped;
}


/* ------------------------------------------------------------
   === ADD: PPM I/O (P6, 8-bit, RGB) ===
   ------------------------------------------------------------ */
static bool load_ppm(const std::string& path, ImageU8& out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if(!f) { std::perror(("open "+path).c_str()); return false; }
    // Read magic
    char magic[3]={0,0,0};
    if(std::fread(magic,1,2,f)!=2){ std::fclose(f); return false; }
    if(magic[0]!='P' || magic[1]!='6'){ std::fclose(f); return false; }

    // Skip comments/whitespace utility
    auto skip_ws_comments = [&](void)->int{
        int c = std::fgetc(f);
        while(c!=EOF){
            if(c=='#'){ while(c!=EOF && c!='\n' && c!='\r') c=std::fgetc(f); c=std::fgetc(f); continue; }
            if(std::isspace(c)){ c=std::fgetc(f); continue; }
            break;
        }
        return c;
    };

    // Read width, height, maxval
    int c = skip_ws_comments(); if(c==EOF){ std::fclose(f); return false; }
    ungetc(c,f);
    int W=0,H=0,maxv=0;
    if(std::fscanf(f,"%d",&W)!=1){ std::fclose(f); return false; }
    c = skip_ws_comments(); if(c==EOF){ std::fclose(f); return false; } ungetc(c,f);
    if(std::fscanf(f,"%d",&H)!=1){ std::fclose(f); return false; }
    c = skip_ws_comments(); if(c==EOF){ std::fclose(f); return false; } ungetc(c,f);
    if(std::fscanf(f,"%d",&maxv)!=1){ std::fclose(f); return false; }
    if(maxv<=0 || maxv>255 || W<=0 || H<=0){ std::fclose(f); return false; }

    // Consume single whitespace after header
    int ws = std::fgetc(f); if(ws==EOF){ std::fclose(f); return false; }

    out.w=W; out.h=H; out.c=3;
    out.data.assign((size_t)W*H*3, 0);
    size_t need = (size_t)W*H*3;
    if(std::fread(&out.data[0],1,need,f)!=need){ std::fclose(f); return false; }
    std::fclose(f);
    return true;
}

static bool save_ppm_u8(const std::string& path, int W, int H, const unsigned char* data) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if(!f){ std::perror(("open "+path).c_str()); return false; }
    std::fprintf(f,"P6\n%d %d\n255\n", W,H);
    size_t n = (size_t)W*H*3;
    if(std::fwrite(data,1,n,f)!=n){ std::fclose(f); return false; }
    std::fclose(f);
    return true;
}

/* Small helpers: choose loader/saver by extension */
static bool has_ext(const std::string& p, const char* ext){
    if(p.size()<3) return false;
    std::string e = p.substr(p.find_last_of('.')+1);
    std::transform(e.begin(), e.end(), e.begin(), ::tolower);
    return e == ext;
}

static bool load_color_any(const std::string& path, ImageU8& img) {
    if(has_ext(path,"bmp")) return load_bmp(path, img);
    if(has_ext(path,"ppm")) return load_ppm(path, img);
    std::cerr << "Unsupported color image format for: " << path << "\n";
    return false;
}

static bool save_color_any(const std::string& path, const ImageU8& img) {
    if(has_ext(path,"ppm")) return save_ppm_u8(path, img.w, img.h, &img.data[0]);
    // (No PNG writer without external libs)
    std::cerr << "Unsupported save format (use .ppm): " << path << "\n";
    return false;
}


/* ------------------------------------------------------------
   === ADD: Alpha composite (foreground over background) ===
   observation: 3ch (RGB/BGR bytes), alpha: 1ch (8-bit), background: 3ch
   Inserts observation at (posY, posX) in background, with cropping.
   Writes result into 'composed' (same size as background).
   ------------------------------------------------------------ */
static bool alpha_composite_over_background(
    const ImageU8& observation,   // foreground RGB (no padding; same as original)
    const ImageU8& alpha_u8,      // 1ch alpha (0..255), same W/H as observation
    const ImageU8& background,    // RGB background (may be larger)
    int posY, int posX,
    ImageU8& composed              // output RGB, same size as background
){
    if(observation.c!=3 || alpha_u8.c!=1 || background.c!=3) {
        std::cerr << "alpha_composite_over_background: channel mismatch\n";
        return false;
    }
    composed = background; // start from background

    const int H = observation.h;
    const int W = observation.w;
    const int BH = background.h;
    const int BW = background.w;

    for(int i=0;i<H;++i){
        int by = posY + i;
        if(by<0 || by>=BH) continue;
        const unsigned char* src = &observation.data[i*W*3];
        const unsigned char* arow = &alpha_u8.data[i*W];
        unsigned char* dst = &composed.data[by*BW*3];
        for(int j=0;j<W;++j){
            int bx = posX + j;
            if(bx<0 || bx>=BW) continue;
            float a = (float)arow[j] * (1.0f/255.0f);
            const int s3 = 3*j;
            const int d3 = 3*bx;
            // dst = a * src + (1-a) * dst
            for(int ch=0; ch<3; ++ch){
                float v = a * (float)src[s3+ch] + (1.0f-a) * (float)dst[d3+ch];
                int vi = (int)std::floor(v + 0.5f);
                if(vi<0) vi=0; else if(vi>255) vi=255;
                dst[d3+ch] = (unsigned char)vi;
            }
        }
    }
    return true;
}


/* ------------------------------------------------------------
   Main (updated): optional composite step
   ------------------------------------------------------------ */
int main(int argc, char** argv){
    if(argc < 4){
        std::cerr << "Usage:\n  " << (argc>0?argv[0]:"bayes_matting_nocv")
                  << " <observation.bmp|ppm> <trimap.bmp> <out_alpha.pgm>"
                  << " [background.bmp|ppm] [out_composite.ppm] [posY posX]\n";
        return 1;
    }
    std::string observation_path = argv[1];
    std::string trimap_path      = argv[2];
    std::string out_alpha_path   = argv[3];

    ImageU8 observation, trimap;
    if(!load_color_any(observation_path, observation)){
        std::cerr << "Failed to load observation: " << observation_path << "\n"; return 1;
    }
    if(!load_bmp(trimap_path, trimap)){ // trimap loader expects BMP 8-bit; adapt if you need PGM
        std::cerr << "Failed to load trimap (8-bit BMP expected): " << trimap_path << "\n"; return 1;
    }
    if(observation.c != 3){
        std::cerr << "Observation must be 3-channel (RGB) image.\n"; return 1;
    }
    if(trimap.c != 1){
        std::cerr << "Trimap must be 1-channel (8-bit) BMP.\n"; return 1;
    }
    if(observation.w != trimap.w || observation.h != trimap.h){
        std::cerr << "Observation and trimap size mismatch.\n"; return 1;
    }

    // 1) Compute alpha matte
    ImageU8 alpha = bayers_matting(observation, trimap,
                                   /*single=*/false,
                                   /*radius=*/12,
                                   /*sigma=*/8.0f,
                                   /*min_cluster_num=*/8,
                                   /*max_iterations=*/37,
                                   /*sigma_c=*/0.01f);

    if(!save_pgm_u8(out_alpha_path, alpha.w, alpha.h, &alpha.data[0])){
        std::cerr << "Failed to save alpha: " << out_alpha_path << "\n"; return 1;
    }
    std::cout << "Wrote alpha matte: " << out_alpha_path << "\n";

    // 2) Optional: composite over background if provided
    if(argc >= 6){
        std::string background_path   = argv[4];
        std::string out_composite_ppm = argv[5];
        int posY = 0, posX = 0;
        if(argc >= 8){
            posY = std::atoi(argv[6]);
            posX = std::atoi(argv[7]);
        } else {
            // Default to the position used in your snippet
            posY = 544; posX = 20;
        }

        ImageU8 background;
        if(!load_color_any(background_path, background)){
            std::cerr << "Failed to load background: " << background_path << "\n"; return 1;
        }
        if(background.c != 3){
            std::cerr << "Background must be 3-channel color image.\n"; return 1;
        }

        ImageU8 composed;
        if(!alpha_composite_over_background(observation, alpha, background, posY, posX, composed)){
            std::cerr << "Compositing failed.\n"; return 1;
        }

        if(!save_color_any(out_composite_ppm, composed)){
            std::cerr << "Failed to save composite (use .ppm extension): " << out_composite_ppm << "\n"; return 1;
        }
        std::cout << "Wrote composite: " << out_composite_ppm << "\n";
    }

    return 0;
}

