	///////////////////////////////////////////////////////////////////////////////////////////////////////

	/*
    Criminisi-style Exemplar-Based Fill (VS2010, C++03, no third-party libs)

    Usage:
      CriminisiFill_VS2010.exe input.ppm mask.pgm output.ppm [patch_size]

    I/O:
      - input.ppm: PPM (P6), 8-bit RGB
      - mask.pgm : PGM (P5), 8-bit, 0=target/unknown, 255=known
      - output.ppm: PPM (P6), 8-bit RGB

    Notes:
      - Implements confidence & data terms per Criminisi et al. (2004).
      - Isophote computed on grayscale (luminance) patch; normal from mask gradient.
      - CIE Lab conversion (D65) used to compare patch similarity.

      This is a practical, no-dependency translation of your Python logic:
        - computeC / computeD → priority = C * D
        - findBestPoint
        - findBestExemplar (masked Lab distance)
        - fill & update confidence and mask
*/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cctype>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <map>          // used for sparse aggregation (via sorted triplets)
#include <limits>

#include "GEOTIFF.H"
#include "ERMapper_64.H"

#ifndef OMP
#define OMP
#endif

#ifdef OMP
#include <omp.h>
#endif

using namespace std;

// ---------- simple images ----------
struct ImageGray {
    int w, h;
    vector<unsigned char> data;
    ImageGray() : w(0), h(0), data() {}
    inline unsigned char& at(int y, int x) { return data[y * w + x]; }
    inline const unsigned char& at(int y, int x) const { return data[y * w + x]; }
    inline bool empty() const { return w <= 0 || h <= 0 || data.empty(); }
};

struct ImageRGB {
    int w, h;
    vector<unsigned char> data; // RGB
    ImageRGB() : w(0), h(0), data() {}
    inline unsigned char* pix(int y, int x) { return &data[(y * w + x) * 3]; }
    inline const unsigned char* pix(int y, int x) const { return &data[(y * w + x) * 3]; }
    inline bool empty() const { return w <= 0 || h <= 0 || data.empty(); }
};


// ---------------------- Utilities ----------------------
static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
//static inline double clampd(double v, double lo, double hi) { return v < lo ? lo : (v > hi ? hi : v); } // defined above
static inline unsigned char u8(int v) { return (unsigned char)(v < 0 ? 0 : (v > 255 ? 255 : v)); }
//static inline int iround_pos(double x) { return (int)floor(x + 0.5); } // for x>=0 // defined above

struct ImageLab {
    int w, h;
    vector<double> data; // L,a,b in planes (or interleaved)
    ImageLab(): w(0), h(0) {}
    inline double* pix(int y, int x) { return &data[(y * w + x) * 3]; }
    inline const double* pix(int y, int x) const { return &data[(y * w + x) * 3]; }
};


// ---------------------- sRGB <-> Lab (D65) ----------------------
// sRGB byte -> linear [0..1]
static inline double srgb_to_linear(double u8v01) {
    if (u8v01 <= 0.04045) return u8v01 / 12.92;
    return pow((u8v01 + 0.055) / 1.055, 2.4);
}
// linear [0..1] -> sRGB byte domain [0..1]
static inline double linear_to_srgb(double v) {
    if (v <= 0.0031308) return 12.92 * v;
    return 1.055 * pow(v, 1.0/2.4) - 0.055;
}
static void rgb_to_lab(const unsigned char* rgb, double* Lab) {
    // 1) sRGB -> linear
    double r = srgb_to_linear(rgb[0] / 255.0);
    double g = srgb_to_linear(rgb[1] / 255.0);
    double b = srgb_to_linear(rgb[2] / 255.0);
    // 2) linear RGB -> XYZ (D65)
    double X = 0.4124564*r + 0.3575761*g + 0.1804375*b;
    double Y = 0.2126729*r + 0.7151522*g + 0.0721750*b;
    double Z = 0.0193339*r + 0.1191920*g + 0.9503041*b;
    // 3) normalize by reference white D65
    const double Xn = 0.95047, Yn = 1.00000, Zn = 1.08883;
    double x = X / Xn, y = Y / Yn, z = Z / Zn;
    // 4) f(t)
    const double eps = 216.0/24389.0;  // ≈0.008856
    const double kappa = 24389.0/27.0; // ≈903.3
    double fx, fy, fz;
    fx = (x > eps) ? pow(x, 1.0/3.0) : ( (kappa*x + 16.0)/116.0 );
    fy = (y > eps) ? pow(y, 1.0/3.0) : ( (kappa*y + 16.0)/116.0 );
    fz = (z > eps) ? pow(z, 1.0/3.0) : ( (kappa*z + 16.0)/116.0 );
    // 5) Lab
    Lab[0] = 116.0*fy - 16.0;      // L* in [0..100]
    Lab[1] = 500.0*(fx - fy);      // a*
    Lab[2] = 200.0*(fy - fz);      // b*
}
static void lab_to_rgb(const double* Lab, unsigned char* rgb) {
    // 1) Lab -> f
    double L = Lab[0], a = Lab[1], b = Lab[2];
    const double eps = 216.0/24389.0;
    const double kappa = 24389.0/27.0;
    double fy = (L + 16.0)/116.0;
    double fx = fy + (a/500.0);
    double fz = fy - (b/200.0);
    // 2) f -> XYZ normalized
    double fx3 = fx*fx*fx, fy3 = fy*fy*fy, fz3 = fz*fz*fz;
    double x = (fx3 > eps) ? fx3 : ( (116.0*fx - 16.0)/kappa );
    double y = (fy3 > eps) ? fy3 : ( (116.0*fy - 16.0)/kappa );
    double z = (fz3 > eps) ? fz3 : ( (116.0*fz - 16.0)/kappa );
    // 3) denormalize XYZ
    const double Xn = 0.95047, Yn = 1.00000, Zn = 1.08883;
    double X = x * Xn;
    double Y = y * Yn;
    double Z = z * Zn;
    // 4) XYZ -> linear RGB
    double rl =  3.2404542*X + (-1.5371385)*Y + (-0.4985314)*Z;
    double gl = -0.9692660*X +  1.8760108*Y +  0.0415560*Z;
    double bl =  0.0556434*X + (-0.2040259)*Y +  1.0572252*Z;
    // 5) linear -> sRGB
    double r = linear_to_srgb(rl);
    double g = linear_to_srgb(gl);
    double b2 = linear_to_srgb(bl);
    rgb[0] = u8(iround_pos(clampd(r,0.0,1.0)*255.0));
    rgb[1] = u8(iround_pos(clampd(g,0.0,1.0)*255.0));
    rgb[2] = u8(iround_pos(clampd(b2,0.0,1.0)*255.0));
}

// Convert entire image
static void rgb_to_lab_image(const ImageRGB& rgb, ImageLab& lab) {
    lab.w = rgb.w; lab.h = rgb.h; lab.data.resize((size_t)rgb.w*rgb.h*3);
    int y, x;
    for (y=0;y<rgb.h;++y) for (x=0;x<rgb.w;++x) {
        const unsigned char* p = rgb.pix(y,x);
        double* q = lab.pix(y,x);
        rgb_to_lab(p, q);
    }
}
static void lab_to_rgb_image(const ImageLab& lab, ImageRGB& rgb) {
    rgb.w = lab.w; rgb.h = lab.h; rgb.data.resize((size_t)lab.w*lab.h*3);
    int y, x;
    for (y=0;y<lab.h;++y) for (x=0;x<lab.w;++x) {
        const double* p = lab.pix(y,x);
        unsigned char* q = rgb.pix(y,x);
        lab_to_rgb(p, q);
    }
}

// ---------------------- Grayscale from RGB ----------------------
static void rgb_to_gray(const ImageRGB& rgb, ImageGray& gray) {
    gray.w = rgb.w; gray.h = rgb.h; gray.data.resize((size_t)rgb.w*rgb.h);
    int y,x;
    for (y=0;y<rgb.h;++y) for (x=0;x<rgb.w;++x) {
        const unsigned char* p = rgb.pix(y,x);
        // ITU-R BT.709 luma approximation
        int g = (int)(0.2126*p[0] + 0.7152*p[1] + 0.0722*p[2] + 0.5);
        gray.at(y,x) = u8(g);
    }
}

// ---------------------- Neighborhood helpers ----------------------
struct Point { int y, x; Point(): y(0),x(0){} Point(int yy,int xx):y(yy),x(xx){} };

static bool extract_patch_gray(const ImageGray& img, int cy, int cx, int size,
                               vector<unsigned char>& patch, int& rows, int& cols)
{
    int r = size/2;
    rows = 0; cols = 0;
    patch.clear();
    int y0 = cy - r, x0 = cx - r;
    int y1 = cy + r, x1 = cx + r;
    int y, x;
    for (y = y0; y <= y1; ++y) {
        bool any = false;
        for (x = x0; x <= x1; ++x) {
            if (y>=0 && y<img.h && x>=0 && x<img.w) {
                patch.push_back(img.at(y,x));
                any = true;
            } else {
                patch.push_back(0);
            }
        }
        if (any) rows++;
    }
    cols = size;
    return rows == size && cols == size;
}

static bool extract_patch_mask(const ImageGray& mask, int cy, int cx, int size,
                               vector<unsigned char>& patch, int& rows, int& cols)
{
    return extract_patch_gray(mask, cy, cx, size, patch, rows, cols);
}

static bool extract_patch_lab(const ImageLab& img, int cy, int cx, int size,
                              vector<double>& patch, int& rows, int& cols)
{
    int r = size/2;
    rows = 0; cols = 0;
    patch.clear();
    int y0 = cy - r, x0 = cx - r;
    int y1 = cy + r, x1 = cx + r;
    int y, x, k;
    for (y = y0; y <= y1; ++y) {
        bool any = false;
        for (x = x0; x <= x1; ++x) {
            if (y>=0 && y<img.h && x>=0 && x<img.w) {
                const double* p = img.pix(y,x);
                for (k=0;k<3;++k) patch.push_back(p[k]);
                any = true;
            } else {
                patch.push_back(0.0); patch.push_back(0.0); patch.push_back(0.0);
            }
        }
        if (any) rows++;
    }
    cols = size;
    return rows == size && cols == size;
}

static void paste_patch_lab(ImageLab& img, int cy, int cx, int size, const vector<double>& patch) {
    int r = size/2;
    int y0 = cy - r, x0 = cx - r;
    int y, x, k, idx=0;
    for (y=y0;y<=cy+r;++y) {
        for (x=x0;x<=cx+r;++x) {
            if (y>=0 && y<img.h && x>=0 && x<img.w) {
                double* p = img.pix(y,x);
                for (k=0;k<3;++k) p[k] = patch[idx++];
            } else {
                idx += 3;
            }
        }
    }
}

// ---------------------- Gradients / Data term ----------------------
// Central differences on a gray patch (size x size), returns grad at pixel of max magnitude
static void max_isophote_on_patch(const vector<unsigned char>& patch, int size,
                                  double& iso_y, double& iso_x)
{
    // Convert to double
    int N = size*size;
    vector<double> P(N, 0.0);
    int i;
    for (i=0;i<N;++i) P[i] = (double)patch[i];

    // Compute central differences (skip border inside the patch)
    double maxMag = -1.0;
    int maxIdx = -1;
    int y, x;
    for (y=1; y<size-1; ++y) {
        for (x=1; x<size-1; ++x) {
            double gx = (P[y*size + (x+1)] - P[y*size + (x-1)]) * 0.5;
            double gy = (P[(y+1)*size + x] - P[(y-1)*size + x]) * 0.5;
            double mag = gx*gx + gy*gy;
            if (mag > maxMag) { maxMag = mag; maxIdx = y*size + x; }
        }
    }
    if (maxIdx >= 0) {
        int my = maxIdx / size;
        int mx = maxIdx % size;
        double gx = (P[my*size + (mx+1)] - P[my*size + (mx-1)]) * 0.5;
        double gy = (P[(my+1)*size + mx] - P[(my-1)*size + mx]) * 0.5;
        // isophote is perpendicular to gradient: [gy, -gx]
        double len = sqrt(gy*gy + gx*gx);
        if (len < 1e-12) { iso_y = 0.0; iso_x = 0.0; }
        else { iso_y = gy/len; iso_x = -gx/len; }
    } else {
        iso_y = 0.0; iso_x = 0.0;
    }
}

// Normal from mask gradient at pixel p (using central differences)
// mask: 0 (target), 255 (known). Normal points from unknown to known roughly.
static void mask_normal(const ImageGray& mask, int y, int x, double& ny, double& nx) {
    // Compute gradient of mask (treat outside as 255 to force inward normals near borders)
    int xm1 = x-1, xp1 = x+1, ym1 = y-1, yp1 = y+1;
    unsigned char l = (xm1>=0) ? mask.at(y, xm1) : 255;
    unsigned char r = (xp1<mask.w) ? mask.at(y, xp1) : 255;
    unsigned char u = (ym1>=0) ? mask.at(ym1, x) : 255;
    unsigned char d = (yp1<mask.h) ? mask.at(yp1, x) : 255;
    double gx = (double)r - (double)l; // d/dx
    double gy = (double)d - (double)u; // d/dy
    double len = sqrt(gx*gx + gy*gy);
    if (len < 1e-12) { ny = 0.0; nx = 0.0; }
    else { ny = gy/len; nx = gx/len; }
}

// ---------------------- Fill front extraction ----------------------
// Extract boundary points where mask == 0 and any 4-neighbor is 255
static void find_fill_front(const ImageGray& mask, vector<Point>& front) {
    front.clear();
    int y, x;
    for (y=1;y<mask.h-1;++y) {
        for (x=1;x<mask.w-1;++x) {
            if (mask.at(y,x) == 0) {
                if (mask.at(y-1,x)==255 || mask.at(y+1,x)==255 ||
                    mask.at(y,x-1)==255 || mask.at(y,x+1)==255)
                {
                    front.push_back(Point(y,x));
                }
            }
        }
    }
}

// ---------------------- Confidence & Data computation ----------------------
static void compute_confidence(const vector<Point>& front,
                               ImageGray& mask_init, // initial known map (unchanged)
                               const ImageGray& mask_now, // current mask (for edges)
                               int size,
                               vector<double>& confidence) // per-pixel [0..1]
{
    // confidence[y*w + x] is already maintained for known region
    int r = size/2;
    int i;
    for (i=0;i<(int)front.size();++i) {
        int y = front[i].y, x = front[i].x;
        int y0 = y - r, y1 = y + r;
        int x0 = x - r, x1 = x + r;
        int yy, xx;
        double sum = 0.0; int cnt = 0;
        for (yy=y0; yy<=y1; ++yy) {
            for (xx=x0; xx<=x1; ++xx) {
                if (yy>=0 && yy<mask_now.h && xx>=0 && xx<mask_now.w) {
                    if (mask_init.at(yy,xx) == 255) {
                        sum += confidence[yy*mask_now.w + xx];
                        cnt++;
                    }
                }
            }
        }
        if (cnt > 0) confidence[y*mask_now.w + x] = sum / (double)cnt;
        else confidence[y*mask_now.w + x] = 0.0;
    }
}

static void compute_data(const vector<Point>& front,
                         const ImageRGB& img_rgb, // for grayscale isophote
                         const ImageGray& mask_now,
                         int size,
                         vector<double>& data) // per-pixel [0..1]
{
    ImageGray gray;
    rgb_to_gray(img_rgb, gray);

    int i, rows, cols;
    for (i=0;i<(int)front.size();++i) {
        int y = front[i].y, x = front[i].x;
        vector<unsigned char> patch;
        if (!extract_patch_gray(gray, y, x, size, patch, rows, cols) || rows!=size || cols!=size) {
            data[y*gray.w + x] = 0.0;
            continue;
        }
        double iso_y=0.0, iso_x=0.0;
        max_isophote_on_patch(patch, size, iso_y, iso_x);

        double ny=0.0, nx=0.0;
        mask_normal(mask_now, y, x, ny, nx);

        double d = fabs(iso_y*ny + iso_x*nx);
        // scale roughly into [0..1]
        if (d > 1.0) d = 1.0;
        data[y*gray.w + x] = d;
    }
}

// ---------------------- Priority & selection ----------------------
static Point find_best_point(const vector<Point>& front,
                             const vector<double>& confidence,
                             const vector<double>& data,
                             int W)
{
    double bestP = -1.0;
    Point best(-1,-1);
    int i;
    for (i=0;i<(int)front.size();++i) {
        int y = front[i].y, x = front[i].x;
        double pr = confidence[y*W + x] * data[y*W + x];
        if (pr > bestP) { bestP = pr; best = front[i]; }
    }
    return best;
}

// ---------------------- Exemplar search ----------------------
// Computes masked L2 distance between Lab patches.
// Only positions with mask==255 in the target's neighborhood are compared.
static bool best_exemplar(const Point& p,
                          const ImageLab& lab,
                          const ImageGray& maskEdgeCase, // mask with borders set to known
                          int size,
                          Point& best_q,
                          vector<double>& best_patch)
{
    int r = size/2;
    int H = lab.h, W = lab.w;
    int y, x, k;
    int rows, cols;

    // Neighbor mask for p
    vector<unsigned char> pMaskPatch; // 0/255
    if (!extract_patch_mask(maskEdgeCase, p.y, p.x, size, pMaskPatch, rows, cols)) return false;

    // Target patch (Lab), masked
    vector<double> pPatch; // Lab interleaved
    if (!extract_patch_lab(lab, p.y, p.x, size, pPatch, rows, cols)) return false;

    double bestD = std::numeric_limits<double>::infinity();
    best_q = Point(-1,-1);
    best_patch.clear();
    // stride ~ size/9 (>=1)
    int stride = size / 9; if (stride < 1) stride = 1;

    for (y=r+1; y<H-(r+2); y+=stride) {
        for (x=r+1; x<W-(r+2); x+=stride) {
            // Skip if candidate patch overlaps target (any mask==0)
            bool overlapsTarget = false;
            int yy, xx;
            for (yy=y-r; yy<=y+r; ++yy) {
                for (xx=x-r; xx<=x+r; ++xx) {
                    if (maskEdgeCase.at(yy,xx) == 0) { overlapsTarget = true; break; }
                }
                if (overlapsTarget) break;
            }
            if (overlapsTarget) continue;

            // Candidate patch
            vector<double> qPatch;
            if (!extract_patch_lab(lab, y, x, size, qPatch, rows, cols)) continue;

            // Distance on known positions of p's neighborhood mask
            double acc = 0.0; int cnt = 0;
            int idx_p = 0, idx_q = 0;
            int wy, wx;
            for (wy=0; wy<size; ++wy) {
                for (wx=0; wx<size; ++wx) {
                    unsigned char m = pMaskPatch[wy*size + wx];
                    if (m == 255) {
                        // 3 channels
                        for (k=0;k<3;++k) {
                            double d = pPatch[idx_p + k] - qPatch[idx_q + k];
                            acc += d*d;
                        }
                        cnt += 3;
                    }
                    idx_p += 3;
                    idx_q += 3;
                }
            }
            if (cnt == 0) continue; // nothing to compare (shouldn't happen)

            double D = sqrt(acc); // Euclidean norm over masked positions
            if (D < bestD) {
                bestD = D;
                best_q = Point(y,x);
                best_patch = qPatch;
            }
        }
    }
    return (best_q.y >= 0);
}

// ---------------------- Fill update ----------------------
static void update_after_fill(const Point& p, const vector<double>& exemplar_patch,
                              int size,
                              ImageLab& lab,
                              ImageRGB& rgb,
                              ImageGray& mask, // current mask: set filled region to 255
                              vector<double>& confidence) // per-pixel confidence
{
    int r = size/2;
    // Paste into Lab image
    paste_patch_lab(lab, p.y, p.x, size, exemplar_patch);
    // Update RGB for visualization / next grayscale (optional now or after loop)
    lab_to_rgb_image(lab, rgb);

    // Update confidence in the patch to center's confidence
    double C = confidence[p.y*mask.w + p.x];
    int y, x;
    for (y=p.y-r; y<=p.y+r; ++y) {
        for (x=p.x-r; x<=p.x+r; ++x) {
            if (y>=0 && y<mask.h && x>=0 && x<mask.w) {
                confidence[y*mask.w + x] = C;
                // Mark as known
                mask.at(y,x) = 255;
            }
        }
    }
}

// Make a copy with edges forced to known to avoid boundary issues
static void mask_with_edges_known(const ImageGray& mask, int r, ImageGray& out) {
    out = mask;
    int y, x;
    for (y=0;y<out.h;++y) {
        for (x=0;x<out.w;++x) {
            if (y<=(r) || y>=out.h-(r+1) || x<=(r) || x>=out.w-(r+1)) {
                out.at(y,x) = 255;
            }
        }
    }
}

// ---------------------- Main solve ----------------------
static bool run_fill(ImageRGB& img_rgb, const ImageGray& mask_in, ImageRGB& out_rgb, int size) {
    if (size < 3) size = 3;
    if ((size & 1) == 0) size += 1; // odd

    // Working copies
    ImageGray mask = mask_in; // 0=unknown, 255=known
    ImageLab img_lab;
    rgb_to_lab_image(img_rgb, img_lab);

    // Confidence map (double), initialized from mask (known→1, unknown→0)
    vector<double> confidence((size_t)mask.w*mask.h, 0.0);
    int i, N = mask.w*mask.h;
    for (i=0;i<N;++i) confidence[i] = (mask.data[i] == 255) ? 1.0 : 0.0;

    // Initial known map (unchanged) for confidence computation
    ImageGray mask_init = mask;

    int r = size/2;
    ImageGray maskEdgeCase;
    mask_with_edges_known(mask, r+1, maskEdgeCase);

    int iteration = 0;
    const int max_iters = mask.w * mask.h; // cap

    while (iteration < max_iters) {
        // 1) Fill front
        vector<Point> front;
        find_fill_front(mask, front);
        if (front.empty()) {
            // Done: no boundary of unknown region remains
            break;
        }

        // 2) Compute priorities
        vector<double> data((size_t)mask.w*mask.h, 0.0);
        compute_confidence(front, mask_init, mask, size, confidence);
        compute_data(front, img_rgb, mask, size, data);

        // 3) Select best point
        Point p = find_best_point(front, confidence, data, mask.w);
        if (p.y < 0) {
            cerr << "No valid fill point found. Exiting.\n";
            break;
        }

        // 4) Best exemplar search (masked Lab distance)
        Point q; vector<double> qpatch;
        if (!best_exemplar(p, img_lab, maskEdgeCase, size, q, qpatch)) {
            cerr << "No valid exemplar found. Exiting.\n";
            break;
        }

        // 5) Copy and update
        update_after_fill(p, qpatch, size, img_lab, img_rgb, mask, confidence);

        // 6) Keep maskEdgeCase consistent (force border known)
        mask_with_edges_known(mask, r+1, maskEdgeCase);

        iteration++;
        if ((iteration % 10) == 0) {
            cout << "Iteration " << iteration << " (front=" << front.size() << ")\n";
        }
    }

    // Output final RGB
    out_rgb = img_rgb;
    return true;
}

// https://csaishih.github.io/projects/inpainting
// based on Python code from https://github.com/csaishih/photo-acne-remover-image-inpainting
// resize: https://github.com/csaishih/seam-carving
// foreground and background seperation https://csaishih.github.io/projects/foregroundextraction.html

// summary about inpainting https://github.com/AlonzoLeeeooo/awesome-image-inpainting-studies

	int inpaintCriminisi(int argc, char** argv)
	{
    if (argc < 4 || argc > 5) {
        cout << "Usage:\n  " << argv[0] << " input.ppm mask.pgm output.ppm [patch_size]\n"
             << "  - mask.pgm: 0 = target (to fill), 255 = known\n";
        return 0;
    }
    string in_img = argv[1];
    string in_mask = argv[2];
    string out_img = argv[3];
    int patch_size = 0;
    if (argc == 5) {
        patch_size = atoi(argv[4]);
        if (patch_size < 3) patch_size = 3;
        if ((patch_size & 1) == 0) patch_size += 1; // odd
    }

    ImageRGB img;
    ImageGray mask;
    if (!load_ers_rgb_8u(in_img, img)) return 1;
    if (!load_ers_grey_8u(in_mask, mask)) return 1;
    if (img.w != mask.w || img.h != mask.h) {
        cerr << "Image and mask dimensions mismatch\n";
        return 1;
		}

    if (patch_size == 0) {
        // Similar to your heuristic: min(H,W)/30 rounded to nearest odd
        int base = (min(img.w, img.h) / 30);
        if (base < 1) base = 1;
        patch_size = base | 1; // force odd
        if (patch_size < 3) patch_size = 3;
    }
    cout << "Patch size: " << patch_size << "\n";

    ImageRGB out;
    if (!run_fill(img, mask, out, patch_size)) {
        cerr << "Inpainting failed.\n";
        return 1;
    }

	if (!save_ers_rgb_u8(in_img,out_img, out)) {
        cerr << "Failed to save output: " << out_img << "\n";
        return 1;
    }

    cout << "Saved: " << out_img << "\n";
    return 0;
}
