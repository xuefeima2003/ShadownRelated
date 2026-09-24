/* orchard_bouman.c
 * ANSI C (C89) Orchard–Bouman-like clustering + PGM example loader.
 *
 * Features per pixel: (x, y, I) normalized to [0,1].
 * Stops splitting when the largest principal covariance eigenvalue <= minVar.
 *
 * Build:
 *   gcc -O2 -std=c89 -Wall -Wextra -o ob_cluster orchard_bouman.c
 *
 * Run:
 *   ./ob_cluster image.pgm [minVar]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

#ifndef DBL_EPSILON
#include <float.h>
#endif

/* ==========================
   Memory helpers
   ========================== */
static void *xmalloc(size_t n)
{
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "Out of memory (%lu bytes)\n", (unsigned long)n); exit(1); }
    return p;
}
static void *xrealloc(void *ptr, size_t n)
{
    void *p = realloc(ptr, n);
    if (!p) { fprintf(stderr, "Out of memory (realloc %lu bytes)\n", (unsigned long)n); exit(1); }
    return p;
}

/* ==========================
   Small math helpers
   ========================== */
static double dot3(const double *a, const double *b)
{
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}
static void add_outer_weighted(double C[3][3], const double *v, double w)
{
    C[0][0] += w * v[0] * v[0];
    C[0][1] += w * v[0] * v[1];
    C[0][2] += w * v[0] * v[2];
    C[1][0] += w * v[1] * v[0];
    C[1][1] += w * v[1] * v[1];
    C[1][2] += w * v[1] * v[2];
    C[2][0] += w * v[2] * v[0];
    C[2][1] += w * v[2] * v[1];
    C[2][2] += w * v[2] * v[2];
}
static void mat_add_eye(double A[3][3], double s)
{
    A[0][0] += s; A[1][1] += s; A[2][2] += s;
}
static void copy_mat3(double dst[3][3], const double src[3][3])
{
    int i, j;
    for (i = 0; i < 3; ++i) for (j = 0; j < 3; ++j) dst[i][j] = src[i][j];
}
static double normalize3(double *v)
{
    double n = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    if (n > 0.0) { v[0] /= n; v[1] /= n; v[2] /= n; }
    return n;
}

/* ==========================
   Jacobi eigen-solver for 3x3 symmetric
   ========================== */
static void jacobi_eigen_3x3(double A[3][3], double evec[3][3], double eval[3])
{
    double V[3][3];
    double a01, a02, a12, app, apq, aqq, tau, t, c, s;
    int i, j, k, iter, p, q;

    V[0][0] = 1.0; V[0][1] = 0.0; V[0][2] = 0.0;
    V[1][0] = 0.0; V[1][1] = 1.0; V[1][2] = 0.0;
    V[2][0] = 0.0; V[2][1] = 0.0; V[2][2] = 1.0;

    for (iter = 0; iter < 50; ++iter) {
        a01 = fabs(A[0][1]);
        a02 = fabs(A[0][2]);
        a12 = fabs(A[1][2]);

        p = 0; q = 1;
        if (a02 > a01 && a02 >= a12) { p = 0; q = 2; }
        else if (a12 >= a01 && a12 >= a02) { p = 1; q = 2; }

        if (fabs(A[p][q]) < 1e-12) break;

        app = A[p][p];
        aqq = A[q][q];
        apq = A[p][q];

        if (fabs(aqq - app) < 1e-30) {
            c = 1.0 / sqrt(2.0);
            s = (apq >= 0.0) ? c : -c;
        } else {
            tau = (aqq - app) / (2.0 * apq);
            t = (tau >= 0.0) ? 1.0 / (tau + sqrt(1.0 + tau*tau))
                             : -1.0 / (-tau + sqrt(1.0 + tau*tau));
            c = 1.0 / sqrt(1.0 + t*t);
            s = t * c;
        }

        for (k = 0; k < 3; ++k) {
            double Akp = A[k][p], Akq = A[k][q];
            A[k][p] = c*Akp - s*Akq;
            A[k][q] = s*Akp + c*Akq;
        }
        for (k = 0; k < 3; ++k) {
            double Apk = A[p][k], Aqk = A[q][k];
            A[p][k] = c*Apk - s*Aqk;
            A[q][k] = s*Apk + c*Aqk;
        }

        A[p][q] = A[q][p] = 0.0;

        for (k = 0; k < 3; ++k) {
            double Vkp = V[k][p], Vkq = V[k][q];
            V[k][p] = c*Vkp - s*Vkq;
            V[k][q] = s*Vkp + c*Vkq;
        }
    }

    eval[0] = A[0][0]; eval[1] = A[1][1]; eval[2] = A[2][2];

    for (i = 0; i < 3; ++i) for (j = 0; j < 3; ++j) evec[i][j] = V[i][j];
    for (j = 0; j < 3; ++j) {
        double col[3];
        col[0] = evec[0][j]; col[1] = evec[1][j]; col[2] = evec[2][j];
        normalize3(col);
        evec[0][j] = col[0]; evec[1][j] = col[1]; evec[2][j] = col[2];
    }
}

/* ==========================
   3x3 matrix inverse and log-determinant (row-major)
   ========================== */

/* Compute inverse of 3x3 matrix A into invA, and return determinant.
   Returns 0 on success, nonzero if singular (near-zero det). */
static int invert3x3(const double A[9], double invA[9], double *det_out)
{
    /* Row-major: A = [a00 a01 a02; a10 a11 a12; a20 a21 a22] */
    double a00=A[0], a01=A[1], a02=A[2];
    double a10=A[3], a11=A[4], a12=A[5];
    double a20=A[6], a21=A[7], a22=A[8];

    double c00 =  (a11*a22 - a12*a21);
    double c01 = -(a10*a22 - a12*a20);
    double c02 =  (a10*a21 - a11*a20);
    double c10 = -(a01*a22 - a02*a21);
    double c11 =  (a00*a22 - a02*a20);
    double c12 = -(a00*a21 - a01*a20);
    double c20 =  (a01*a12 - a02*a11);
    double c21 = -(a00*a12 - a02*a10);
    double c22 =  (a00*a11 - a01*a10);

    double det = a00*c00 + a01*c01 + a02*c02;

    if (det_out) *det_out = det;
    if (fabs(det) < 1e-15) return 1; /* singular */

    /* invA = (1/det) * adj(A) = (1/det) * C^T */
    invA[0] = c00 / det; invA[1] = c10 / det; invA[2] = c20 / det;
    invA[3] = c01 / det; invA[4] = c11 / det; invA[5] = c21 / det;
    invA[6] = c02 / det; invA[7] = c12 / det; invA[8] = c22 / det;
    return 0;
}

/* Mahalanobis squared distance (x - mu)^T invS (x - mu); x,mu length 3; invS 3x3 row-major */
static double mahal_sq_3(const double *x, const double *mu, const double *invS)
{
    double dx0 = x[0]-mu[0], dx1 = x[1]-mu[1], dx2 = x[2]-mu[2];
    double y0 = invS[0]*dx0 + invS[1]*dx1 + invS[2]*dx2;
    double y1 = invS[3]*dx0 + invS[4]*dx1 + invS[5]*dx2;
    double y2 = invS[6]*dx0 + invS[7]*dx1 + invS[8]*dx2;
    return dx0*y0 + dx1*y1 + dx2*y2;
}

/* Write an 8-bit PGM (P5) file; data length must be W*H */
static int save_pgm_u8(const char *path, int W, int H, const unsigned char *data)
{
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Failed to write %s\n", path); return -1; }
    fprintf(f, "P5\n%d %d\n255\n", W, H);
    if ((int)fwrite(data, 1, (size_t)(W*H), f) != W*H) {
        fprintf(stderr, "Short write on %s\n", path);
        fclose(f); return -1;
    }
    fclose(f);
    return 0;
}

/* ==========================
   Dataset / Node
   ========================== */
typedef struct {
    const double *X; /* n x 3 row-major */
    const double *w; /* length n */
    int n;
} Dataset;

typedef struct Node_ {
    int *idx;      /* m indices into dataset */
    int m;
    double W;
    double mu[3];
    double cov[3][3];
    double lambda; /* largest |eigenvalue| */
    double e[3];   /* corresponding eigenvector */
} Node;

static int node_init(Node *node, const Dataset *D, const int *idx, int m)
{
    int i, j;
    double W = 0.0;
    double mu[3];
    double C[3][3];
    double evec[3][3], eval[3];
    double A[3][3];

    mu[0]=mu[1]=mu[2]=0.0;
    C[0][0]=C[0][1]=C[0][2]=0.0;
    C[1][0]=C[1][1]=C[1][2]=0.0;
    C[2][0]=C[2][1]=C[2][2]=0.0;

    if (m <= 0) {
        node->idx = NULL; node->m = 0; node->W = 0.0; node->lambda = 0.0;
        node->e[0]=node->e[1]=0.0; node->e[2]=1.0;
        return 0;
    }
    node->idx = (int*)xmalloc(sizeof(int)*m);
    for (i = 0; i < m; ++i) node->idx[i] = idx[i];
    node->m = m;

    for (i = 0; i < m; ++i) {
        int r = idx[i];
        double wi = D->w[r];
        const double *xi = &D->X[3*r];
        W += wi;
        mu[0] += wi * xi[0];
        mu[1] += wi * xi[1];
        mu[2] += wi * xi[2];
    }
    if (W <= 0.0) W = 1.0;
    mu[0] /= W; mu[1] /= W; mu[2] /= W;

    for (i = 0; i < m; ++i) {
        int r = idx[i];
        double wi = D->w[r];
        const double *xi = &D->X[3*r];
        double v[3];
        v[0] = xi[0] - mu[0];
        v[1] = xi[1] - mu[1];
        v[2] = xi[2] - mu[2];
        add_outer_weighted(C, v, wi);
    }
    for (i = 0; i < 3; ++i) for (j = 0; j < 3; ++j) C[i][j] /= W;

    mat_add_eye(C, 1e-5);
    copy_mat3(A, C);
    jacobi_eigen_3x3(A, evec, eval);

    {
        int kmax = 0;
        double a0 = fabs(eval[0]), a1 = fabs(eval[1]), a2 = fabs(eval[2]);
        if (a1 > a0 && a1 >= a2) kmax = 1;
        else if (a2 >= a0 && a2 >= a1) kmax = 2;
        node->lambda = fabs(eval[kmax]);
        node->e[0] = evec[0][kmax];
        node->e[1] = evec[1][kmax];
        node->e[2] = evec[2][kmax];
    }

    node->W = W;
    node->mu[0]=mu[0]; node->mu[1]=mu[1]; node->mu[2]=mu[2];
    copy_mat3(node->cov, C);
    return 1;
}
static void node_free(Node *node)
{
    if (!node) return;
    if (node->idx) free(node->idx);
    node->idx = NULL; node->m = 0;
}

/* ==========================
   Dynamic array of Node*
   ========================== */
typedef struct {
    Node **data;
    int size;
    int cap;
} NodeVec;

static void nodevec_init(NodeVec *v) { v->data=NULL; v->size=0; v->cap=0; }
static void nodevec_push(NodeVec *v, Node *n)
{
    if (v->size == v->cap) {
        v->cap = v->cap ? v->cap*2 : 4;
        v->data = (Node**)xrealloc(v->data, v->cap * sizeof(Node*));
    }
    v->data[v->size++] = n;
}
static void nodevec_free(NodeVec *v)
{
    int i;
    if (!v) return;
    for (i = 0; i < v->size; ++i) { node_free(v->data[i]); free(v->data[i]); }
    free(v->data); v->data=NULL; v->size=v->cap=0;
}
static int argmax_lambda(const NodeVec *v)
{
    int i, imax = 0;
    double best = -1.0;
    for (i = 0; i < v->size; ++i) if (v->data[i]->lambda > best) { best = v->data[i]->lambda; imax = i; }
    return imax;
}

/* Split the node with max lambda; returns 1 if split, 0 if degenerate */
static int split_once(NodeVec *nodes, const Dataset *D)
{
    int idx_max = argmax_lambda(nodes);
    Node *C = nodes->data[idx_max];
    double t0 = dot3(C->mu, C->e);
    int m = C->m;
    int *Aidx = (int*)xmalloc(sizeof(int)*m);
    int *Bidx = (int*)xmalloc(sizeof(int)*m);
    int i, ma=0, mb=0;

    for (i = 0; i < m; ++i) {
        int r = C->idx[i];
        const double *x = &D->X[3*r];
        double proj = x[0]*C->e[0] + x[1]*C->e[1] + x[2]*C->e[2];
        if (proj <= t0) Aidx[ma++] = r; else Bidx[mb++] = r;
    }
    if (ma == 0 || mb == 0) {
        free(Aidx); free(Bidx);
        C->lambda = 0.0; /* mark unsplittable */
        return 0;
    }

    {
        Node *Ca = (Node*)xmalloc(sizeof(Node));
        Node *Cb = (Node*)xmalloc(sizeof(Node));
        if (!node_init(Ca, D, Aidx, ma) || !node_init(Cb, D, Bidx, mb)) {
            node_free(Ca); free(Ca); node_free(Cb); free(Cb);
            free(Aidx); free(Bidx);
            C->lambda = 0.0;
            return 0;
        }
        node_free(C); free(C);
        nodes->data[idx_max] = Ca;
        nodevec_push(nodes, Cb);
    }
    free(Aidx); free(Bidx);
    return 1;
}

/* ==========================
   Public API: clustFunc
   ========================== */
int clustFunc(
    const double *X, const double *w, int n, double minVar,
    double **mu_out, double **Sigma_out, int *K_out
)
{
    Dataset D;
    NodeVec nodes;
    Node *root;
    int *root_idx;
    int i, K;

    if (!X || !w || n <= 0 || !mu_out || !Sigma_out || !K_out) return 1;

    D.X = X; D.w = w; D.n = n;
    nodevec_init(&nodes);

    root_idx = (int*)xmalloc(sizeof(int)*n);
    for (i = 0; i < n; ++i) root_idx[i] = i;

    root = (Node*)xmalloc(sizeof(Node));
    if (!node_init(root, &D, root_idx, n)) {
        free(root_idx); node_free(root); free(root); nodevec_free(&nodes);
        return 2;
    }
    free(root_idx);
    nodevec_push(&nodes, root);

    for (;;) {
        int idx_max = argmax_lambda(&nodes);
        double max_lambda = nodes.data[idx_max]->lambda;
        int i2, any, did_split;

        if (max_lambda <= minVar) break;
        did_split = split_once(&nodes, &D);
        if (!did_split) {
            any = 0;
            for (i2 = 0; i2 < nodes.size; ++i2)
                if (nodes.data[i2]->lambda > minVar) { any = 1; break; }
            if (!any) break;
        }
    }

    K = nodes.size;
    *K_out = K;
    *mu_out = (double*)xmalloc(sizeof(double)*K*3);
    *Sigma_out = (double*)xmalloc(sizeof(double)*K*9);

    for (i = 0; i < K; ++i) {
        Node *C = nodes.data[i];
        double *mu = &(*mu_out)[3*i];
        double *S  = &(*Sigma_out)[9*i];
        int r;
        mu[0]=C->mu[0]; mu[1]=C->mu[1]; mu[2]=C->mu[2];
        for (r = 0; r < 3; ++r) {
            S[3*r+0] = C->cov[r][0];
            S[3*r+1] = C->cov[r][1];
            S[3*r+2] = C->cov[r][2];
        }
    }
    nodevec_free(&nodes);
    return 0;
}

/* ==========================
   Minimal PGM loader (P2/P5), returns normalized [0,1] doubles
   ========================== */

/* Read next non-comment token from stream into buf */
static int pgm_next_token(FILE *f, char *buf, int bufsz)
{
    int c, i = 0;
    if (bufsz <= 0) return -1;

    /* Skip whitespace and comments */
    c = fgetc(f);
    while (c != EOF) {
        if (isspace(c)) { c = fgetc(f); continue; }
        if (c == '#') { /* skip comment line */
            while (c != '\n' && c != '\r' && c != EOF) c = fgetc(f);
            c = fgetc(f);
            continue;
        }
        break;
    }
    if (c == EOF) return -1;

    /* Read token */
    while (c != EOF && !isspace(c) && c != '#') {
        if (i < bufsz - 1) buf[i++] = (char)c;
        c = fgetc(f);
    }
    buf[i] = '\0';

    /* If we hit a comment start, discard rest of that line */
    if (c == '#') {
        while (c != '\n' && c != '\r' && c != EOF) c = fgetc(f);
    }
    return (i > 0) ? 0 : -1;
}

/* Loads PGM and returns normalized pixel buffer (double in [0,1]) */
static int load_pgm_normalized(const char *path, int *W, int *H, double **out)
{
    FILE *f = fopen(path, "rb");
    char tok[64];
    int width, height, maxval;
    int ascii = 0; /* P2 or P5 */
    int count, i;
    double *buf;

    if (!f) { fprintf(stderr, "Failed to open %s\n", path); return -1; }

    /* Magic number */
    if (pgm_next_token(f, tok, sizeof(tok)) != 0) { fclose(f); return -1; }
    if (strcmp(tok, "P5") == 0) ascii = 0;
    else if (strcmp(tok, "P2") == 0) ascii = 1;
    else { fprintf(stderr, "Unsupported PGM magic: %s (need P2 or P5)\n", tok); fclose(f); return -1; }

    /* width, height, maxval */
    if (pgm_next_token(f, tok, sizeof(tok)) != 0) { fclose(f); return -1; }
    width = atoi(tok);
    if (pgm_next_token(f, tok, sizeof(tok)) != 0) { fclose(f); return -1; }
    height = atoi(tok);
    if (pgm_next_token(f, tok, sizeof(tok)) != 0) { fclose(f); return -1; }
    maxval = atoi(tok);
    if (width <= 0 || height <= 0 || maxval <= 0) {
        fprintf(stderr, "Invalid PGM header values\n"); fclose(f); return -1;
    }

    buf = (double*)xmalloc(sizeof(double) * width * height);

    if (ascii) {
        /* P2: ASCII integers */
        for (i = 0; i < width * height; ++i) {
            int v;
            if (pgm_next_token(f, tok, sizeof(tok)) != 0) { free(buf); fclose(f); return -1; }
            v = atoi(tok);
            if (v < 0) v = 0; if (v > maxval) v = maxval;
            buf[i] = (double)v / (double)maxval;
        }
    } else {
        /* P5: Binary data. After maxval, there is one whitespace byte then pixel stream. */
        int n = width * height;
        if (maxval <= 255) {
            unsigned char *tmp = (unsigned char*)xmalloc(n);
            size_t got = fread(tmp, 1, (size_t)n, f);
            if ((int)got != n) { fprintf(stderr, "Short read in P5 data\n"); free(tmp); free(buf); fclose(f); return -1; }
            for (i = 0; i < n; ++i) buf[i] = (double)tmp[i] / (double)maxval;
            free(tmp);
        } else if (maxval <= 65535) {
            /* Big-endian 16-bit per Netpbm spec */
            int k;
            unsigned char *tmp2 = (unsigned char*)xmalloc(n * 2);
            size_t got2 = fread(tmp2, 1, (size_t)(n*2), f);
            if ((int)got2 != n*2) { fprintf(stderr, "Short read in P5 16-bit data\n"); free(tmp2); free(buf); fclose(f); return -1; }
            for (k = 0; k < n; ++k) {
                unsigned int hi = (unsigned int)tmp2[2*k+0];
                unsigned int lo = (unsigned int)tmp2[2*k+1];
                unsigned int v = (hi << 8) | lo;
                if ((int)v > maxval) v = (unsigned int)maxval;
                buf[k] = (double)v / (double)maxval;
            }
            free(tmp2);
        } else {
            fprintf(stderr, "Unsupported maxval > 65535\n");
            free(buf); fclose(f); return -1;
        }
    }

    fclose(f);
    *W = width; *H = height; *out = buf;
    return 0;
}

/* ==========================
   Assign each pixel (x,y,I)->cluster ID using Gaussian log-likelihood
   Inputs:
     W,H: image size
     X:   features, n x 3 (row-major), here n = W*H
     mu:  K x 3 means
     S:   K x 9 covariances (row-major 3x3 each)
     K:   number of clusters
   Output:
     labels: W*H bytes (0..K-1), caller allocates
   ========================== */
static void assign_clusters(
    int W, int H,
    const double *X,
    const double *mu,
    const double *S,
    int K,
    unsigned char *labels
)
{
    const int n = W*H;
    int k, i;

    /* Precompute inv(Sigma_k) and log|Sigma_k| for each cluster */
    double *invS = (double*)xmalloc(sizeof(double) * K * 9);
    double *logdetS = (double*)xmalloc(sizeof(double) * K);
    for (k = 0; k < K; ++k) {
        const double *Sk = &S[9*k];
        double *iSk = &invS[9*k];
        double det;
        int rc = invert3x3(Sk, iSk, &det);
        if (rc != 0 || det <= 0.0) {
            /* Numerical guard: add tiny jitter on the diagonal and retry */
            double J[9];
            memcpy(J, Sk, sizeof(double)*9);
            J[0] += 1e-8; J[4] += 1e-8; J[8] += 1e-8;
            if (invert3x3(J, iSk, &det) != 0 || det <= 0.0) {
                /* Fallback: identity inverse, zero logdet */
                iSk[0]=1; iSk[1]=0; iSk[2]=0;
                iSk[3]=0; iSk[4]=1; iSk[5]=0;
                iSk[6]=0; iSk[7]=0; iSk[8]=1;
                det = 1.0;
            }
        }
        logdetS[k] = log(det);
    }

    /* For each sample, compute score and choose best cluster */
    for (i = 0; i < n; ++i) {
        const double *xi = &X[3*i];
        double best_score = 1e300;
        int best_k = 0;

        for (k = 0; k < K; ++k) {
            const double *muk = &mu[3*k];
            const double *iSk = &invS[9*k];
            /* Negative log-likelihood up to constant:
               0.5*(Mahalanobis^2 + log|Sigma|). We omit +1.5*log(2π). */
            double d2 = mahal_sq_3(xi, muk, iSk);
            double score = 0.5 * (d2 + logdetS[k]);
            if (score < best_score) { best_score = score; best_k = k; }
        }
        labels[i] = (unsigned char)best_k;
    }

    free(invS);
    free(logdetS);
}

/* ==========================
   Example main(): use PGM as dataset
   ========================== */
int main(int argc, char **argv)
{
    const char *path;
    double minVar = 0.05; /* default threshold (features in [0,1]) */
    int W, H;
    double *I = NULL;   /* normalized intensities [0,1] */
    double *X = NULL;   /* n x 3 features: (x, y, I) normalized */
    double *w = NULL;   /* weights (all ones) */
    double *mu = NULL;  /* K x 3 */
    double *S = NULL;   /* K x 9 */
    int n, i, x, y, idx, rc, K;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s image.pgm [minVar]\n", argv[0]);
        return 1;
    }
    path = argv[1];
    if (argc >= 3) minVar = atof(argv[2]);
    if (minVar <= 0.0) minVar = 0.05;

    if (load_pgm_normalized(path, &W, &H, &I) != 0) {
        fprintf(stderr, "Failed to load PGM: %s\n", path);
        return 1;
    }

    n = W * H;
    X = (double*)xmalloc(sizeof(double) * n * 3);
    w = (double*)xmalloc(sizeof(double) * n);

    /* Build features: (x, y, intensity), all normalized to [0,1] */
    idx = 0;
    for (y = 0; y < H; ++y) {
        for (x = 0; x < W; ++x) {
            double fx = (W > 1) ? ((double)x / (double)(W - 1)) : 0.0;
            double fy = (H > 1) ? ((double)y / (double)(H - 1)) : 0.0;
            double fi = I[y*W + x]; /* already in [0,1] */
            X[3*idx + 0] = fx;
            X[3*idx + 1] = fy;
            X[3*idx + 2] = fi;
            w[idx] = 1.0;
            idx++;
        }
    }

    rc = clustFunc(X, w, n, minVar, &mu, &S, &K);
    if (rc != 0) {
        fprintf(stderr, "clustFunc failed: %d\n", rc);
        free(I); free(X); free(w);
        return 1;
    }

    printf("PGM: %s  (%dx%d), n=%d samples, minVar=%.6g\n", path, W, H, n, minVar);
    printf("Clusters K = %d\n", K);

    /* Print cluster stats (means and covariances) */
    for (i = 0; i < K; ++i) {
        int r0 = 9*i;
        printf("mu[%d] = [%.6f %.6f %.6f]\n", i, mu[3*i+0], mu[3*i+1], mu[3*i+2]);
        printf("Sigma[%d] =\n", i);
        printf("  % .6e % .6e % .6e\n", S[r0+0], S[r0+1], S[r0+2]);
        printf("  % .6e % .6e % .6e\n", S[r0+3], S[r0+4], S[r0+5]);
        printf("  % .6e % .6e % .6e\n", S[r0+6], S[r0+7], S[r0+8]);
    }

	/* Build label image by assigning each pixel to the best cluster */
    {
        unsigned char *labels = (unsigned char*)xmalloc(W * H);
        assign_clusters(W, H, X, mu, S, K, labels);
        /* Save labels as an 8-bit PGM (IDs 0..K-1). Many viewers will map them to gray. */
        if (save_pgm_u8("labels.pgm", W, H, labels) == 0) {
            printf("Wrote labels.pgm (cluster IDs 0..%d)\n", K-1);
        }
        free(labels);
    }

    free(mu); free(S);
    free(I); free(X); free(w);
    return 0;
}