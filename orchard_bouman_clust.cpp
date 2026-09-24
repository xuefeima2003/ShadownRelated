/*
Great—here’s a pure ANSI C (C89/C90) translation of your Python Orchard–Bouman clustering snippet.
It keeps the same logic:

Maintain a list of clusters (nodes).
At each step, pick the node with the largest principal eigenvalue of its (weighted) covariance and
split along its top eigenvector at the projection of the mean.
Stop when the max eigenvalue ≤ minVar.
Return arrays of cluster means and covariances.

The implementation below:

Assumes feature dimensionality d = 3 (as in your code).
Uses a Jacobi eigen‑solver for symmetric 3×3 matrices (no external deps).
Uses weights as in your code and adds 1e‑5·I to covariance.
Stores input features as a flat row‑major array X[n*3] and weights w[n].
Returns mu[K*3] and Sigma[K*9] (row‑major 3×3 blocks), where K is the final number of clusters.


Notes vs. your Python version

Weighted stats: identical computation to your einsum‐based mean and (t.T@t)/W + 1e-5*I covariance
 (we directly accumulate Σ w_i (x−μ)(x−μ)^T / W).
Eigen decomposition: your np.linalg.eig is replaced with a small Jacobi solver for 3×3 symmetric
 matrices. We pick the eigenvector with largest |λ| (principal axis), matching np.argmax(np.abs(V)).
Split rule: same predicate x·e ≤ μ·e and complement for the other side.
Stopping: loop ends when the node with max λ ≤ minVar.
Degenerate splits (all points to one side) are handled by disabling that node’s further splitting
 (lambda = 0) and continuing; this avoids empty sets.

How to use

Prepare your X as a flat array of shape n × 3 (row‑major), and w of length n.
Call:

double *mu = NULL, *Sigma = NULL;
int K = 0;
int rc = clustFunc(X, w, n, 0.05, &mu, &Sigma, &K);

On success, you get:

K clusters,
mu length K*3,
Sigma length K*9 (each 3×3 covariance in row‑major).


Free the returned arrays with free(mu); free(Sigma);.


*/


/* orchard_bouman.c
 * ANSI C (C89) implementation of a simple Orchard–Bouman-style clustering
 * corresponding to the provided Python/Numpy snippet.
 *
 * Build:
 *   gcc -O2 -std=c89 -Wall -Wextra -o orchard_bouman orchard_bouman.c
 *
 * Notes:
 * - Feature dimension is fixed to 3 (matches the Python code).
 * - Input:
 *     X: double*, flattened row-major (n x 3)
 *     w: double*, length n, non-negative weights
 * - Output:
 *     mu_out:  allocated double*, K x 3 means (caller frees)
 *     Sigma_out: allocated double*, K x 9 covariances (caller frees, row-major 3x3)
 *     K_out: number of clusters
 * - Stopping rule: max principal eigenvalue <= minVar
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef DBL_EPSILON
#include <float.h>
#endif

/* ==========================
   Utility helpers
   ========================== */

static void *xmalloc(size_t n)
{
    void *p = malloc(n);
    if (!p) {
        fprintf(stderr, "Out of memory (%lu bytes)\n", (unsigned long)n);
        exit(1);
    }
    return p;
}

static void *xrealloc(void *ptr, size_t n)
{
    void *p = realloc(ptr, n);
    if (!p) {
        fprintf(stderr, "Out of memory (realloc %lu bytes)\n", (unsigned long)n);
        exit(1);
    }
    return p;
}

static double dot3(const double *a, const double *b)
{
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

static void add_outer_weighted(double C[3][3], const double *v, double w)
{
    /* C += w * (v v^T) */
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
    A[0][0] += s;
    A[1][1] += s;
    A[2][2] += s;
}

static void copy_vec3(double *dst, const double *src)
{
    dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2];
}

static void copy_mat3(double dst[3][3], const double src[3][3])
{
    int i, j;
    for (i = 0; i < 3; ++i)
        for (j = 0; j < 3; ++j)
            dst[i][j] = src[i][j];
}

/* Normalize vector v (length-3). Returns its original norm. */
static double normalize3(double *v)
{
    double n = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    if (n > 0.0) {
        v[0] /= n; v[1] /= n; v[2] /= n;
    }
    return n;
}

/* ==========================
   Jacobi eigen-solver for 3x3 symmetric matrices
   Returns eigenvalues (eval[0..2]) and eigenvectors in columns of evec (3x3).
   ========================== */
static void jacobi_eigen_3x3(double A[3][3], double evec[3][3], double eval[3])
{
    double V[3][3];
    double a01, a02, a12, app, apq, aqq, phi, t, c, s, tau, tmp;
    int i, j, k;
    int iter, p, q;

    /* Initialize V as identity */
    V[0][0] = 1.0; V[0][1] = 0.0; V[0][2] = 0.0;
    V[1][0] = 0.0; V[1][1] = 1.0; V[1][2] = 0.0;
    V[2][0] = 0.0; V[2][1] = 0.0; V[2][2] = 1.0;

    /* Jacobi iterations */
    for (iter = 0; iter < 50; ++iter) {
        /* Find largest off-diagonal element (by absolute value) */
        a01 = fabs(A[0][1]);
        a02 = fabs(A[0][2]);
        a12 = fabs(A[1][2]);

        p = 0; q = 1; /* default to (0,1) */
        if (a02 > a01 && a02 >= a12) { p = 0; q = 2; }
        else if (a12 >= a01 && a12 >= a02) { p = 1; q = 2; }

        if (fabs(A[p][q]) < 1e-12)
            break; /* sufficiently diagonal */

        /* Compute Jacobi rotation to zero A[p][q] */
        app = A[p][p];
        aqq = A[q][q];
        apq = A[p][q];

        /* tan(2*phi) = 2*apq / (aqq - app) */
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

        /* Apply rotation to A: A' = J^T A J (only update needed elements) */
        for (k = 0; k < 3; ++k) {
            double Akp = A[k][p];
            double Akq = A[k][q];
            A[k][p] = c*Akp - s*Akq;
            A[k][q] = s*Akp + c*Akq;
        }
        for (k = 0; k < 3; ++k) {
            double Apk = A[p][k];
            double Aqk = A[q][k];
            A[p][k] = c*Apk - s*Aqk;
            A[q][k] = s*Apk + c*Aqk;
        }

        /* Maintain symmetry explicitly */
        A[p][q] = A[q][p] = 0.0;

        /* Update eigenvector matrix V = V * J */
        for (k = 0; k < 3; ++k) {
            double Vkp = V[k][p];
            double Vkq = V[k][q];
            V[k][p] = c*Vkp - s*Vkq;
            V[k][q] = s*Vkp + c*Vkq;
        }
    }

    /* Extract eigenvalues from diagonal of A */
    eval[0] = A[0][0];
    eval[1] = A[1][1];
    eval[2] = A[2][2];

    /* Copy eigenvectors (columns of V) */
    for (i = 0; i < 3; ++i)
        for (j = 0; j < 3; ++j)
            evec[i][j] = V[i][j];

    /* (Optional) normalize eigenvectors */
    for (j = 0; j < 3; ++j) {
        double col[3];
        col[0] = evec[0][j];
        col[1] = evec[1][j];
        col[2] = evec[2][j];
        normalize3(col);
        evec[0][j] = col[0];
        evec[1][j] = col[1];
        evec[2][j] = col[2];
    }
}

/* ==========================
   Dataset and Node structures
   ========================== */

typedef struct {
    const double *X; /* flattened row-major: n x 3 */
    const double *w; /* length n */
    int n;           /* number of rows */
} Dataset;

typedef struct Node_ {
    int *idx;        /* indices into dataset, length m */
    int m;           /* number of indices */
    double W;        /* sum of weights */
    double mu[3];    /* weighted mean */
    double cov[3][3];/* weighted covariance (+ 1e-5 I) */
    double lambda;   /* largest (abs) eigenvalue */
    double e[3];     /* corresponding eigenvector */
} Node;

/* Construct a node given a set of indices; compute mu, cov, eig */
static int node_init(Node *node, const Dataset *D, const int *idx, int m)
{
    int i, j;
    double W = 0.0;
    double mu[3] = {0.0, 0.0, 0.0};
    double C[3][3];
    double evec[3][3], eval[3];
    double A[3][3];

    if (m <= 0) {
        /* empty cluster is invalid for stats */
        node->idx = NULL;
        node->m = 0;
        node->W = 0.0;
        node->lambda = 0.0;
        node->e[0] = node->e[1] = 0.0; node->e[2] = 1.0;
        return 0;
    }

    node->idx = (int*)xmalloc(sizeof(int) * m);
    for (i = 0; i < m; ++i) node->idx[i] = idx[i];
    node->m = m;

    /* Weighted mean */
    for (i = 0; i < m; ++i) {
        int r = idx[i];
        double wi = D->w[r];
        const double *xi = &D->X[3*r];
        W += wi;
        mu[0] += wi * xi[0];
        mu[1] += wi * xi[1];
        mu[2] += wi * xi[2];
    }

    if (W <= 0.0) W = 1.0; /* prevent divide-by-zero; degenerate */

    mu[0] /= W; mu[1] /= W; mu[2] /= W;

    /* Weighted covariance: (1/W) * sum_i w_i * (x - mu)(x - mu)^T + 1e-5 I */
    C[0][0] = C[0][1] = C[0][2] = 0.0;
    C[1][0] = C[1][1] = C[1][2] = 0.0;
    C[2][0] = C[2][1] = C[2][2] = 0.0;

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

    for (i = 0; i < 3; ++i)
        for (j = 0; j < 3; ++j)
            C[i][j] /= W;

    mat_add_eye(C, 1e-5);

    /* Copy to A for eigen-solver (which overwrites its input) */
    copy_mat3(A, C);

    jacobi_eigen_3x3(A, evec, eval);

    /* Pick eigenvector of largest |eigenvalue| */
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
    node->mu[0] = mu[0]; node->mu[1] = mu[1]; node->mu[2] = mu[2];
    copy_mat3(node->cov, C);

    return 1;
}

static void node_free(Node *node)
{
    if (!node) return;
    if (node->idx) free(node->idx);
    node->idx = NULL;
    node->m = 0;
}

/* ==========================
   Dynamic array of Node*
   ========================== */

typedef struct {
    Node **data;
    int size;
    int cap;
} NodeVec;

static void nodevec_init(NodeVec *v)
{
    v->data = NULL; v->size = 0; v->cap = 0;
}

static void nodevec_push(NodeVec *v, Node *n)
{
    if (v->size == v->cap) {
        v->cap = v->cap ? (v->cap * 2) : 4;
        v->data = (Node**)xrealloc(v->data, v->cap * sizeof(Node*));
    }
    v->data[v->size++] = n;
}

static void nodevec_remove_at(NodeVec *v, int idx)
{
    int i;
    if (idx < 0 || idx >= v->size) return;
    for (i = idx + 1; i < v->size; ++i)
        v->data[i-1] = v->data[i];
    v->size--;
}

static void nodevec_free(NodeVec *v)
{
    int i;
    if (!v) return;
    for (i = 0; i < v->size; ++i) {
        node_free(v->data[i]);
        free(v->data[i]);
    }
    free(v->data);
    v->data = NULL;
    v->size = v->cap = 0;
}

/* ==========================
   Split operation
   ========================== */

static int argmax_lambda(const NodeVec *v)
{
    int i, imax = 0;
    double best = -1.0;
    for (i = 0; i < v->size; ++i) {
        if (v->data[i]->lambda > best) {
            best = v->data[i]->lambda;
            imax = i;
        }
    }
    return imax;
}

/* Returns 1 if a split occurred, 0 otherwise (e.g., degenerate) */
static int split_once(NodeVec *nodes, const Dataset *D)
{
    int i, j;
    int idx_max = argmax_lambda(nodes);
    Node *C = nodes->data[idx_max];
    double t0 = dot3(C->mu, C->e);

    /* Partition indices based on projection onto eigenvector */
    int m = C->m;
    int *Aidx = (int*)xmalloc(sizeof(int) * m);
    int *Bidx = (int*)xmalloc(sizeof(int) * m);
    int ma = 0, mb = 0;

    for (i = 0; i < m; ++i) {
        int r = C->idx[i];
        const double *x = &D->X[3*r];
        double proj = x[0]*C->e[0] + x[1]*C->e[1] + x[2]*C->e[2];
        if (proj <= t0) Aidx[ma++] = r;
        else            Bidx[mb++] = r;
    }

    /* Guard against empty child (can happen with ties) */
    if (ma == 0 || mb == 0) {
        free(Aidx); free(Bidx);
        /* mark as unsplittable by zeroing lambda so loop can terminate */
        C->lambda = 0.0;
        return 0;
    }

    /* Create two new nodes */
    {
        Node *Ca = (Node*)xmalloc(sizeof(Node));
        Node *Cb = (Node*)xmalloc(sizeof(Node));
        if (!node_init(Ca, D, Aidx, ma) || !node_init(Cb, D, Bidx, mb)) {
            /* Fallback: clean-up */
            node_free(Ca); free(Ca);
            node_free(Cb); free(Cb);
            free(Aidx); free(Bidx);
            C->lambda = 0.0;
            return 0;
        }

        /* Replace C with Ca, append Cb */
        node_free(C);
        free(C);
        nodes->data[idx_max] = Ca;
        nodevec_push(nodes, Cb);
    }

    free(Aidx); free(Bidx);
    return 1;
}

/* ==========================
   Public API: clustFunc
   ========================== */

/* Returns 0 on success, nonzero on failure */
int clustFunc(
    const double *X,    /* n x 3 row-major */
    const double *w,    /* length n */
    int n,
    double minVar,      /* stopping threshold on max eigenvalue */
    double **mu_out,    /* (K x 3) caller frees */
    double **Sigma_out, /* (K x 9) caller frees */
    int *K_out          /* number of clusters */
)
{
    Dataset D;
    NodeVec nodes;
    Node *root;
    int *root_idx;
    int i, k, K;

    if (!X || !w || n <= 0 || !mu_out || !Sigma_out || !K_out) return 1;

    D.X = X; D.w = w; D.n = n;

    nodevec_init(&nodes);

    /* Create root node with all indices */
    root_idx = (int*)xmalloc(sizeof(int) * n);
    for (i = 0; i < n; ++i) root_idx[i] = i;

    root = (Node*)xmalloc(sizeof(Node));
    if (!node_init(root, &D, root_idx, n)) {
        free(root_idx);
        node_free(root); free(root);
        nodevec_free(&nodes);
        return 2;
    }
    free(root_idx);

    nodevec_push(&nodes, root);

    /* Repeatedly split the node with the largest lambda */
    for (;;) {
        int idx_max = argmax_lambda(&nodes);
        double max_lambda = nodes.data[idx_max]->lambda;
        int did_split;

        if (max_lambda <= minVar) break;

        did_split = split_once(&nodes, &D);
        if (!did_split) {
            /* If we couldn't split (degenerate), check if any other node has > minVar */
            int any = 0;
            for (i = 0; i < nodes.size; ++i) {
                if (nodes.data[i]->lambda > minVar) { any = 1; break; }
            }
            if (!any) break;
        }
    }

    /* Export results */
    K = nodes.size;
    *K_out = K;
    *mu_out = (double*)xmalloc(sizeof(double) * K * 3);
    *Sigma_out = (double*)xmalloc(sizeof(double) * K * 9);

    for (k = 0; k < K; ++k) {
        Node *C = nodes.data[k];
        double *mu = &(*mu_out)[3*k];
        double *S = &(*Sigma_out)[9*k];
        mu[0] = C->mu[0]; mu[1] = C->mu[1]; mu[2] = C->mu[2];

        /* row-major copy of 3x3 covariance */
        S[0] = C->cov[0][0]; S[1] = C->cov[0][1]; S[2] = C->cov[0][2];
        S[3] = C->cov[1][0]; S[4] = C->cov[1][1]; S[5] = C->cov[1][2];
        S[6] = C->cov[2][0]; S[7] = C->cov[2][1]; S[8] = C->cov[2][2];
    }

    nodevec_free(&nodes);
    return 0;
}

/* ==========================
   Example usage (optional)
   ========================== */

int main(void)
{
    // Tiny synthetic example: 4 points in 3D with equal weights
    double X[] = {
        0.0, 0.0, 0.0,
        0.9, 0.9, 0.9,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0
    };
    double w[] = {1.0, 1.0, 1.0, 1.0};
    int n = 4;

    double *mu = NULL, *S = NULL;
    int K = 0;
    double minVar = 0.05;
    int rc = clustFunc(X, w, n, minVar, &mu, &S, &K);
    if (rc != 0) {
        fprintf(stderr, "clustFunc failed: %d\n", rc);
        return 1;
    }

    printf("K = %d clusters\n", K);
    for (int k = 0; k < K; ++k) {
        printf("mu[%d] = [%g %g %g]\n", k, mu[3*k], mu[3*k+1], mu[3*k+2]);
        printf("Sigma[%d] = \n", k);
        for (int r = 0; r < 3; ++r) {
            printf("  % .6e % .6e % .6e\n",
                S[9*k + 3*r + 0], S[9*k + 3*r + 1], S[9*k + 3*r + 2]);
        }
    }

    free(mu);
    free(S);
    return 0;
}

