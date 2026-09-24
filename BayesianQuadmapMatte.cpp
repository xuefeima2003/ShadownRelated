// code for paper titled "A Bayesian Approach for Shadow Extraction from a Single Image"
// Based on the mathematical framework and parameters provided in the sources, here is a C code implementation of
// the Bayesian shadow extraction logic.
// This code focuses on the core mathematical updates described in Section 3, specifically the Poisson-guided
// estimation and the Bayesian iterative optimization that preserves textures.
// C Implementation of Bayesian Shadow Extraction

// Key Logic Notes
// Anisotropic Smoothing: The code implements the derivative of Equation 13 in the final optimization loop.
// This term specifically penalizes differences in the shadow-free image F, which "maintains texture gradients"
// even if the shadow β is adjusted
// 
// Poisson Guidance: The F_hat array stores the initial guess. Without this, the Bayesian likelihood would lack
// a reference point to distinguish between a change in texture and a change in shadow
// 
// Channel Independence: As per the image model I=βF, the optimization is run three times (once for R, G, and B)
// to allow the shadow to have different color tints
// 
// Would you like me to create a tailored report detailing how to implement the more complex Gaussian Mixture Model
// (GMM) for the color statistics part of this code?


#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Region definitions from the Quadmap [1]
#define REGION_E     0   // Excluded (shadow-casting object)
#define REGION_U     64  // Uncertain (penumbra/shadow boundary)
#define REGION_S_BAR 128 // Non-shadowed (shadow-free reference)
#define REGION_S     255 // Definitely shadowed

// Mathematical constants defined in Section 3.6 [2]
const float LAMBDA_C = 0.5f;       // 1/(2*sigma_c^2) where sigma_c = 1
const float LAMBDA_1 = 1000.0f;    // 1/(2*sigma_1^2)
const float SIGMA_P_SQ = 0.0005f;  // sigma_p^2 for smoothness weight k(x,y)
const float LAMBDA_2 = 0.001f;     // 1/(2*sigma_2^2). Set low unless hard boundary.
const float SOR_OMEGA = 1.9f;      // Successive Over-Relaxation parameter [3]

typedef struct {
    float r, g, b;
} ColorF;

// Core processing function for one channel [4]
void optimize_shadow_channel(int width, int height, float* I, unsigned char* quadmap, float* beta_out) {
    float* F_hat = (float*)calloc(width * height, sizeof(float));
    float* beta = (float*)malloc(width * height * sizeof(float));
    float* affinity = (float*)calloc(width * height, sizeof(float));
    
    // 1. Initial Beta Estimation in S [5]
    float sum_S = 0, sum_S_bar = 0;
    int count_S = 0, count_S_bar = 0;
    for (int i = 0; i < width * height; i++) {
        if (quadmap[i] == REGION_S) { sum_S += I[i]; count_S++; }
        if (quadmap[i] == REGION_S_BAR) { sum_S_bar += I[i]; count_S_bar++; }
    }
    float beta_S = (sum_S / count_S) / (sum_S_bar / count_S_bar);

    // 2. Approximate Affinity & Initial Beta in U [6, 7]
    // (Simplified: using a single Gaussian model for S/S_bar instead of full GMM)
    for (int i = 0; i < width * height; i++) {
        beta[i] = 1.0f; // Initialize Beta to 1 [2]
        if (quadmap[i] == REGION_S) beta[i] = beta_S;
        else if (quadmap[i] == REGION_U) {
            // Placeholder for model affinity logic: P_S / (P_S + P_S_bar)
            affinity[i] = 0.5f; 
            beta[i] = 1.0f + affinity[i] * (beta_S - 1.0f); // Eqn 3 in [7]
        }
    }

    // 3. Solve Poisson Equation for F_hat (Initial Shadowless Estimate) [3]
    // Using Gauss-Seidel with SOR
    for (int iter = 0; iter < 100; iter++) {
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                int idx = y * width + x;
                if (quadmap[idx] == REGION_E) continue;

                // Guidance field F_est = I/beta [8]
                float f_est = I[idx] / beta[idx];
                // Confidence C = lambda * (1 - affinity*(1-affinity)) [3]
                float C = 1.0f * (1.0f - affinity[idx] * (1.0f - affinity[idx]));

                float target = (F_hat[idx+1] + F_hat[idx-1] + F_hat[idx+width] + F_hat[idx-width]) / 4.0f;
                // Simplified divergence of C*grad(F_est)
                F_hat[idx] = (1.0f - SOR_OMEGA) * F_hat[idx] + SOR_OMEGA * (target + C * f_est * 0.1f);
            }
        }
    }

    // 4. Bayesian Iterative Optimization [2]
    for (int iter = 0; iter < 50; iter++) {
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                int idx = y * width + x;
                if (quadmap[idx] == REGION_S_BAR || quadmap[idx] == REGION_E) continue;

                float numer = LAMBDA_C * F_hat[idx] * I[idx];
                float denom = LAMBDA_C * F_hat[idx] * F_hat[idx];

                // Neighbors for smoothness [9] and texture preservation [10]
                int neighbors[] = {idx-1, idx+1, idx-width, idx+width};
                for (int n = 0; n < 4; n++) {
                    int n_idx = neighbors[n];
                    float k = expf(-powf(affinity[idx] - affinity[n_idx], 2) / SIGMA_P_SQ);
                    
                    // Eqn 12: Smoothness
                    numer += LAMBDA_1 * beta[n_idx] * k;
                    denom += LAMBDA_1 * k;

                    // Eqn 13: Anisotropic Texture Smoothing
                    float tex_weight = affinity[idx] * affinity[n_idx] * LAMBDA_2;
                    numer += tex_weight * beta[n_idx] * I[idx] * I[n_idx];
                    denom += tex_weight * I[n_idx] * I[n_idx];
                }
                beta[idx] = numer / denom; // Optimization update derived from Eqn 5 [2]
            }
        }
    }

    for (int i = 0; i < width * height; i++) beta_out[i] = beta[i];
    free(F_hat); free(beta); free(affinity);
}

void process_image(int w, int h, unsigned char* rgb_in, unsigned char* quadmap, unsigned char* rgb_out) {
    float* channel = (float*)malloc(w * h * sizeof(float));
    float* beta = (float*)malloc(w * h * sizeof(float));

    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < w * h; i++) channel[i] = rgb_in[i * 3 + c] / 255.0f;
        
        optimize_shadow_channel(w, h, channel, quadmap, beta);

        for (int i = 0; i < w * h; i++) {
            // Shadowless image F = I / beta [4]
            float f_val = (channel[i] / beta[i]) * 255.0f;
            rgb_out[i * 3 + c] = (unsigned char)fminf(255.0f, fmaxf(0.0f, f_val));
        }
    }
    free(channel); free(beta);
}


///////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////       refined version       /////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

// Structural Improvements
// Separation of Concerns: The code is now split into Phase 1 (Poisson guidance) and Phase 2 (Bayesian
// optimization), reflecting the structure described in Section 1.2
// 
// Explicit Dirichlet Boundaries: The solve_poisson_guidance function now initializes F_hat using the
// observed image I for non-shadowed areas and the adjusted estimate for shadowed areas, as required
// for proper boundary conditions
// 
// Math-to-Source Mapping: Comments specifically link the code logic to the likelihood and prior
// definitions in Sections 3.4 and 3.5
// 
// Texture Preservation: The "Anisotropic Smoothing" logic in optimize_beta is isolated, showing how
// it uses the affinity map to identify hard boundaries and preserve underlying gradients

// The code uses the parameters specified in the sources:
// Sigma2(c)=1.0
// Sigma2(p)=0.0005
// 1.0/(2*Sigma2(I))=1000.0

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Region labels from the user-supplied Quadmap [4] */
typedef enum { EXCLUDED = 0, UNCERTAIN = 64, NON_SHADOWED = 128, SHADOWED = 255 } Region;

/* Parameters from Section 3.6 [3] */
#define LAMBDA_1 1000.0f  /* Smoothness weight: 1 / (2 * sigma_1^2) */
#define SIGMA_P_SQ 0.0005f /* For weight k(x,y) in Eqn 12 */
#define LAMBDA_C 0.5f     /* Likelihood weight: 1 / (2 * sigma_c^2) where sigma_c = 1 */
#define SOR_OMEGA 1.9f    /* Successive Over-Relaxation parameter [5] */

/**
 * PHASE 1: Initial Shadowless Image Estimation (F_hat*) 
 * Solving the Poisson Equation (Eqn 10) to guide optimization [5].
 */
void solve_poisson_guidance(int w, int h, float* I, unsigned char* qmap, float* affinity, float* beta_init, float* F_hat) {
    /* Initialize F_hat with Dirichlet boundary conditions [6] */
    for (int i = 0; i < w * h; i++) {
        if (qmap[i] == SHADOWED) F_hat[i] = I[i] / beta_init[i];
        else F_hat[i] = I[i];
    }

    /* Gauss-Seidel with SOR to solve delta(F_hat) = div(C * grad(I/beta)) [5] */
    for (int iter = 0; iter < 150; iter++) {
        for (int y = 1; y < h - 1; y++) {
            for (int x = 1; x < w - 1; x++) {
                int idx = y * w + x;
                if (qmap[idx] != UNCERTAIN) continue;

                /* Eqn 8: Confidence measurement weakens guidance at hard boundaries [5] */
                float C = (1.0f - affinity[idx] * (1.0f - affinity[idx]));
                
                /* Guidance Field: Gradient of (I / beta) [7] */
                float f_est = I[idx] / beta_init[idx];
                
                /* Poisson relaxation step */
                float neighbors = (F_hat[idx-1] + F_hat[idx+1] + F_hat[idx-w] + F_hat[idx+w]);
                float div_G = C * f_est * 0.1f; // Simplified divergence contribution
                
                float target = (neighbors + div_G) / 4.0f;
                F_hat[idx] = (1.0f - SOR_OMEGA) * F_hat[idx] + SOR_OMEGA * target;
            }
        }
    }
}

/**
 * PHASE 2: Bayesian Optimization
 * Optimizes beta by solving Eqn 5 (First derivative = 0) [2, 3].
 */
void optimize_beta(int w, int h, float* I, unsigned char* qmap, float* affinity, float* F_hat, float* beta) {
    for (int iter = 0; iter < 50; iter++) {
        for (int y = 1; y < h - 1; y++) {
            for (int x = 1; x < w - 1; x++) {
                int idx = y * w + x;
                if (qmap[idx] == NON_SHADOWED || qmap[idx] == EXCLUDED) continue;

                /* 1. Likelihood Term (Eqn 11): Models measurement error [8] */
                float num = LAMBDA_C * F_hat[idx] * I[idx];
                float den = LAMBDA_C * F_hat[idx] * F_hat[idx];

                /* 2. Prior Terms (Eqn 12 & 13): Smoothness and Texture Preservation [9, 10] */
                int n_list[] = {idx - 1, idx + 1, idx - w, idx + w};
                for (int i = 0; i < 4; i++) {
                    int nid = n_list[i];
                    
                    /* Weight k: weakens smoothness on high-frequency components [9] */
                    float k = expf(-powf(affinity[idx] - affinity[nid], 2) / (2.0f * SIGMA_P_SQ));
                    
                    /* Smoothness Prior (Eqn 12) */
                    num += LAMBDA_1 * beta[nid] * k;
                    den += LAMBDA_1 * k;

                    /* Anisotropic Texture Smoothing (Eqn 13) 
                       Operates by minimizing (F(x) - F(y))^2 to maintain texture gradients [10] */
                    if (affinity[idx] > 0.8f) { // Only if near a hard boundary
                        float lambda_2 = 100.0f; 
                        num += lambda_2 * affinity[idx] * affinity[nid] * beta[nid] * I[idx] * I[nid];
                        den += lambda_2 * affinity[idx] * affinity[nid] * I[nid] * I[nid];
                    }
                }
                beta[idx] = num / den; // Update beta based on the derivative [3]
            }
        }
    }
}

/**
 * Top-level routine to process one color channel independently [11].
 */
void process_channel(int w, int h, float* img_in, unsigned char* qmap, float* img_out) {
    float* beta = (float*)malloc(w * h * sizeof(float));
    float* F_hat = (float*)calloc(w * h, sizeof(float));
    float* affinity = (float*)calloc(w * h, sizeof(float));

    /* Initialize: find mean beta in S to use for initial U estimation [12, 13] */
    float beta_S = 0.4f; // Placeholder: Mean(I_S) / Mean(I_S_bar)
    for (int i = 0; i < w * h; i++) {
        affinity[i] = (qmap[i] == SHADOWED) ? 1.0f : (qmap[i] == UNCERTAIN ? 0.5f : 0.0f);
        beta[i] = 1.0f - affinity[i] * (1.0f - beta_S); // Eqn 3 approximation [13]
    }

    solve_poisson_guidance(w, h, img_in, qmap, affinity, beta, F_hat);
    optimize_beta(w, h, img_in, qmap, affinity, F_hat, beta);

    /* Final Result: Shadowless Image F = I / beta [11] */
    for (int i = 0; i < w * h; i++) {
        img_out[i] = img_in[i] / beta[i];
    }

    free(beta); free(F_hat); free(affinity);
}