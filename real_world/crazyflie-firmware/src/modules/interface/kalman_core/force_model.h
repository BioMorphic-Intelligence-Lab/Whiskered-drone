#ifndef FORCE_MODEL_H
#define FORCE_MODEL_H

#include <stdbool.h>

// --- physical / model constants (tune if needed) ---
static const double EI_CONST = 6e-5; // E*I (N·m^2) - replace if you have better value

static const double COEF_A = 2.654855;
static const double COEF_B = 0.0509896;
static const double COEF_C = 0.0126749;
static const double COEF_D = 0.00142039;
static const double COEF_EPOLY = 5.84525e-5;

static const double EPS = 1e-12;

/**
 * Compute normal force magnitude N for one whisker using bending model
 *
 * Inputs:
 *   lo        - whisker original length (same units as d) 
 *   d         - contact depth (same units)
 *   alpha_rad - whisker mounting angle alpha in radians
 *
 * Returns N >= 0. Returns 0 if geometry invalid.
 */
double whisker_compute_N(double lo, double d, double alpha_rad);

/**
 * Compute 2D force vector (Fx, Fy) in body frame from left and right whiskers
 *
 * Inputs:
 *   lo           - whisker original length
 *   d_left       - contact depth left whisker
 *   d_right      - contact depth right whisker
 *   spacing      - distance between left and right whisker roots
 *   placement_rad- whisker mounting angle in radians
 *
 * Outputs:
 *   outFx        - pointer to x component of force
 *   outFy        - pointer to y component of force
 */
void whisker_force_vector_2d(double lo,
                             double d_left, double d_right,
                             double spacing, double placement_rad,
                             double *outFx, double *outFy);

#endif // FORCE_MODEL_H
