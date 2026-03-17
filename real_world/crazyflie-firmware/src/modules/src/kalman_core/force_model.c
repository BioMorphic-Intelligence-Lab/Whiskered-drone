#include <math.h>
#include <stdbool.h>
#include "force_model.h"

/**
 * Compute normal force magnitude N for one whisker using your specified formula:
 *   F_bend = EI * poly(n) * theta * gamma^2 / (Pnorm2 * (cos theta + n sin theta))
 *   N = F_bend / sin(3/2 theta + alpha)
 *
 * Inputs:
 *   lo        - whisker original length (same units as d) 
 *   d         - contact depth (same units)
 *   alpha_rad - whisker mounting angle alpha in radians (as used in Pc^S)
 *
 * Returns N (>=0). If geometry invalid or denom ~ 0, returns 0.
 */
double whisker_compute_N(double lo, double d, double alpha_rad)
{
    // contact point in sensor/body frame
    double Px = lo - d * cos(alpha_rad);
    double Py = d * sin(alpha_rad);

    // squared norm
    double Pnorm2 = Px*Px + Py*Py;
    if (Pnorm2 < EPS) return 0.0;

    // theta (rotation angle)
    double theta = atan2(Py, Px); // in radians

    // beta ~ 3/2 theta, n = tan(beta)
    double beta = 1.5 * theta;
    double n = tan(beta);

    // gamma (characteristic radius factor)
    double gamma = 0.841655 - 0.0067807 * n + 0.000438 * n * n;

    // polynomial factor
    double poly = COEF_A - COEF_B * n + COEF_C * n * n
                  - COEF_D * n * n * n + COEF_EPOLY * n * n * n * n;

    // denominator terms
    double denom_ang = cos(theta) + n * sin(theta);
    double denom_geom = sin(beta + alpha_rad); // sin(3/2 theta + alpha)

    // protect against near-zero denominators (model singularities)
    if (fabs(denom_ang) < EPS) return 0.0;
    if (fabs(denom_geom) < EPS) return 0.0;

    // bending force (F_bend)
    double F_bend = (EI_CONST * poly * theta * gamma * gamma) / (Pnorm2 * denom_ang);

    // normal-to-plane force N
    double N = F_bend / denom_geom;

    // return absolute magnitude (direction handled separately)
    if (N < 0.0) N = -N;
    return N;
}

/**
 * Given whisker parameters (lo,d,alpha) compute 2D force vector (Fx,Fy) in the body frame.
 * Direction is from root (0,0) to contact point (Px,Py).
 */
void whisker_force_vector_2d(double lo,
                                    double d_left, double d_right,
                                    double spacing, double placement_rad,
                                    double *outFx, double *outFy)
{
    // 1. 两个 whisker 的 bending force（沿 x 轴的大小）
    double N_left  = whisker_compute_N(lo, d_left, placement_rad);
    double N_right = whisker_compute_N(lo, d_right, placement_rad);
    double N_total = N_left + N_right;

    if (N_total <= 0.0) {
        *outFx = 0.0;
        *outFy = 0.0;
        return;
    }

    // 2. 用深度差和间距算出墙壁斜率
    double wall_vector = (d_left - d_right) / spacing;

    // 3. x 方向：反作用力向后，y 方向：随斜率偏移
    *outFx = -N_total;
    *outFy = -N_total * wall_vector;
}
