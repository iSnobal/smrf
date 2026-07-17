#include "envphys.h"
#include "envphys_c.h"

#include <math.h>
#include <omp.h>
#include <stdio.h>

/*
 * Calculates the dew point temperature for a grid of vapor pressures.
 */
void dew_point_t_for_grid(
    int ngrid,       /* Number of grid points in the passed arrays */
    double *ea,      /* Pointer to vapor pressure array */
    double *dpt,     /* Pointer to dew point temp (return) */
    int nthreads,    /* Number of threads for parallel processing */
    double tolerance /* Dew_point tolerance threshold */
) {
    omp_set_dynamic(0);            // Explicitly disable dynamic teams
    omp_set_num_threads(nthreads); // Use N threads for all consecutive parallel regions

#pragma omp parallel shared(ngrid, ea, dpt)
    {
#pragma omp for
        for (int i = 0; i < ngrid; i++) {
            dpt[i] = dew_point_temperature(ea[i], tolerance) - FREEZE;
        }
    }
}

/* ------------------------------------------------------------------------ */

/*
 * Calculates the residual between two pressures.
 */
static double svp_residual(
    double t,  /* temperature (K) */
    double svp /* vapor pressure  */
) {
    return (svp - saturation_vapor_pressure(&t));
}

/* ------------------------------------------------------------------------ */

/*
 * Finds the dew point by numerically inverting the saturation vapor pressure (SVP) function via
 * a bracketed root-finder (`zero_break` — Brent's method)
 */
double dew_point_temperature(
    double vp_current, /* vapor pressure (Pa) */
    double tolerance   /* tolerable relative error */
) {
    double lower_temperature;
    double upper_temperature;
    double result;

    if (vp_current < 0 || vp_current > 1.5 * SEA_LEVEL) {
        fprintf(
            stderr, "Vapor pressure %.4f Pa out of range [0, %.1f]\n", vp_current, 1.5 * SEA_LEVEL
        );
    }

    /* Select starting guesses to span root */
    lower_temperature = FREEZE;
    while (vp_current < saturation_vapor_pressure(&lower_temperature))
        lower_temperature *= .75;

    upper_temperature = FREEZE + 15;
    while (vp_current > saturation_vapor_pressure(&upper_temperature))
        upper_temperature *= 1.25;

    result = zero_break(lower_temperature, upper_temperature, vp_current, tolerance);

    return result;
}

/* ------------------------------------------------------------------------ */

double zero_break(
    double lower_temperature, /* Spanning guess for root */
    double upper_temperature, /* Spanning guess for root */
    double vp_current,        /* Current vapor pressure */
    double tolerance          /* Tolerable relative error */
) {
    double c = 0.;
    double d = 0.;
    double e = 0.;
    double fa;
    double fb;
    double fc;
    double m;
    double p;
    double q;
    double r;
    double s;
    int max_iter;

    fa = svp_residual(lower_temperature, vp_current);
    fb = svp_residual(upper_temperature, vp_current);

    if (fa <= fb || (upper_temperature - lower_temperature) <= tolerance) {
        fprintf(
            stderr,
            "Invalid bracket [%.6f, %.6f] K with f(a)=%.4e, f(b)=%.4e\n",
            lower_temperature,
            fa,
            upper_temperature,
            fb
        );
        return 0.;
    }

    s        = log(fabs(upper_temperature - lower_temperature) / tolerance) / log(2.);
    max_iter = (int)ceil(s) + 2;

    fc = fb; /* initialize so first iteration resets c via sign-agreement check */

    while (max_iter--) {

        if (fb * fc >= 0) {
            c  = lower_temperature;
            fc = fa;
            d = e = upper_temperature - lower_temperature;
        }

        if (fabs(fc) < fabs(fb)) {
            lower_temperature = upper_temperature;
            upper_temperature = c;
            c                 = lower_temperature;
            fa                = fb;
            fb                = fc;
            fc                = fa;
        }

        m = (c - upper_temperature) / 2;

        if (fabs(m) < tolerance || fb == 0)
            return upper_temperature;

        /* see if bisection is forced */
        if (fabs(e) < tolerance || fabs(fa) <= fabs(fb))
            d = e = m;

        else {
            s = fb / fa;

            if (lower_temperature == c) { /* linear interpolation */
                p = 2 * m * s;
                q = 1 - s;
            }

            else { /* inverse quadratic interpolation */
                q = fa / fc;
                r = fb / fc;
                p = s * (2 * m * q * (q - r) - (upper_temperature - lower_temperature) * (r - 1));
                q -= 1;
                q *= (r - 1) * (s - 1);
            }

            if (p > 0)
                q = -q;
            else
                p = -p;

            s = e;
            e = d;

            if (2 * p < 3 * m * q - fabs(tolerance * q) && p < fabs(s * q / 2))
                d = p / q;
            else
                d = e = m;
        }

        lower_temperature = upper_temperature;
        fa                = fb;

        if (fabs(d) > tolerance)
            upper_temperature += d;
        else if (m > 0)
            upper_temperature += tolerance;
        else
            upper_temperature -= tolerance;

        fb = svp_residual(upper_temperature, vp_current);
    }

    fprintf(
        stderr,
        "Dew point temperature id not converge with set tolerance %.2f ; last estimate %.2f K\n",
        tolerance,
        upper_temperature
    );

    return 0.;
}
