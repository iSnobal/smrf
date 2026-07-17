#include "envphys.h"
#include "envphys_c.h"
#include <errno.h>
#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <tgmath.h>

extern int errno;

void dewpt(
    int ngrid,        /* number of grid points */
    double *ea,       /* vapor pressure */
    int nthreads,     /* number of threads for parrallel processing */
    double tolerance, /* dew_point tolerance threshold */
    double *dpt       /* dew point temp (return) */
) {
    int samp;
    double ea_p; // pixel values
    float dpt_p; // pixel value

    omp_set_dynamic(0);            // Explicitly disable dynamic teams
    omp_set_num_threads(nthreads); // Use N threads for all consecutive parallel regions

#pragma omp parallel shared(ngrid, ea) private(samp, ea_p, dpt_p)
    {
#pragma omp for

        for (samp = 0; samp < ngrid; samp++) {

            ea_p = ea[samp];

            dpt_p = (float)dew_pointp((double)ea_p, tolerance);

            /*	convert from K to C	*/
            dpt_p -= FREEZE;

            /* set output band */

            dpt[samp] = dpt_p;
        }
    }
}

/* ------------------------------------------------------------------------ */

/*
 * Calculates the residual between two pressures.
 */
static double svp_residual(
    double t,   /* temperature (K) */
    double *svp /* vapor pressure  */
) {
    return (*svp - saturation_vapor_pressure(&t));
}

/* ------------------------------------------------------------------------ */

double dew_pointp(
    double vp_current, /* vapor pressure (Pa) */
    double tolerance   /* tolerable relative error */
) {
    double lower_temperature;
    double upper_temperature;
    double result;

    if (vp_current < 0 || vp_current > 1.5 * SEA_LEVEL) {
        perror("dew_point: vapor pressure < 0 or 1.5*SEA_LEVEL");
    }

    /* select starting guesses to span root */
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
    double lower_temperature, /* spanning guess for root */
    double upper_temperature, /* spanning guess for root */
    double vp_current,        /* current vapor pressure */
    double tolerance          /* tolerable relative error */
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
    double tol;
    double meps;
    int maxfun;

    meps  = DBL_EPSILON;
    errno = 0;

    if (lower_temperature == upper_temperature) {
        perror("zerobr: a = b");
        return (lower_temperature);
    }

    fb     = (fabs(upper_temperature) >= fabs(lower_temperature)) ? fabs(upper_temperature)
                                                                  : fabs(lower_temperature);
    tol    = 5.e-1 * tolerance + 2 * meps * fb;
    s      = log(fabs(upper_temperature - lower_temperature) / tol) / log(2.);
    maxfun = s * s + 1;

    fa = svp_residual(lower_temperature, &vp_current);
    fc = fb = svp_residual(upper_temperature, &vp_current);
    if (errno) {
        return (0.);
    }

    if (fabs(fb) <= tol)
        return (upper_temperature);
    if (fabs(fa) <= tol)
        return (lower_temperature);

    if (fb * fa > 0) {
        perror("zerobr: root not spanned");
        return (0.);
    }

    while (maxfun--) {

        if ((fb > 0 && fc > 0) || (fb <= 0 && fc <= 0)) {
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

        tol = meps * fabs(upper_temperature) + tolerance;
        m   = (c - upper_temperature) / 2;

        if (fabs(m) < tol || fb == 0)
            return (upper_temperature);

        /* see if bisection is forced */
        if (fabs(e) < tol || fabs(fa) <= fabs(fb))
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

            if (2 * p < 3 * m * q - fabs(tol * q) && p < fabs(s * q / 2))
                d = p / q;
            else
                d = e = m;
        }

        lower_temperature = upper_temperature;
        fa                = fb;

        if (fabs(d) > tol)
            upper_temperature += d;
        else if (m > 0)
            upper_temperature += tol;
        else
            upper_temperature -= tol;

        fb = svp_residual(upper_temperature, &vp_current);
        if (errno) {
            return (0.);
        }
    }
    perror("did not converge");

    return (0.);
}
