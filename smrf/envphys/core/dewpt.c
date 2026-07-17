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
		int ngrid,		/* number of grid points */
		double *ea,		/* vapor pressure */
		int nthreads,	/* number of threads for parrallel processing */
		double tol,		/* dew_point tolerance threshold */
		double *dpt		/* dew point temp (return) */
)
{
	int samp;
	double ea_p; // pixel values
	float dpt_p; // pixel value

	omp_set_dynamic(0);     // Explicitly disable dynamic teams
	omp_set_num_threads(nthreads); // Use N threads for all consecutive parallel regions

#pragma omp parallel shared(ngrid, ea) private(samp, ea_p, dpt_p)
	{
#pragma omp for

		for (samp=0; samp < ngrid; samp++) {

			ea_p = ea[samp];

			dpt_p = (float) dew_pointp((double) ea_p, tol);

			/*	convert from K to C	*/
			dpt_p -= FREEZE;

			/* set output band */

			dpt[samp] = dpt_p;

		}
	}

}

/* ------------------------------------------------------------------------ */

static double e_pass;
#pragma omp threadprivate(e_pass)

static double
satm(double t)
{
	return(e_pass - saturation_vapor_pressure(&t));
}

/* ------------------------------------------------------------------------ */

double
dew_pointp(
		double e, 	/* vapor pressure (Pa) */
		double tol)	/* tolerance threshold */
{
	double	a;
	double	b;
	double	result;

	if (e < 0 || e > 1.5*SEA_LEVEL) {
		perror("dew_point: vapor pressure < 0 or 1.5*SEA_LEVEL");
	}

	/* select starting guesses to span root */

	/* lower */
	a = FREEZE;
	while (e < saturation_vapor_pressure(&a))
		a *= .75;

	/* upper */
	b = FREEZE + 15;
	while (e > saturation_vapor_pressure(&b))
		b *= 1.25;

	e_pass = e;
	result = zerobr(a, b, tol);

	return(result);
}

/* ------------------------------------------------------------------------ */

double
zerobr(
	double	a,		/* spanning guess for root	*/
	double	b,		/* spanning guess for root	*/
	double	t)		/* tolerable relative error	*/
{
	double	c =0.;
	double	d =0.;
	double	e =0.;
	double	fa;
	double	fb;
	double	fc;
	double	m;
	double	p;
	double	q;
	double	r;
	double	s;
	double	tol;
	double	meps;
	int	maxfun;

	meps = DBL_EPSILON;
	errno = 0;

	/* compute max number of function evaluations */
	if (a == b) {
		perror("zerobr: a = b");
//		errno = ERROR;
		return(a);
	}

    fb     = (fabs(b) >= fabs(a)) ? fabs(b) : fabs(a);
    tol    = 5.e-1 * t + 2 * meps * fb;
    s      = log(fabs(b - a) / tol) / log(2.);
    maxfun = s * s + 1;

	fa = satm(a);
	fc = fb = satm(b);
	if (errno) {
		return(0.);
	}

    if (fabs(fb) <= tol)
        return (b);
    if (fabs(fa) <= tol)
        return (a);

	if (fb*fa > 0) {
		perror("zerobr: root not spanned");
//		perror("root not spanned:\n\ta %g, b %g, f(a) %g, f(b) %g",
//			a, b, fa, fb);
//		errno = ERROR;
		return(0.);
	}

	while (maxfun--) {

		if ((fb > 0 && fc > 0)  ||  (fb <= 0  && fc <= 0))  {
			c = a;
			fc = fa;
			d = e = b - a;
		}

        if (fabs(fc) < fabs(fb)) {
            a  = b;
            b  = c;
            c  = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }

        tol = meps * fabs(b) + t;
        m   = (c - b) / 2;

        if (fabs(m) < tol || fb == 0)
            return (b);

        /* see if bisection is forced */
        if (fabs(e) < tol || fabs(fa) <= fabs(fb))
            d = e = m;

		else {
			s = fb/fa;

			if (a == c) {	/* linear interpolation */
				p = 2 * m * s;
				q = 1 - s;
			}

			else {		/* inverse quadratic interpolation */
				q = fa/fc;
				r = fb/fc;
				p = s * (2 * m * q * (q-r) - (b-a) * (r-1));
				q -= 1;
				q *= (r-1) * (s-1);
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

		a = b;
		fa = fb;

        if (fabs(d) > tol)
            b += d;
        else if (m > 0)
            b += tol;
        else
            b -= tol;

		fb = satm(b);
		if (errno) {
			return(0.);
		}
	}
	perror("did not converge");

//	errno = ERROR;
	return(0.);
}

