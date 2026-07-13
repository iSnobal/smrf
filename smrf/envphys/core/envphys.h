#include <math.h>

/*
 * Environmental physics.
 */

/* ----------------------------------------------------------------------- */

/*
 *  Constants
 */
 
/*
 *  molecular weight of air (kg / kmole)
 */
#define MOL_AIR         28.9644

/*
 *  molecular weight of water vapor (kg / kmole)
 */
#define MOL_H2O         18.0153

/*
 *  gas constant (J / kmole / deg)
 */
#define RGAS            8.31432e3

/*
 *  triple point of water at standard pressure (deg K)
 */
#define FREEZE          2.7316e2
#define BOIL            3.7315e2

/*
 *  specific heat of air at constant pressure (J / kg / deg)
 */
#define CP_AIR          1.005e3

/*
 *  specific heat of water at 0C (J / (kg K))
 */
#define CP_W0   	4217.7

/*
 *  standard sea level pressure (Pa)
 */
#define SEA_LEVEL       1.013246e5

/*
 *  standard sea level air temp (K)
 */
#define STD_AIRTMP      2.88e2

/*
 *  standard lapse rate (K/m)
 */
#define STD_LAPSE_M      -0.0065

/*
 *  standard lapse rate (K/km)
 */
#define STD_LAPSE       -6.5

/*
 *  gravitational acceleration at reference latitude 45d 32m 33s (m/s^2)
 */
#define GRAVITY		9.80665

/*
 * Stefan–Boltzmann constant
 */
#define STEF_BOLTZ 5.6697e-8

/* ------------------------------------------------------------------------ */

/*
 *  Macros
 */

/*
 *  integral of hydrostatic equation over layer with linear temperature
 *  variation
 *
 *	pb = base level pressure
 *	tb = base level temp (K)
 *	L  = lapse rate (deg/km)
 *	h  = layer thickness (km)
 *      g  = grav accel (m/s^2)
 *	m  = molec wt (kg/kmole)
 *
 *	(the factors 1.e-3 and 1.e3 are for units conversion)
 */
#define HYSTAT(pb,tb,L,h,g,m)           ((pb) * (((L)==0.) ?\
                exp(-(g)*(m)*(h)*1.e3/(RGAS*(tb))) :\
                pow((tb)/((tb)+(L)*(h)),(g)*(m)/(RGAS*(L)*1.e-3))))

/*
 *  latent heat of vaporization
 *
 *	t = temperature (K)
 */
#define LH_VAP(t)               (2.5e6 - 2.95573e3 *((t) - FREEZE))

/*
 *  latent heat of fusion
 *
 *	t = temperature (K)
 */
#define LH_FUS(t)               (3.336e5 + 1.6667e2 * (FREEZE - (t)))

/* ------------------------------------------------------------------------ */

