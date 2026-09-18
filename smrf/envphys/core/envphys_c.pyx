cimport cython
import numpy as np


cdef extern from "envphys_c.h":
    void topotherm(int ngrid, double *ta, double *tw, double *z, double *skvfac, int nthreads, double *thermal)
    void dewpt(int ngrid, double *ea, int nthreads, double tol, double *dpt)
    void iwbt(int ngrid, double *ta, double *td,	double *z, int nthreads, double tol, double *tw)
    double saturation_vapor_pressure(double *ta)


def svp_for_temperatures(double[:, ::1] temperature_k, double[:, ::1] output_array):
    """
    Wrapper function to calculate the saturation vapor pressure for given temperature.

    This is not parallelized since the math is very minimal and only serves to reduce
    duplicated code when trying to apply this to a numpy array.
    Main use:
    * :mod:`smrf.envphys.vapor_pressure`

    Args:
        temperature_k: Input array with air temperatures
        output_array: Output array to write the results to
    """
    cdef Py_ssize_t i, j
    cdef Py_ssize_t rows = temperature_k.shape[0]
    cdef Py_ssize_t cols = temperature_k.shape[1]
    cdef double value

    for i in range(rows):
        for j in range(cols):
            value = temperature_k[i, j]

            if value <= 0.0 or value != value:  # The latter checks for NaN
                output_array[i, j] = np.nan
            else:
                output_array[i, j] = saturation_vapor_pressure(&value)


def ctopotherm(double[:, ::1] ta,
               double[:, ::1] tw,
               double[:, ::1] z,
               double[:, ::1] skvfac,
               double[:, ::1] thermal,
               int nthreads=1):
    """
    Calculate clear-sky thermal radiation over a grid.
    Calls topotherm in topotherm.c

    Args:
        ta: Air temperature grid (K)
        tw: Dew point temperature grid (K)
        z: Elevation grid (m)
        skvfac: Sky view factor grid (0-1)
        thermal: Output array for thermal radiation (W/m^2), modified in place
        nthreads: Number of threads for parallel processing
    """
    cdef int ngrid = ta.shape[0] * ta.shape[1]

    topotherm(ngrid, &ta[0, 0], &tw[0, 0], &z[0, 0], &skvfac[0, 0], nthreads, &thermal[0, 0])


def cdewpt(double[:, ::1] vp, double[:, ::1] dwpt, float tolerance=0, int nthreads=1):
    """
    Calculate dew point temperature from vapor pressure over a grid.
    Calls dewpt in dewpt.c

    Args:
        vp: Vapor pressure grid (Pa)
        dwpt: Output array for dew point temperature (C), modified in place
        tolerance: Convergence tolerance threshold
        nthreads: Number of threads for parallel processing
    """
    cdef int ngrid = vp.shape[0] * vp.shape[1]

    dewpt(ngrid, &vp[0, 0], nthreads, tolerance, &dwpt[0, 0])


def cwbt(double[:, ::1] ta,
         double[:, ::1] td,
         double[:, ::1] z,
         double[:, ::1] tw,
         float tolerance=0,
         int nthreads=1):
    """
    Calculate wet bulb temperature from air temperature, dew point, and elevation over a grid.
    Calls iwbt in iwbt.c

    Args:
        ta: Air temperature grid (C)
        td: Dew point temperature grid (C)
        z: Elevation grid (m)
        tw: Output array for wet bulb temperature (C), modified in place
        tolerance: Convergence tolerance threshold
        nthreads: Number of threads for parallel processing
    """
    cdef int ngrid = ta.shape[0] * ta.shape[1]

    iwbt(ngrid, &ta[0, 0], &td[0, 0], &z[0, 0], nthreads, tolerance, &tw[0, 0])
