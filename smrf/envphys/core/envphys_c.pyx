cimport numpy as np # noqa
cimport cython # noqa
import numpy as np # noqa

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()


cdef extern from "envphys_c.h":
    void topotherm(int ngrid, double *ta, double *tw, double *z, double *skvfac, int nthreads, double *thermal)
    void dew_point_t_for_grid(int ngrid, double *ea, double *dpt, int nthreads, double tol)
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


# https://github.com/cython/cython/wiki/tutorials-NumpyPointerToC
def ctopotherm(np.ndarray[double, mode="c", ndim=2] ta,
               np.ndarray[double, mode="c", ndim=2] tw,
               np.ndarray[double, mode="c", ndim=2] z,
               np.ndarray[double, mode="c", ndim=2] skvfac,
               np.ndarray[double, mode="c", ndim=2] thermal not None,
               int nthreads=1):
    '''
    Call the function krige_grid in krige.c which will iterate over the grid
    within the C code

    Args:
        ta, tw, z, skvfac
    Out:
        thermal changed in place

    20160325 Scott Havens
    '''

    cdef int ngrid
    ngrid = ta.shape[0] * ta.shape[1]

    # convert the ta array to C
    cdef np.ndarray[double, mode="c", ndim=2] ta_arr
    ta_arr = np.ascontiguousarray(ta, dtype=np.float64)

    # convert the tw array to C
    cdef np.ndarray[double, mode="c", ndim=2] tw_arr
    tw_arr = np.ascontiguousarray(tw, dtype=np.float64)

    # convert the ta array to C
    cdef np.ndarray[double, mode="c", ndim=2] z_arr
    z_arr = np.ascontiguousarray(z, dtype=np.float64)

    # convert the skvfac to C
    cdef np.ndarray[double, mode="c", ndim=2] skvfac_arr
    skvfac_arr = np.ascontiguousarray(skvfac, dtype=np.float64)

    # call the C function
    topotherm(ngrid, &ta_arr[0,0], &tw_arr[0,0], &z_arr[0,0], &skvfac_arr[0,0], nthreads, &thermal[0,0])

    return None


def cdewpt(
    double[:, ::1] vp,
    double[:, ::1] dew_point_temperatures not None,
    double tolerance=0.0,
    int n_threads=1
):
    """
    Args:
        vp - Vapor pressure
        dew_point_temperatures - Output array to write results to
        tolerance - Tolerance for convergence of dew point temperatures
        n_threads - Number of threads to use for parallel processing
    Returns:
        None - dew_point_temperatures changed in place
    """
    cdef int ngrid = vp.shape[0] * vp.shape[1]

    dew_point_t_for_grid(ngrid, &vp[0, 0], &dew_point_temperatures[0, 0], n_threads, tolerance)


# https://github.com/cython/cython/wiki/tutorials-NumpyPointerToC
def cwbt(np.ndarray[double, mode="c", ndim=2] ta,
         np.ndarray[double, mode="c", ndim=2] td,
         np.ndarray[double, mode="c", ndim=2] z,
         np.ndarray[double, mode="c", ndim=2] tw not None,
         float tolerance=0,
         int nthreads=1):
    '''
    Call the function iwbt in iwbt.c which will iterate over the grid
    within the C code

    Args:
        ta, td, z
    Out:
        tw changed in place (wet bulb temperature)

    20180611 Micah Sandusky
    '''

    cdef int ngrid
    ngrid = ta.shape[0] * ta.shape[1]

    # convert the ta array to C
    cdef np.ndarray[double, mode="c", ndim=2] ta_arr
    ta_arr = np.ascontiguousarray(ta, dtype=np.float64)

    # convert the dew point array to C
    cdef np.ndarray[double, mode="c", ndim=2] td_arr
    td_arr = np.ascontiguousarray(td, dtype=np.float64)

    # convert the ta array to C
    cdef np.ndarray[double, mode="c", ndim=2] z_arr
    z_arr = np.ascontiguousarray(z, dtype=np.float64)

    # call the C function
    iwbt(ngrid, &ta_arr[0,0], &td_arr[0,0], &z_arr[0,0], nthreads, tolerance, &tw[0,0])

    return None
