# cython: language_level=3str
# cython: embedsignature=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: binding=True

import numpy as np
cimport numpy as np
from cython.parallel cimport parallel, prange
from libc.stdlib cimport malloc, free
from libc.string cimport memset

np.import_array()

cdef Py_ssize_t NUM_ARRAYS = 6

cdef class TopoSplit:
    cdef:
        double[:,:] _sky_view_factor
        readonly Py_ssize_t ny, nx
        readonly float min_value
        readonly int num_threads

    def __init__(self, double[:,:] sky_view_factor, float min_value=1.0, int num_threads=1):
        """
        Parameters
        ----------
        sky_view_factor : ndarray
            2D array of sky view factors, determines grid dimensions
        min_value : float, optional
            Minimum value to calculate values for (default: 1.0)
        num_threads : int, optional
            Number of threads for parallel processing
        """
        self._sky_view_factor = sky_view_factor
        self.ny = sky_view_factor.shape[0]
        self.nx = sky_view_factor.shape[1]
        self.min_value = min_value
        self.num_threads = num_threads

    cdef int _process_row(
        self,
        Py_ssize_t row_idx,
        const double[:,:] dswrf,
        const double[:,:] direct_normal,
        const double[:,:] diffuse_horizontal,
        const double cos_z,
        const double[:,:] illumination_angles,
        const double[:,:] albedo_vis,
        const double[:,:] albedo_ir,
        double[:,:] results
    ) noexcept nogil:
        cdef:
            Py_ssize_t col, i, base_idx
            double ghi_vis, k_val
            double* row_arrays = <double*>malloc(self.nx * NUM_ARRAYS * sizeof(double))

        try:
            # Initialize all values to 0
            memset(row_arrays, 0, self.nx * NUM_ARRAYS * sizeof(double))

            # Process each column
            for col in range(self.nx):
                base_idx = col * NUM_ARRAYS

                # Only calculate for values above the minimum value (set in the initialize)
                # Interpolation in early morning or late evening can cause negative values
                # Keeping all values below than 1 as 0
                if dswrf[row_idx, col] > self.min_value:

                    # GHI
                    ghi_vis = direct_normal[row_idx, col] * cos_z + diffuse_horizontal[row_idx, col]
                    row_arrays[base_idx] = ghi_vis

                    # K (diffuse fraction)
                    k_val = diffuse_horizontal[row_idx, col] / ghi_vis
                    row_arrays[base_idx + 1] = k_val

                    # DHI and DNI
                    row_arrays[base_idx + 2] = dswrf[row_idx, col] * k_val
                    row_arrays[base_idx + 3] = (dswrf[row_idx, col] * (1.0 - k_val)) / cos_z

                    # HRRR solar
                    row_arrays[base_idx + 4] = (
                        row_arrays[base_idx + 3] * illumination_angles[row_idx, col] +
                        row_arrays[base_idx + 2] * self._sky_view_factor[row_idx, col]
                    )

                    # Net solar
                    row_arrays[base_idx + 5] = row_arrays[base_idx + 4] * (
                        0.54 * albedo_vis[row_idx, col] +
                        0.46 * albedo_ir[row_idx, col]
                    )

            # Copy results
            for col in range(self.nx):
                for i in range(NUM_ARRAYS):
                    results[row_idx * NUM_ARRAYS + i, col] = row_arrays[col * NUM_ARRAYS + i]

        finally:
            free(row_arrays)

        return 1

    def calculate(
        self,
        const double[:,:] dswrf,
        const double[:,:] direct_normal,
        const double[:,:] diffuse_horizontal,
        const double cos_z,
        const double[:,:] illumination_angles,
        const double[:,:] albedo_vis,
        const double[:,:] albedo_ir
    ):
        """"
        Parameters
        ----------
        dswrf : ndarray
            Downwelling shortwave radiation flux
        direct_normal : ndarray
            Direct normal irradiance
        diffuse_horizontal : ndarray
            Diffuse horizontal irradiance
        cos_z : float
            Cosine of solar zenith angle
        illumination_angles : ndarray
            Array of illumination angles
        albedo_vis : ndarray
            Visible albedo array
        albedo_ir : ndarray
            Infrared albedo array

        Returns
        -------
        dict
            Dictionary containing the calculated solar components
        """
        cdef:
            double[:,:] results
            np.ndarray[double, ndim=2] results_array
            Py_ssize_t slice_idx

        if cos_z <= 0:
            zero_array = np.zeros((self.ny, self.nx), dtype=np.float64)
            return {
                'solar_ghi_vis': zero_array.copy(),
                'solar_k': zero_array.copy(),
                'solar_dhi': zero_array.copy(),
                'solar_dni': zero_array.copy(),
                'hrrr_solar': zero_array.copy(),
                'net_solar': zero_array.copy()
            }

        results_array = np.zeros((self.ny * NUM_ARRAYS, self.nx), dtype=np.float64)
        results = results_array

        with nogil:
            for row_idx in prange(self.ny, num_threads=self.num_threads):
                self._process_row(
                    row_idx,
                    dswrf, direct_normal, diffuse_horizontal,
                    cos_z, illumination_angles,
                    albedo_vis, albedo_ir,
                    results
                )

        return {
            'ghi_vis': results_array[0:results_array.shape[0]:NUM_ARRAYS],
            'k': results_array[1:results_array.shape[0]:NUM_ARRAYS],
            'dhi': results_array[2:results_array.shape[0]:NUM_ARRAYS],
            'dni': results_array[3:results_array.shape[0]:NUM_ARRAYS],
            'solar': results_array[4:results_array.shape[0]:NUM_ARRAYS],
            'net_solar': results_array[5:results_array.shape[0]:NUM_ARRAYS]
        }
