import unittest
from unittest import mock

import numpy as np
import numpy.testing as npt
from numpy.polynomial import Polynomial

from smrf.envphys.air_temp import (
    CURVE_K,
    ELEVATION_Z0,
    adjust_by_elevation,
    adjust_lapse_rate,
)

CONFIG = {
    "time": {
        "start_date": "2025-09-20 00:00",
        "time_zone": "utc",
    },
}
DEM = np.array([[1000, 700], [500, 900]])
# Multiplier is from the lookup table in the distribute class
MULTIPLIER = 0.8856932782309219


class TestEnvphysAirTemp(unittest.TestCase):
    def test_adjust_lapse_rate(self) -> None:
        orig_temp = np.array([[1.0, 2.0], [3.0, 2.0]])
        air_temp = orig_temp.copy()

        adjust_lapse_rate(air_temp, DEM, MULTIPLIER)

        poly_1st = Polynomial.fit(DEM.ravel(), orig_temp.ravel(), 1)

        # Math of the method
        orig_temp -= poly_1st(DEM.ravel()).reshape(DEM.shape)
        b, m = poly_1st.convert().coef
        slope = m * MULTIPLIER

        orig_temp += slope * DEM + b

        # Allow floating point noise
        npt.assert_allclose(air_temp, orig_temp, atol=1e-15, rtol=0)

    @mock.patch("smrf.envphys.air_temp.adjust_lapse_rate")
    def test_adjust_by_elevation(self, mock_lapse_rate) -> None:
        orig_temp = np.array([[1.0, 2.0], [3.0, 2.0]])
        adjusted_temp = np.array([[1.2, 2.15], [3.05, 2.1]])

        mock_lapse_rate.return_value = adjusted_temp

        new_temp = orig_temp + (0.2 / (1 + np.exp(-CURVE_K * (DEM - ELEVATION_Z0))))
        temperature = adjust_by_elevation(orig_temp, DEM, MULTIPLIER)

        npt.assert_allclose(new_temp, temperature, atol=1e-15, rtol=0)
