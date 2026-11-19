import unittest

import numpy as np
import numpy.testing as npt

from smrf.envphys.constants import EPS
from smrf.envphys.vapor_pressure import (
    saturation_vapor_pressure,
    vapor_pressure_from_mixing_ratio,
)


class TestVaporPressure(unittest.TestCase):
    def test_vapor_pressure(self):
        # Temperatures in C: 25, nan, 0, -10
        temperatures = np.array([[298.15, -1], [273.15, 263.15]])
        expected_pressures = np.array([[3166.703565, np.nan], [610.207270, 259.471371]])

        npt.assert_allclose(saturation_vapor_pressure(temperatures), expected_pressures, equal_nan=True)


class TestVaporPressureFromMixingRatio(unittest.TestCase):
    def test_epsilon_is_ratio_of_molecular_weights(self):
        # Water vapor and dry air molecular weights give about 0.622
        self.assertAlmostEqual(EPS, 0.622, places=3)

    def test_zero_mixing_ratio_gives_zero_vapor_pressure(self):
        mixing_ratio = np.array([0.0])
        pressure = np.array([101325.0])

        vapor = vapor_pressure_from_mixing_ratio(mixing_ratio, pressure)

        npt.assert_allclose(vapor, [0.0])

    def test_inverts_mixing_ratio_definition(self):
        # Dalton's law: w = EPS * e / (p - e), so the conversion must return e
        pressure = np.array([101325.0, 85000.0])
        vapor = np.array([2339.0, 1000.0])
        mixing_ratio = EPS * vapor / (pressure - vapor)

        result = vapor_pressure_from_mixing_ratio(mixing_ratio, pressure)

        npt.assert_allclose(result, vapor, rtol=1e-12)

    def test_saturation_at_20_c_round_trip(self):
        # Saturation vapor pressure at 20 C gives a mixing ratio near 14.7 g/kg
        saturation = saturation_vapor_pressure(np.array([[293.15]]))
        pressure = np.array([[101325.0]])
        mixing_ratio = EPS * saturation / (pressure - saturation)

        result = vapor_pressure_from_mixing_ratio(mixing_ratio, pressure)

        npt.assert_allclose(result, saturation, rtol=1e-12)
        npt.assert_allclose(mixing_ratio, [[0.0147]], rtol=1e-2)

    def test_keeps_array_shape(self):
        mixing_ratio = np.full((2, 3), 0.005)
        pressure = np.full((2, 3), 90000.0)

        vapor = vapor_pressure_from_mixing_ratio(mixing_ratio, pressure)

        self.assertEqual(vapor.shape, (2, 3))
