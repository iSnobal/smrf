import unittest

import numpy as np
import numpy.testing as npt

from smrf.envphys.vapor_pressure import saturation_vapor_pressure


class TestVaporPressure(unittest.TestCase):
    def test_vapor_pressure(self):
        # Temperatures in C: 25, nan, 0, -10
        temperatures = np.array([[298.15, -1], [273.15, 263.15]])
        expected_pressures = np.array([[3166.703565, np.nan], [610.207270, 259.471371]])

        npt.assert_allclose(saturation_vapor_pressure(temperatures), expected_pressures)
