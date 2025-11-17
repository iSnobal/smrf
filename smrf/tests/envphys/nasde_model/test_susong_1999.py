from smrf.envphys.nasde_model import Susong1999
import unittest
import numpy as np
import numpy.testing as npt


class TestSusong1999(unittest.TestCase):
    TEMPERATURES = np.array([-6, -4, -2.5, -.75, -0.25, 0, 0.25, 2])
    PRECIPITATION = np.ones_like(TEMPERATURES)

    def test_returns_precipitation(self):
        results = Susong1999.run(self.TEMPERATURES, self.PRECIPITATION)

        self.assertTrue("rho_s" in results)
        self.assertTrue("pcs" in results)

        npt.assert_equal([1, 1, 1, 1, 0.75, 0.25, 0.25, 0], results['pcs'])
        npt.assert_equal([75, 100, 150, 175, 200, 250, 250, 0], results['rho_s'])

    def test_returns_no_precipitation(self):
        precipitation = np.zeros(self.TEMPERATURES.shape)

        results = Susong1999.run(self.TEMPERATURES, precipitation)

        self.assertTrue("rho_s" in results)
        self.assertTrue("pcs" in results)

        npt.assert_equal(precipitation, results['pcs'])
        npt.assert_equal(precipitation, results['rho_s'])
