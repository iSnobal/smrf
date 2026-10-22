import unittest

import numpy as np
import numpy.testing as npt

from smrf.envphys.solar.twostream import twostream

OUTPUT_KEYS = [
    "reflectance",
    "transmittance",
    "direct_transmittance",
    "upwelling_irradiance",
    "irradiance_at_bottom",
    "irradiance_normal_to_beam",
]

S0 = 1000.0
# Optical depth shaped like a small DEM
TAU_GRID = np.array([[0.15, 0.2, 0.25], [0.2, 0.3, 0.35]])


class TestTwoStream(unittest.TestCase):
    def assert_zero_outputs(self, result, shape):
        self.assertCountEqual(OUTPUT_KEYS, result.keys())
        for key in OUTPUT_KEYS:
            self.assertIsInstance(result[key], np.ndarray)
            self.assertEqual(result[key].dtype, float)
            self.assertEqual(result[key].shape, shape)
            npt.assert_array_equal(result[key], 0.0)

    def test_sun_up(self):
        """
        Scalar tau returns float arrays of shape (1,) with physically bounded values.
        """
        cosz = 0.5

        result = twostream(cosz, S0, tau=0.2)

        self.assertCountEqual(OUTPUT_KEYS, result.keys())
        for key in OUTPUT_KEYS:
            self.assertIsInstance(result[key], np.ndarray)
            self.assertEqual(result[key].shape, (1,))

        self.assertGreater(result["irradiance_at_bottom"][0], 0.0)
        self.assertLessEqual(result["irradiance_at_bottom"][0], S0 * cosz)
        for key in ["reflectance", "transmittance", "direct_transmittance"]:
            self.assertTrue(0.0 <= result[key][0] <= 1.0)

    def test_sun_up_grid(self):
        """
        A DEM-shaped tau returns outputs of the same shape, as used by Elevrad.
        """
        result = twostream(0.5, S0, tau=TAU_GRID.copy())

        for key in OUTPUT_KEYS:
            self.assertEqual(result[key].shape, TAU_GRID.shape)

        self.assertTrue(np.all(result["irradiance_at_bottom"] > 0.0))

    def test_sun_exactly_on_horizon(self):
        """
        A cosz of exactly zero returns zero arrays instead of failing.
        """
        self.assert_zero_outputs(twostream(0.0, S0, tau=0.2), (1,))

    def test_sun_down(self):
        """
        A negative cosz returns zero arrays with the same shape as tau.
        """
        self.assert_zero_outputs(twostream(-0.1, S0, tau=0.2), (1,))

    def test_sun_down_grid(self):
        """
        A DEM-shaped tau returns zero arrays of the same shape for cosz of 0 and below.
        """
        for cosz in [0.0, -0.1]:
            result = twostream(cosz, S0, tau=TAU_GRID.copy())

            self.assert_zero_outputs(result, TAU_GRID.shape)

    def test_sun_down_outputs_are_independent(self):
        """
        Each zero output is its own array, so changing one does not affect the others.
        """
        result = twostream(0.0, S0, tau=0.2)

        result["reflectance"][0] = 1.0

        for key in OUTPUT_KEYS[1:]:
            npt.assert_array_equal(result[key], 0.0)
