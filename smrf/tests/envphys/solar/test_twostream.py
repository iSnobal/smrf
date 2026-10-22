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
TAU = np.array([[0.15, 0.2, 0.25], [0.2, 0.3, 0.35]])


class TestTwoStream(unittest.TestCase):
    def test_ipw_example_mid_sun(self):
        """
        Results match the IPW twostream man page example:
        twostream -u 0.68 -t 0.2 -w 0.85 -g 0.3 -r 0.5 -s 159
        """
        result = twostream(0.68, 159, tau=np.array([0.2]), omega=0.85, g=0.3, R0=0.5)

        # IPW prints with %g, i.e. six significant digits
        npt.assert_allclose(
            [result[key][0] for key in OUTPUT_KEYS],
            [0.472814, 0.912634, 0.745189, 51.1207, 98.674, 118.485],
            rtol=1e-5,
        )

    def test_ipw_example_low_sun(self):
        """
        Results match the IPW twostream man page example:
        twostream -u 0.05 -t 0.3 -w 0.8 -g 0.35 -r 0.6 -s 60
        """
        result = twostream(0.05, 60, tau=np.array([0.3]), omega=0.8, g=0.35, R0=0.6)

        npt.assert_allclose(
            [result[key][0] for key in OUTPUT_KEYS],
            [0.577519, 0.365645, 0.00247875, 1.73256, 1.09694, 0.148725],
            rtol=1e-5,
        )

    def test_sun_up_bounds(self):
        """
        The irradiance at the bottom is positive and cannot exceed the beam
        projected onto the horizontal surface.
        """
        cosz = 0.5

        result = twostream(cosz, S0, tau=np.array([0.2]))

        npt.assert_array_less(0.0, result["irradiance_at_bottom"])
        npt.assert_array_less(result["irradiance_at_bottom"], S0 * cosz)

    def test_sun_up(self):
        """
        A DEM-shaped tau returns outputs of the same shape, as used by Elevrad.
        """
        result = twostream(0.5, S0, tau=TAU.copy())

        self.assertEqual(
            {key: TAU.shape for key in OUTPUT_KEYS},
            {key: value.shape for key, value in result.items()},
        )

    def test_sun_on_horizon(self):
        """
        A cosz of exactly zero returns zero arrays instead of failing.
        """
        result = twostream(0.0, S0, tau=np.array([0.2]))

        npt.assert_equal(result, {key: np.zeros(1) for key in OUTPUT_KEYS})

    def test_sun_down(self):
        """
        A negative cosz returns zero arrays with the same shape as a DEM-shaped tau.
        """
        result = twostream(-0.1, S0, tau=TAU.copy())

        npt.assert_equal(result, {key: np.zeros(TAU.shape) for key in OUTPUT_KEYS})

    def test_sun_down_outputs_are_independent(self):
        """
        Each zero output is its own array, so changing one does not affect the others.
        """
        result = twostream(0.0, S0, tau=np.array([0.2]))

        result["reflectance"][0] = 1.0

        expected = {key: np.zeros(1) for key in OUTPUT_KEYS}
        expected["reflectance"] = np.ones(1)
        npt.assert_equal(result, expected)
