import unittest

import numpy as np

from smrf.envphys.albedo import decay_burned

ALBEDO_VIS = np.array([[0.8, 0.6], [0.7, 0.5]])
ALBEDO_IR = np.array([[0.7, 0.5], [0.6, 0.4]])
LAST_SNOW = np.array([[1.0, 1.5], [0.0, 2.0]])
K_BURNED = 0.06
K_UNBURNED = 0.02


class TestDecayBurned(unittest.TestCase):
    def test_decay_burned_with_burned_and_unburned_areas(self):
        burn_mask = np.array([[1, 0], [0, 1]])

        alb_vis, alb_ir = decay_burned(
            ALBEDO_VIS, ALBEDO_IR, LAST_SNOW, burn_mask, K_BURNED, K_UNBURNED
        )

        np.testing.assert_array_almost_equal(
            np.array([[0.753412, 0.582267], [0.7, 0.44346]]), alb_vis, decimal=6
        )
        np.testing.assert_array_almost_equal(
            np.array([[0.659235, 0.485223], [0.6, 0.354768]]), alb_ir, decimal=6
        )

    def test_decay_burned_all_burned(self):
        burn_mask = np.ones_like(ALBEDO_VIS)

        alb_vis, alb_ir = decay_burned(
            ALBEDO_VIS, ALBEDO_IR, LAST_SNOW, burn_mask, K_BURNED, K_UNBURNED
        )

        np.testing.assert_array_almost_equal(
            np.array([[0.753412, 0.548359], [0.7, 0.44346]]), alb_vis, decimal=6
        )
        np.testing.assert_array_almost_equal(
            np.array([[0.659235, 0.456966], [0.6, 0.354768]]), alb_ir, decimal=6
        )

    def test_decay_burned_all_unburned(self):
        burn_mask = np.zeros_like(ALBEDO_VIS)

        alb_vis, alb_ir = decay_burned(
            ALBEDO_VIS, ALBEDO_IR, LAST_SNOW, burn_mask, K_BURNED, K_UNBURNED
        )

        np.testing.assert_array_almost_equal(
            np.array([[0.784159, 0.582267], [0.7, 0.480395]]), alb_vis, decimal=6
        )
        np.testing.assert_array_almost_equal(
            np.array([[0.686139, 0.485223], [0.6, 0.384316]]), alb_ir, decimal=6
        )
