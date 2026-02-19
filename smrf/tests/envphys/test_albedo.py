import unittest

import numpy as np
import numpy.testing as npt

from smrf.envphys.albedo import decay_alb_power, decay_burned

ALBEDO_VIS = np.array([[0.8, 0.6], [0.7, 0.5]])
ALBEDO_IR = np.array([[0.7, 0.5], [0.6, 0.4]])
LAST_SNOW = np.array([[1.0, 1.5], [0.0, 2.0]])
BURNED_NO_SNOWFALL = np.array([[0.0, 0.0], [0.0, 0.0]])
K_BURNED = 0.06


class TestDecayAlbPower(unittest.TestCase):
    def setUp(self):
        self.veg = {
            "default": 0.2,
            "41": 0.25,
            "42": 0.3,
        }
        self.veg_type = np.array(
            [
                [0, 41],
                [42, 0],
            ]
        )

    def test_max_decay_after_end(self):
        """Test with current_hours > decay_hours, ensuring maximum decay is applied."""
        current_hours = 150
        decay_hours = 100
        pwr = 1.0

        expected_v = ALBEDO_VIS - np.array(
            [
                [0.2, 0.25],
                [0.3, 0.2],
            ]
        )
        expected_ir = ALBEDO_IR - np.array(
            [
                [0.2, 0.25],
                [0.3, 0.2],
            ]
        )

        alb_v_d, alb_ir_d = decay_alb_power(
            self.veg,
            self.veg_type,
            current_hours,
            decay_hours,
            pwr,
            ALBEDO_VIS.copy(),
            ALBEDO_IR.copy(),
            BURNED_NO_SNOWFALL,
        )

        npt.assert_array_almost_equal(expected_v, alb_v_d)
        npt.assert_array_almost_equal(expected_ir, alb_ir_d)

    def test_power_decay_during_window(self):
        """Test power function decay during the decay period."""
        current_hours = 50
        decay_hours = 100
        pwr = 2.0

        # Decay rates
        tao_default = decay_hours / (self.veg["default"] ** (1.0 / pwr))
        tao_41 = decay_hours / (self.veg["41"] ** (1.0 / pwr))
        tao_42 = decay_hours / (self.veg["42"] ** (1.0 / pwr))

        expected_decay = np.zeros_like(self.veg_type, dtype=float)
        expected_decay[self.veg_type == 0] = (current_hours / tao_default) ** pwr
        expected_decay[self.veg_type == 41] = (current_hours / tao_41) ** pwr
        expected_decay[self.veg_type == 42] = (current_hours / tao_42) ** pwr

        expected_v = ALBEDO_VIS - expected_decay
        expected_ir = ALBEDO_IR - expected_decay

        alb_v_d, alb_ir_d = decay_alb_power(
            self.veg,
            self.veg_type,
            current_hours,
            decay_hours,
            pwr,
            ALBEDO_VIS.copy(),
            ALBEDO_IR.copy(),
            BURNED_NO_SNOWFALL,
        )

        npt.assert_array_almost_equal(expected_v, alb_v_d, decimal=5)
        npt.assert_array_almost_equal(expected_ir, alb_ir_d, decimal=5)

    def test_power_decay_during_window_with_burn(self):
        """Burned pixels will not be changed by power decay after snowfall"""
        current_hours = 50
        decay_hours = 100
        pwr = 2.0
        burned_no_snowfall = np.array([[0.0, 1.0], [0.0, 0.0]])

        # Decay rates
        tao_default = decay_hours / (self.veg["default"] ** (1.0 / pwr))
        tao_41 = decay_hours / (self.veg["41"] ** (1.0 / pwr))
        tao_42 = decay_hours / (self.veg["42"] ** (1.0 / pwr))

        expected_decay = np.zeros_like(self.veg_type, dtype=float)
        expected_decay[self.veg_type == 0] = (current_hours / tao_default) ** pwr
        expected_decay[self.veg_type == 41] = (current_hours / tao_41) ** pwr
        expected_decay[self.veg_type == 42] = (current_hours / tao_42) ** pwr

        # Burned pixels should not be changed by power decay
        expected_v = ALBEDO_VIS - expected_decay
        expected_v[0][1] = ALBEDO_VIS[0][1]
        expected_ir = ALBEDO_IR - expected_decay
        expected_ir[0][1] = ALBEDO_IR[0][1]

        alb_v_d, alb_ir_d = decay_alb_power(
            self.veg,
            self.veg_type,
            current_hours,
            decay_hours,
            pwr,
            ALBEDO_VIS.copy(),
            ALBEDO_IR.copy(),
            burned_no_snowfall,
        )

        npt.assert_array_almost_equal(expected_v, alb_v_d, decimal=5)
        npt.assert_array_almost_equal(expected_ir, alb_ir_d, decimal=5)

    def test_partial_coverage_during_decay(self):
        """Test decay for mixed vegetation types."""
        current_hours = 75
        decay_hours = 150
        pwr = 1.5

        # Calculate decay
        tao_default = decay_hours / (self.veg["default"] ** (1.0 / pwr))
        decay_default = (current_hours / tao_default) ** pwr

        tao_41 = decay_hours / (self.veg["41"] ** (1.0 / pwr))
        tao_42 = decay_hours / (self.veg["42"] ** (1.0 / pwr))

        expected_decay = np.zeros_like(self.veg_type, dtype=float)
        expected_decay[self.veg_type == 0] = decay_default
        expected_decay[self.veg_type == 41] = (current_hours / tao_41) ** pwr
        expected_decay[self.veg_type == 42] = (current_hours / tao_42) ** pwr

        expected_v = ALBEDO_VIS - expected_decay
        expected_ir = ALBEDO_IR - expected_decay

        alb_v_d, alb_ir_d = decay_alb_power(
            self.veg,
            self.veg_type,
            current_hours,
            decay_hours,
            pwr,
            ALBEDO_VIS.copy(),
            ALBEDO_IR.copy(),
            BURNED_NO_SNOWFALL,
        )

        npt.assert_array_almost_equal(expected_v, alb_v_d, decimal=5)
        npt.assert_array_almost_equal(expected_ir, alb_ir_d, decimal=5)

class TestDecayBurned(unittest.TestCase):
    def test_decay_burned_with_burned_and_unburned_areas(self):
        burn_mask = np.array([[1, 0], [0, 1]])

        # Arrays are updated in place, so pass a copy
        alb_vis, alb_ir = decay_burned(
            ALBEDO_VIS.copy(), ALBEDO_IR.copy(), LAST_SNOW, burn_mask, K_BURNED
        )

        npt.assert_array_almost_equal(
            np.array([[0.753412, ALBEDO_VIS[0][1]], [ALBEDO_VIS[1][0], 0.44346]]), alb_vis, decimal=6
        )
        npt.assert_array_almost_equal(
            np.array([[0.659235, ALBEDO_IR[0][1]], [ALBEDO_IR[1][0], 0.354768]]), alb_ir, decimal=6
        )

    def test_decay_burned_all_burned(self):
        burn_mask = np.ones_like(ALBEDO_VIS)

        # Arrays are updated in place, so pass a copy
        alb_vis, alb_ir = decay_burned(
            ALBEDO_VIS.copy(), ALBEDO_IR.copy(), LAST_SNOW, burn_mask, K_BURNED
        )

        npt.assert_array_almost_equal(
            np.array([[0.753412, 0.548359], [0.7, 0.44346]]), alb_vis, decimal=6
        )
        npt.assert_array_almost_equal(
            np.array([[0.659235, 0.456966], [0.6, 0.354768]]), alb_ir, decimal=6
        )

    def test_decay_burned_all_unburned(self):
        burn_mask = np.zeros_like(ALBEDO_VIS)

        # Arrays are updated in place, so pass a copy
        alb_vis, alb_ir = decay_burned(
            ALBEDO_VIS.copy(), ALBEDO_IR.copy(), LAST_SNOW, burn_mask, K_BURNED
        )

        npt.assert_array_equal(ALBEDO_VIS, alb_vis)
        npt.assert_array_equal(ALBEDO_IR, alb_ir)

    def test_decay_burned_k_burned_none(self):
        burn_mask = np.array([[1, 0], [0, 1]])

        with self.assertRaises(ValueError) as context:
            decay_burned(
                ALBEDO_VIS, ALBEDO_IR, LAST_SNOW, burn_mask, None
            )

        self.assertIn("k_burned", str(context.exception))
