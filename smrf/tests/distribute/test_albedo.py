import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

from smrf.distribute.albedo import Albedo

CONFIG = {
    "decay_method": "date_method",
    "decay_start": pd.to_datetime("2025-04-01"),
    "decay_end": pd.to_datetime("2025-07-01"),
    "grain_size": 100.0,
    "max_grain": 800.0,
    "max": 1.0,
    "min": 0.0,
    "dirt": 2,
    "date_method_veg_default": 0.2,
}
DECAY_TIME = pd.to_datetime("2025-04-30")
NO_DECAY_TIME = pd.to_datetime("2025-03-01")
STORM_DAYS = np.array([[0.0, 1.0], [1.0, 0.0]])
TOPO = MagicMock(
    veg_type=MagicMock("veg_type"),
)
DATA = MagicMock()

ALBEDO_VIS = np.array([[0.0, 1.0], [1.0, 0.0]])
ALBEDO_IR = np.array([[0.0, 1.0], [1.0, 0.0]])


class TestAlbedo(unittest.TestCase):
    def setUp(self):
        self.subject = Albedo(CONFIG)
        self.subject.initialize(TOPO, DATA)

    def test_before_decay_window(self):
        current, decay = self.subject.decay_window(NO_DECAY_TIME)
        self.assertEqual(-1, current)

    def test_in_decay_window(self):
        current, decay = self.subject.decay_window(DECAY_TIME)
        self.assertEqual(696.0, current)
        self.assertEqual(2184.0, decay)

    @patch('smrf.distribute.albedo.albedo.albedo', return_value=(ALBEDO_VIS, ALBEDO_IR))
    @patch('smrf.distribute.albedo.albedo.decay_alb_power')
    def test_distribute_date_method_not_in_window(self, decay_alb_power, envphys_albedo):
        self.subject.distribute(NO_DECAY_TIME, 1, STORM_DAYS)

        envphys_albedo.assert_called()
        envphys_albedo.assert_called_with(
            STORM_DAYS, 1, CONFIG["grain_size"], CONFIG["max_grain"], CONFIG["dirt"]
        )
        decay_alb_power.assert_not_called()

    @patch('smrf.distribute.albedo.albedo.albedo', return_value=(ALBEDO_VIS, ALBEDO_IR))
    @patch('smrf.distribute.albedo.albedo.decay_alb_power', return_value=(ALBEDO_VIS, ALBEDO_IR))
    def test_distribute_date_method_in_window(self, decay_alb_power, _envphys_albedo):
        self.subject.config["date_method_decay_power"] = 0.7
        self.subject.config["date_method_veg_default"] = 0.2
        self.subject.distribute(DECAY_TIME, 1, STORM_DAYS)

        decay_alb_power.assert_called_with(
            {"default": 0.2}, TOPO.veg_type, 696.0, 2184.0, 0.7, ALBEDO_VIS, ALBEDO_IR
        )

    @patch('smrf.distribute.albedo.albedo.albedo', return_value=(ALBEDO_VIS, ALBEDO_IR))
    @patch('smrf.distribute.albedo.albedo.decay_burned')
    def test_distribute_post_fire_not_in_window(self, decay_burned, envphys_albedo):
        self.subject.config["decay_method"] = "post_fire"
        self.subject.distribute(NO_DECAY_TIME, 1, STORM_DAYS)

        envphys_albedo.assert_called()
        envphys_albedo.assert_called_with(
            STORM_DAYS, 1, CONFIG["grain_size"], CONFIG["max_grain"], CONFIG["dirt"]
        )
        decay_burned.assert_not_called()

    @patch('smrf.distribute.albedo.albedo.albedo', return_value=(ALBEDO_VIS, ALBEDO_IR))
    @patch('smrf.distribute.albedo.albedo.decay_burned', return_value=(ALBEDO_VIS, ALBEDO_IR))
    def test_distribute_post_fire_in_window(self, decay_burned, _envphys_albedo):
        self.subject.config["decay_method"] = "post_fire"
        self.subject.config["post_fire_k_burned"] = 0.06
        self.subject.config["post_fire_k_unburned"] = 0.02
        self.subject.distribute(DECAY_TIME, 1, STORM_DAYS)

        decay_burned.assert_called_with(
            ALBEDO_VIS, ALBEDO_IR, STORM_DAYS, TOPO.burn_mask, 0.06, 0.02
        )
