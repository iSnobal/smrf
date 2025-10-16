import unittest
<<<<<<< HEAD
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytz

from smrf.distribute.albedo import Albedo
from smrf.tests.smrf_config import SMRFConfig

TOPO = MagicMock(
    veg_type=MagicMock("veg_type"),
)
CONFIG = {
    "time": {
        "time_zone": "utc",
        "start_date": "2025-10-01 00:00",
    },
    "albedo": {
        "decay_method": "date_method",
    },
}
TIMEZONE = pytz.timezone(CONFIG["time"]["time_zone"])
TIMESTEP = pd.to_datetime("2025-10-01 00:00:00")
STORM_DAY = np.array([0, 1, 0, 0, 5, 0])


class TestAlbedo(SMRFConfig, unittest.TestCase):
    def test_init_default(self):
        subject = Albedo(CONFIG, TOPO)

        self.assertIsNone(subject.albedo)
        self.assertIsNone(subject.albedo_vis)
        self.assertIsNone(subject.albedo_ir)
        self.assertIsNone(subject.albedo_direct)
        self.assertIsNone(subject.albedo_diffuse)
        self.assertIsNone(subject.source_files)

        # Check inheritance to variable base
        self.assertEqual(TIMESTEP.strftime("%Y%m%d"), subject.start_date)
        self.assertEqual(TIMEZONE, subject.time_zone)

    @patch("smrf.distribute.variable_base.ReadNetCDF")
    def test_distribute_file_broadband(self, mock_read_netcdf):
        values = np.array([0.9, 0.5])
        mock_source = MagicMock()
        mock_source.variables = ["albedo"]
        mock_source.load.return_value = values
        mock_read_netcdf.return_value = mock_source

        config = self._copy_config(CONFIG)
        config["albedo"]["source_files"] = "path/to/files"
        subject = Albedo(config, TOPO)
        subject.initialize(pd.DataFrame())

        subject.distribute(TIMESTEP, np.ndarray([10]), STORM_DAY)

        npt.assert_equal(subject.albedo, values)
        mock_source.load.assert_called_with("albedo", TIMESTEP)

    @patch("smrf.distribute.variable_base.ReadNetCDF")
    def test_distribute_file_vis_ir(self, mock_read_netcdf):
        values_vis = np.array([0.8, 0.7])
        values_ir = np.array([0.9, 0.5])
        mock_source = MagicMock()
        mock_source.variables = ["albedo_vis", "albedo_ir", "albedo"]
        mock_source.load.side_effect = [values_vis, values_ir]
        mock_read_netcdf.return_value = mock_source

        config = self._copy_config(CONFIG)
        config["albedo"]["source_files"] = "path/to/files"
        subject = Albedo(config, TOPO)
        subject.initialize(pd.DataFrame())

        subject.distribute(TIMESTEP, np.ndarray([10]), STORM_DAY)

        np.testing.assert_array_equal(subject.albedo_vis, values_vis)
        np.testing.assert_array_equal(subject.albedo_ir, values_ir)
        # Test that the priority is given to vis and ir
        self.assertIsNone(subject.albedo)

        mock_source.load.assert_any_call("albedo_vis", TIMESTEP)
        mock_source.load.assert_any_call("albedo_ir", TIMESTEP)

    @patch("smrf.distribute.variable_base.ReadNetCDF")
    def test_distribute_file_direct_diffuse(self, mock_read_netcdf):
        values_direct = np.array([0.85, 0.75])
        values_diffuse = np.array([0.95, 0.59])
        mock_source = MagicMock()
        mock_source.variables = ["albedo_direct", "albedo_diffuse", "albedo"]
        mock_source.load.side_effect = [values_direct, values_diffuse]
        mock_read_netcdf.return_value = mock_source

        config = self._copy_config(CONFIG)
        config["albedo"]["source_files"] = "path/to/files"
        subject = Albedo(config, TOPO)
        subject.initialize(pd.DataFrame())

        subject.distribute(TIMESTEP, np.ndarray([10]), STORM_DAY)

        np.testing.assert_array_equal(subject.albedo_direct, values_direct)
        np.testing.assert_array_equal(subject.albedo_diffuse, values_diffuse)
        # Test that the priority is given to direct and diffuse
        self.assertIsNone(subject.albedo)

        mock_source.load.assert_any_call("albedo_direct", TIMESTEP)
        mock_source.load.assert_any_call("albedo_diffuse", TIMESTEP)

    @patch("smrf.envphys.albedo.decay_alb_power")
    @patch("smrf.envphys.albedo.albedo")
    def test_distribute_calculated(self, mock_albedo_calc, mock_albedo_decay):
        values_vis = np.array([0.8, 0.7])
        values_ir = np.array([0.9, 0.5])
        mock_albedo_calc.return_value = (values_vis, values_ir)

        values_vis_decay = np.array([0.75, 0.65])
        values_ir_decay = np.array([0.85, 0.45])
        mock_albedo_decay.return_value = (values_vis_decay, values_ir_decay)

        config = self._copy_config(CONFIG)
        config["albedo"]["grain_size"] = "100"
        config["albedo"]["max_grain"] = "700"
        config["albedo"]["dirt"] = "1.2"
        config["albedo"]["date_method_start_decay"] = "2026-04-01"
        config["albedo"]["date_method_end_decay"] = "2026-06-30"
        config["albedo"]["date_method_decay_power"] = "1.2"

        subject = Albedo(config, TOPO)
        subject.initialize(pd.DataFrame())

        subject.distribute(TIMESTEP, np.ndarray([10]), STORM_DAY)

        mock_albedo_calc.assert_called_once()
        mock_albedo_decay.assert_called_once()
        np.testing.assert_array_equal(subject.albedo_vis, values_vis_decay)
        np.testing.assert_array_equal(subject.albedo_ir, values_ir_decay)
=======
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

from smrf.distribute.albedo import Albedo

CONFIG = {
    "albedo": {
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
}
DECAY_TIME = pd.to_datetime("2025-04-30")
NO_DECAY_TIME = pd.to_datetime("2025-03-01")
STORM_DAYS = np.array([[0.0, 1.0], [1.0, 0.0]])
COS_Z = np.array([[10.0, 10.0], [10.0, 10.0]])
TOPO = MagicMock(
    veg_type=MagicMock("veg_type"),
)
DATA = MagicMock()

ALBEDO_VIS = np.array([[0.0, 1.0], [1.0, 0.0]])
ALBEDO_IR = np.array([[0.0, 1.0], [1.0, 0.0]])


class TestAlbedo(unittest.TestCase):
    def setUp(self):
        self.subject = Albedo(CONFIG, TOPO)
        self.subject.initialize(DATA)

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
        self.subject.distribute(NO_DECAY_TIME, COS_Z, STORM_DAYS)

        envphys_albedo.assert_called()
        envphys_albedo.assert_called_with(
            STORM_DAYS,
            COS_Z,
            CONFIG["albedo"]["grain_size"],
            CONFIG["albedo"]["max_grain"],
            CONFIG["albedo"]["dirt"]
        )
        decay_alb_power.assert_not_called()

    @patch('smrf.distribute.albedo.albedo.albedo', return_value=(ALBEDO_VIS, ALBEDO_IR))
    @patch('smrf.distribute.albedo.albedo.decay_alb_power', return_value=(ALBEDO_VIS, ALBEDO_IR))
    def test_distribute_date_method_in_window(self, decay_alb_power, _envphys_albedo):
        self.subject.config["date_method_decay_power"] = 0.7
        self.subject.config["date_method_veg_default"] = 0.2
        self.subject.distribute(DECAY_TIME, COS_Z, STORM_DAYS)

        decay_alb_power.assert_called_with(
            {"default": 0.2}, TOPO.veg_type, 696.0, 2184.0, 0.7, ALBEDO_VIS, ALBEDO_IR
        )

    @patch('smrf.distribute.albedo.albedo.albedo', return_value=(ALBEDO_VIS, ALBEDO_IR))
    @patch('smrf.distribute.albedo.albedo.decay_burned')
    def test_distribute_post_fire_not_in_window(self, decay_burned, envphys_albedo):
        self.subject.config["decay_method"] = "post_fire"
        self.subject.distribute(NO_DECAY_TIME, COS_Z, STORM_DAYS)

        envphys_albedo.assert_called()
        envphys_albedo.assert_called_with(
            STORM_DAYS,
            COS_Z,
            CONFIG["albedo"]["grain_size"],
            CONFIG["albedo"]["max_grain"],
            CONFIG["albedo"]["dirt"]
        )
        decay_burned.assert_not_called()

    @patch('smrf.distribute.albedo.albedo.albedo', return_value=(ALBEDO_VIS, ALBEDO_IR))
    @patch('smrf.distribute.albedo.albedo.decay_burned', return_value=(ALBEDO_VIS, ALBEDO_IR))
    def test_distribute_post_fire_in_window(self, decay_burned, _envphys_albedo):
        self.subject.config["decay_method"] = "post_fire"
        self.subject.config["post_fire_k_burned"] = 0.06
        self.subject.config["post_fire_k_unburned"] = 0.02
        self.subject.distribute(DECAY_TIME, COS_Z, STORM_DAYS)

        decay_burned.assert_called_with(
            ALBEDO_VIS, ALBEDO_IR, STORM_DAYS, TOPO.burn_mask, 0.06, 0.02
        )
>>>>>>> 2da0b90 (Albedo - Add new decay function based on binary burn mask)
