import unittest
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
        "decay_start": pd.to_datetime("2025-04-01"),
        "decay_end": pd.to_datetime("2025-07-01"),
        "grain_size": 100.0,
        "max_grain": 800.0,
        "max": 1.0,
        "min": 0.0,
        "dirt": 2,
        "date_method_veg_default": 0.2,
        "date_method_decay_power": 1,
    },
}
TIMEZONE = pytz.timezone(CONFIG["time"]["time_zone"])
TIMESTEP = pd.to_datetime(CONFIG["time"]["start_date"])

DECAY_TIME = pd.to_datetime("2025-04-30")
NO_DECAY_TIME = pd.to_datetime("2025-03-01")

STORM_DAYS = np.array([[0.0, 1.0], [1.0, 0.0]])
COS_Z = np.array([[10.0, 10.0], [10.0, 10.0]])

DATA = MagicMock()

ALBEDO_VIS = np.array([[0.9, 1.0], [1.0, 0.8]])
ALBEDO_IR = np.array([[0.8, 1.0], [1.0, 0.9]])


class TestAlbedo(SMRFConfig, unittest.TestCase):
    def setUp(self):
        self.subject = Albedo(CONFIG, TOPO)
        self.subject.initialize(DATA)

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

    def test_load_burn_mask_with_defined_mask(self):
        self.subject.topo.burn_mask = np.array([[1, 0], [0, 1]])

        self.subject.load_burn_mask()

        np.testing.assert_array_equal(self.subject.burn_mask, self.subject.topo.burn_mask)

    def test_load_burn_mask_without_defined_mask(self):
        self.subject.topo.burn_mask = None
        self.subject.topo.dem = np.array([[1, 0], [0, 1]])

        self.subject.load_burn_mask()

        expected_burn_mask = np.zeros_like(self.subject.topo.dem)
        np.testing.assert_array_equal(self.subject.burn_mask, expected_burn_mask)

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

        subject.distribute(TIMESTEP, COS_Z, STORM_DAYS)

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

        subject.distribute(TIMESTEP, COS_Z, STORM_DAYS)

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

        subject.distribute(TIMESTEP, COS_Z, STORM_DAYS)

        np.testing.assert_array_equal(subject.albedo_direct, values_direct)
        np.testing.assert_array_equal(subject.albedo_diffuse, values_diffuse)
        # Test that the priority is given to direct and diffuse
        self.assertIsNone(subject.albedo)

        mock_source.load.assert_any_call("albedo_direct", TIMESTEP)
        mock_source.load.assert_any_call("albedo_diffuse", TIMESTEP)

    @patch("smrf.distribute.albedo.Albedo.date_method")
    @patch("smrf.envphys.albedo.albedo")
    def test_distribute_calculated(self, mock_albedo_calc, mock_date_method):
        mock_albedo_calc.return_value = (ALBEDO_VIS, ALBEDO_IR)
        date_albedo_vis = ALBEDO_VIS - 0.1
        date_albedo_ir = ALBEDO_IR - 0.2
        mock_date_method.return_value = (date_albedo_vis, date_albedo_ir)

        self.subject.distribute(TIMESTEP, COS_Z, STORM_DAYS)

        mock_albedo_calc.assert_called_once()
        mock_date_method.assert_called_once()

        npt.assert_array_equal(self.subject.albedo_vis, date_albedo_vis)
        npt.assert_array_equal(self.subject.albedo_ir, date_albedo_ir)

    def test_before_decay_window(self):
        current, decay = self.subject.decay_window(NO_DECAY_TIME)
        self.assertEqual(-1, current)

    def test_in_decay_window(self):
        current, decay = self.subject.decay_window(DECAY_TIME)
        self.assertEqual(696.0, current)
        self.assertEqual(2184.0, decay)

    @patch("smrf.distribute.albedo.albedo.albedo", return_value=(ALBEDO_VIS, ALBEDO_IR))
    @patch("smrf.distribute.albedo.Albedo.date_method")
    def test_distribute_date_method_not_in_window(
        self, date_decay_method, envphys_albedo
    ):
        self.subject.distribute(NO_DECAY_TIME, COS_Z, STORM_DAYS)

        envphys_albedo.assert_called()
        envphys_albedo.assert_called_with(
            STORM_DAYS,
            COS_Z,
            CONFIG["albedo"]["grain_size"],
            CONFIG["albedo"]["max_grain"],
            CONFIG["albedo"]["dirt"],
        )
        date_decay_method.assert_not_called()

    @patch("smrf.envphys.albedo.decay_alb_power")
    def test_date_method_no_post_fire(self, mock_decay_power):
        expected_v = ALBEDO_VIS - 0.05
        expected_ir = ALBEDO_IR - 0.05
        mock_decay_power.return_value = (expected_v, expected_ir)

        self.subject.config["post_fire"] = False
        current_hours, decay_hours = self.subject.decay_window(DECAY_TIME)

        res_v, res_ir = self.subject.date_method(
            ALBEDO_VIS, ALBEDO_IR, current_hours, decay_hours, STORM_DAYS
        )

        # Burned_no_snowfall mask should be all zeros (last argument)
        burn_mask = mock_decay_power.call_args[0][-1]
        npt.assert_array_equal(burn_mask, np.zeros_like(STORM_DAYS))

        npt.assert_array_equal(res_v, expected_v)
        npt.assert_array_equal(res_ir, expected_ir)

    @patch("smrf.envphys.albedo.decay_burned")
    @patch("smrf.envphys.albedo.decay_alb_power")
    def test_date_method_with_post_fire(self, mock_decay_power, mock_decay_burned):
        self.subject.config["post_fire"] = True
        self.subject.config["post_fire_k_burned"] = 1.2
        burn_mask = np.array([[1.0, 1.0], [0.0, 0.0]])
        self.subject.burn_mask = burn_mask

        power_v = ALBEDO_VIS - 0.02
        power_ir = ALBEDO_IR - 0.02
        mock_decay_power.return_value = (power_v, power_ir)

        final_v = power_v - 0.1
        final_ir = power_ir - 0.1
        mock_decay_burned.return_value = (final_v, final_ir)

        current_hours, decay_hours = self.subject.decay_window(DECAY_TIME)

        res_v, res_ir = self.subject.date_method(
            ALBEDO_VIS, ALBEDO_IR, current_hours, decay_hours, STORM_DAYS
        )

        expected_mask = np.array([[False, True], [False, False]])
        received_mask = mock_decay_power.call_args[0][-1]
        npt.assert_array_equal(received_mask, expected_mask)

        mock_decay_burned.assert_called_once_with(
            power_v,
            power_ir,
            STORM_DAYS,
            self.subject.burn_mask,
            self.subject.config["post_fire_k_burned"],
        )

        npt.assert_array_equal(res_v, final_v)
