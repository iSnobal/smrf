import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytz

from smrf.distribute.variable_base import VariableBase
from smrf.tests.smrf_config import SMRFConfig

TOPO = MagicMock(
    name="Topo NC",
    dem=np.array([[1, 2], [3, 4]]),
    mask=np.array([[0, 0], [0, 0]]),
    sky_view_factor=np.array([[1, 1], [1, 1]]),
    veg_height=np.array([[2, 2], [2, 2]]),
    veg_k=np.array([[3, 3], [3, 3]]),
    veg_tau=np.array([[4, 4], [4, 4]]),
    veg_type=np.array([[5, 5], [5, 5]]),
    X=np.array([1, 3]),
    Y=np.array([2, 4]),
)
STATIONS = ["station 1", "station 2"]
METADATA = pd.DataFrame(
    {
        "utm_x": [1, 3],
        "utm_y": [2, 4],
        "elevation": [100, 200],
    },
    index=STATIONS,
)


class TestVariable(VariableBase):
    DISTRIBUTION_KEY = "test_variable"
    OUTPUT_VARIABLES = {
        DISTRIBUTION_KEY: {
            "units": "Unit",
            "standard_name": "CF Name",
            "long_name": "Long Name",
            "description": "description",
        },
    }


class TestVariableBase(SMRFConfig, unittest.TestCase):
    CONFIG = {
        "test_variable": {
            "stations": ["station 2"],
            "min": 0,
            "max": 100,
            "distribution": "grid",
            "detrend": True,
            "detrend_slope": 1,
        },
        "system": {
            "threads": 4,
        },
        "time": {
            "start_date": "2025-10-01",
            "time_zone": "utc",
        },
    }

    def setUp(self):
        self.logger_patch = patch("smrf.distribute.variable_base.logging")
        self.logger_patch.start()
        self.grid_patch = patch("smrf.distribute.variable_base.Grid", CONFIG_KEY="grid")
        self.grid = self.grid_patch.start()

        self.subject = TestVariable(config=self.CONFIG, topo=TOPO)

    def tearDown(self):
        self.grid_patch.stop()
        self.logger_patch.stop()

    def test_init(self):
        self.assertEqual(self.subject.config, self.CONFIG["test_variable"])
        self.assertIsNone(self.subject.test_variable)

        self.assertListEqual(self.subject.stations, ["station 2"])
        self.assertEqual(self.subject.min, 0)
        self.assertEqual(self.subject.max, 100)
        self.assertTrue(self.subject.gridded)

        self.assertIsNone(self.subject.source_files)

        self.assertEqual(self.CONFIG["system"]["threads"], self.subject.threads)
        self.assertEqual(pytz.timezone(self.CONFIG["time"]["time_zone"]), self.subject.time_zone)

    def test_init_default_threads(self):
        config = self._copy_config(self.CONFIG)
        del config["system"]["threads"]

        subject = TestVariable(config=config, topo=TOPO)

        self.assertEqual(1, subject.threads)

    def test_init_no_config_section(self):
        base_class = VariableBase()

        self.assertIsNone(base_class.config)
        self.assertIsNone(base_class.topo)
        self.assertIsNone(base_class.metadata)

    def test_output_variable_options(self):
        npt.assert_equal(
            set(TestVariable.OUTPUT_VARIABLES.keys()), self.subject.OUTPUT_OPTIONS
        )

    def test_module_name(self):
        # Note this is the name of this file.
        self.assertEqual("test_variable_base", str(self.subject))

    def test_loaded_data(self):
        self.assertEqual(["test_variable"], self.subject.LOADED_DATA)

    def test_is_requested(self):
        self.assertTrue(self.subject.is_requested(set(["test_variable"])))
        self.assertFalse(self.subject.is_requested(set(["not_a_variable"])))

    def test_initialize(self):
        self.subject.initialize(METADATA)
        station_subset = METADATA.loc[["station 2"]]

        (args, kwargs) = self.grid.call_args

        self.assertEqual(self.CONFIG["test_variable"], args[0])
        npt.assert_equal(station_subset.utm_x.values, args[1])
        npt.assert_equal(station_subset.utm_y.values, args[2])
        npt.assert_equal(TOPO.X, args[3])
        npt.assert_equal(TOPO.Y, args[4])

        npt.assert_equal(station_subset.elevation.values, kwargs["mz"])
        npt.assert_equal(TOPO.dem, kwargs["grid_z"])
        npt.assert_equal(TOPO.mask, kwargs["mask"])
        pdt.assert_frame_equal(station_subset, kwargs["metadata"])

    @patch("smrf.distribute.variable_base.VariableBase._initialize")
    @patch("smrf.distribute.variable_base.VariableBase._open_source_files")
    def test_initialzie_with_source_files(self, mock_open_source_files, _mock_initialzie):
        external_source = "/path/to/forcing/files"

        config = self._copy_config(self.CONFIG)
        config["test_variable"]["source_files"] = external_source
        del config["test_variable"]["stations"]

        subject = TestVariable(config=config, topo=TOPO)
        subject.initialize(pd.DataFrame())

        mock_open_source_files.assert_called_once_with(external_source)

    @patch("smrf.distribute.variable_base.ReadNetCDF")
    def test_open_source_files(self, mock_read_netcdf):
        external_source = "/path/to/forcing/files"

        subject = TestVariable(config=self.CONFIG, topo=TOPO)
        subject._open_source_files(external_source)

        mock_read_netcdf.assert_called_once_with(
            Path(external_source) / "20251001" / "test_variable.nc",
            pytz.timezone(self.CONFIG["time"]["time_zone"]),
        )

    def test_initialize_no_stations(self):
        config = self._copy_config(self.CONFIG)
        del config["test_variable"]["stations"]

        self.subject = TestVariable(config=config, topo=TOPO)
        self.subject.initialize(METADATA)

        npt.assert_equal(None, self.subject.stations)

    def test_topo_accessors(self):
        self.assertEqual(self.subject.topo, TOPO)
        npt.assert_equal(self.subject.dem, TOPO.dem)
        npt.assert_equal(self.subject.sky_view_factor, TOPO.sky_view_factor)
        npt.assert_equal(self.subject.veg_height, TOPO.veg_height)
        npt.assert_equal(self.subject.veg_tau, TOPO.veg_tau)
        npt.assert_equal(self.subject.veg_k, TOPO.veg_k)
        npt.assert_equal(self.subject.veg_type, TOPO.veg_type)
