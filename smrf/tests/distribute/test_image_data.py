import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt

from smrf.distribute.image_data import ImageData

TOPO = MagicMock(
    name="Topo NC",
    dem=np.array([[1, 2], [3, 4]]),
    mask=np.array([[0, 0], [0, 0]]),
    sky_view_factor=np.array([[1, 1], [1, 1]]),
    veg_height=np.array([[2, 2], [2, 2]]),
    veg_k=np.array([[3, 3], [3, 3]]),
    veg_tau=np.array([[4, 4], [4, 4]]),
    X=np.array([1, 3]),
    Y=np.array([2, 4]),
)
STATIONS = ["station 1", "station 2"]
METADATA = pd.DataFrame({
    'utm_x': [1, 3],
    'utm_y': [2, 4],
    'elevation': [100, 200],
    },
    index=STATIONS,
)


class TestVariable(ImageData):
    VARIABLE = "test_variable"
    OUTPUT_VARIABLES = {
        VARIABLE: {
            "units": "Unit",
            "standard_name": "CF Name",
            "long_name": "Long Name",
            "description": "description",
        },
    }


class TestImageData(unittest.TestCase):
    CONFIG = {
        "test_variable": {
            "stations": STATIONS,
            "min": 0,
            "max": 100,
            "distribution": "grid",
            "detrend": True,
            "detrend_slope": 1,
        },
        "distribution": "grid",
    }

    def setUp(self):
        self.logger_patch = patch("smrf.distribute.image_data.logging")
        self.logger_patch.start()
        self.grid_patch = patch("smrf.distribute.image_data.grid.GRID")
        self.grid = self.grid_patch.start()

        self.subject = TestVariable(self.CONFIG)

    def tearDown(self):
        self.grid_patch.stop()
        self.logger_patch.stop()

    def test_init(self):
        self.assertEqual(self.subject.config, self.CONFIG["test_variable"])
        self.assertIsNone(self.subject.test_variable)

        self.assertListEqual(self.subject.stations, ["station 1", "station 2"])
        self.assertEqual(self.subject.min, 0)
        self.assertEqual(self.subject.max, 100)
        self.assertTrue(self.subject.gridded)

    def test_init_no_config_section(self):
        base_class = ImageData()

        self.assertIsNone(base_class.config)
        self.assertIsNone(base_class.topo)
        self.assertIsNone(base_class.metadata)

    def test_output_variable_options(self):
        npt.assert_equal(
            set(TestVariable.OUTPUT_VARIABLES.keys()), self.subject.OUTPUT_OPTIONS
        )

    def test_module_name(self):
        self.assertEqual("test_image_data", self.subject.MODULE_NAME)

    def test_loaded_data(self):
        self.assertEqual(["test_variable"], self.subject.LOADED_DATA)

    def test_is_requested(self):
        self.assertTrue(self.subject.is_requested(set(["test_variable"])))
        self.assertFalse(self.subject.is_requested(set(["not_a_variable"])))

    def test_initialize(self):
        self.subject.initialize(TOPO, METADATA)

        (args, kwargs) = self.grid.call_args

        self.assertEqual(self.CONFIG["test_variable"], args[0])
        npt.assert_equal(METADATA.utm_x.values, args[1])
        npt.assert_equal(METADATA.utm_y.values, args[2])
        npt.assert_equal(TOPO.X, args[3])
        npt.assert_equal(TOPO.Y, args[4])

        npt.assert_equal(METADATA.elevation.values, kwargs["mz"])
        npt.assert_equal(TOPO.dem, kwargs["GridZ"])
        npt.assert_equal(TOPO.mask, kwargs["mask"])
        pdt.assert_frame_equal(METADATA, kwargs["metadata"])

    def test_initialize_no_stations(self):
        config = self.CONFIG.copy()
        del config["test_variable"]["stations"]

        self.subject = TestVariable(config)
        self.subject.initialize(TOPO, METADATA)

        npt.assert_equal(STATIONS, self.subject.stations)

    def test_topo_accessors(self):
        self.subject.initialize(TOPO, METADATA)

        self.assertEqual(self.subject.topo, TOPO)
        npt.assert_equal(self.subject.dem, TOPO.dem)
        npt.assert_equal(self.subject.sky_view_factor, TOPO.sky_view_factor)
        npt.assert_equal(self.subject.veg_height, TOPO.veg_height)
        npt.assert_equal(self.subject.veg_tau, TOPO.veg_tau)
        npt.assert_equal(self.subject.veg_k, TOPO.veg_k)
