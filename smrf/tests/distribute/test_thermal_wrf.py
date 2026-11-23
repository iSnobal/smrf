import unittest
from unittest.mock import patch

import numpy as np
import numpy.testing as npt
import pandas as pd

from smrf.distribute import ThermalWRF
from smrf.envphys.constants import EMISS_TERRAIN, FREEZE, STEF_BOLTZ
from smrf.tests.distribute import topo_mock

GRID_SKY_VIEW = np.full((2, 2), 0.5)
GRID_METHOD = "linear"
DETREND_SLOPE = 0
DATE = "2025-09-20"

INTERPOLATED = np.array([[300.0, 310.0], [320.0, 330.0]])
AIR_TEMP_MOCK = np.array([[1.0, 2.0], [3.0, 4.0]])

LONGWAVE = pd.Series([300.0, 310.0, 320.0], index=["grid_1", "grid_2", "grid_3"])

METADATA = pd.DataFrame(
    {
        "utm_x": [1.0, 3.0, 5.0],
        "utm_y": [2.0, 4.0, 6.0],
        "elevation": [100.0, 200.0, 300.0],
    },
    index=["grid_1", "grid_2", "grid_3"],
)

CONFIG = {
    "time": {
        "start_date": "2025-09-20 00:00",
        "time_zone": "utc",
    },
    "thermal": {
        "distribution": "grid",
        "detrend": True,
        "detrend_slope": DETREND_SLOPE,
        "grid_method": GRID_METHOD,
        "correct_veg": True,
    },
}


class TestThermalWRF(unittest.TestCase):
    def setUp(self):
        # We don't want to test the actually interpolation. Patch here
        self.grid_patch = patch("smrf.distribute.variable_base.Grid", CONFIG_KEY="grid")
        self.grid = self.grid_patch.start()
        self.addCleanup(self.grid_patch.stop)

        self.grid.return_value.detrended_interpolation.return_value = INTERPOLATED

        self.topo = topo_mock(GRID_SKY_VIEW)
        self.topo.X = np.array([1.0, 3.0])
        self.topo.Y = np.array([2.0, 4.0])
        self.topo.mask = np.zeros((2, 2))

        self.subject = ThermalWRF(config=CONFIG, topo=self.topo)

    def test_initialize(self):
        self.subject.initialize(METADATA)

        self.grid.assert_called_once()
        config, mx, my, grid_x, grid_y = self.grid.call_args[0]
        self.assertIs(config, self.subject.config)
        npt.assert_equal(mx, METADATA["utm_x"].to_numpy())
        npt.assert_equal(my, METADATA["utm_y"].to_numpy())
        npt.assert_equal(grid_x, self.topo.X)
        npt.assert_equal(grid_y, self.topo.Y)

        kwargs = self.grid.call_args[1]
        npt.assert_equal(kwargs["mz"], METADATA["elevation"].to_numpy())
        npt.assert_equal(kwargs["grid_z"], self.topo.dem)
        npt.assert_equal(kwargs["mask"], self.topo.mask)
        self.assertIs(kwargs["metadata"], METADATA)

    @patch("smrf.distribute.thermal_hrrr.vegetation")
    def test_distribute(self, mock_vegetation):
        self.subject.initialize(METADATA)
        self.subject.distribute(DATE, LONGWAVE, AIR_TEMP_MOCK)

        self.grid.return_value.detrended_interpolation.assert_called_once()
        data, detrend_slope, grid_method = (
            self.grid.return_value.detrended_interpolation.call_args[0]
        )
        npt.assert_equal(data.to_numpy(), LONGWAVE.to_numpy())
        self.assertEqual(detrend_slope, DETREND_SLOPE)
        self.assertEqual(grid_method, GRID_METHOD)

        clear_sky = EMISS_TERRAIN * STEF_BOLTZ * (AIR_TEMP_MOCK + FREEZE) ** 4
        sky_corrected = (GRID_SKY_VIEW * INTERPOLATED) + (1 - GRID_SKY_VIEW) * clear_sky

        veg_thermal, veg_air_temp, veg_tau, veg_height = (
            mock_vegetation.thermal_correct_canopy.call_args[0]
        )
        npt.assert_allclose(veg_thermal, sky_corrected)
        npt.assert_equal(veg_air_temp, AIR_TEMP_MOCK)
        npt.assert_equal(veg_tau, self.topo.veg_tau)
        npt.assert_equal(veg_height, self.topo.veg_height)

        self.assertIs(
            self.subject.thermal,
            mock_vegetation.thermal_correct_canopy.return_value,
        )
