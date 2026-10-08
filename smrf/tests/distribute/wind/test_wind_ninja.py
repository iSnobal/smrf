import numpy as np
import pandas as pd
import pytz

from smrf.data import Topo
from smrf.distribute.wind.wind import Wind
from smrf.tests.smrf_test_case_lakes import SMRFTestCaseLakes
from smrf.utils.utils import date_range


class TestWindNinja(SMRFTestCaseLakes):
    def setup_wind_ninja(self, config):
        """Setup the wind ninja class

        Args:
            config (dict): config dict
        """
        topo = Topo(config["topo"])

        # Get the time steps correctly in the time zone
        tzinfo = pytz.timezone(config["time"]["time_zone"])
        date_time = date_range(
            config["time"]["start_date"],
            config["time"]["end_date"],
            config["time"]["time_step"],
            tzinfo,
        )

        wind = Wind(config=config, topo=topo)
        wind.initialize(pd.DataFrame())
        wn = wind.wind_model
        wn.initialize_interp(date_time[0])
        g_vel, g_ang = wn.convert_wind_ninja(date_time[0])

        return topo, wn, g_vel, g_ang

    def test_wind_ninja(self):

        config = self.base_config.cfg
        _, wn, _, _ = self.setup_wind_ninja(config)

        # The x values are ascending
        self.assertTrue(np.all(np.diff(wn.windninja_x) > 0))

        # The y values are descending
        self.assertTrue(np.all(np.diff(wn.windninja_y) < 0))

    def test_wind_ninja_interpolation(self):

        config = self.base_config_copy().cfg
        config["wind"]["wind_ninja_dxdy"] = 50
        topo, wn, g_vel, g_ang = self.setup_wind_ninja(config)

        # The implied assumption is that this does not throw an
        # exception when running
        self.assertTrue(np.all(np.diff(wn.windninja_x) > 0))
        self.assertTrue(np.all(np.diff(wn.windninja_y) < 0))
        self.assertTrue(np.sum(np.isnan(g_vel)) == 0)
        self.assertTrue(np.sum(np.isnan(g_ang)) == 0)
        self.assertEqual(g_vel.shape, topo.dem.shape)
        self.assertEqual(g_ang.shape, topo.dem.shape)

    def test_fill_data_edges_only(self):
        config = self.base_config_copy().cfg
        _, wn, _, _ = self.setup_wind_ninja(config)

        shape = wn.X.shape
        grid = np.ones(shape)
        grid[0, :] = np.nan
        grid[-1, :] = np.nan
        grid[:, 0] = np.nan
        grid[:, -1] = np.nan

        filled = wn.fill_data(grid)
        self.assertFalse(np.any(np.isnan(filled)))
        np.testing.assert_allclose(filled, 1.0)

        # A NaN in the middle of the grid is not an edge value
        grid = np.ones(shape)
        grid[shape[0] // 2, shape[1] // 2] = np.nan
        with self.assertRaises(ValueError):
            wn.fill_data(grid)

    def test_fill_nan_leaves_interior(self):
        x = np.arange(6, dtype=float)
        data = np.array([np.nan, 1, np.nan, 3, 4, np.nan])
        result = type(
            self.setup_wind_ninja(self.base_config_copy().cfg)[1]
        ).fill_nan(data, x)

        self.assertTrue(np.isnan(result[2]))
        np.testing.assert_allclose(result[[0, 5]], [0.0, 5.0])
