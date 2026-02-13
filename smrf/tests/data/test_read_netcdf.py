import unittest
from datetime import datetime
from unittest import mock
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pytz

from smrf.data.read_netcdf import ReadNetCDF


class TestReadNetCDF(unittest.TestCase):
    def setUp(self):
        self.test_file = Path("test_data.nc")
        self.time_zone = pytz.timezone("UTC")

        self.logger_patcher = mock.patch("smrf.data.read_netcdf.logging.getLogger")
        self.mock_getLogger = self.logger_patcher.start()
        self.mock_logger = mock.MagicMock()
        self.mock_getLogger.return_value = self.mock_logger

        self.atexit_patcher = mock.patch("smrf.data.read_netcdf.atexit.register")
        self.mock_atexit = self.atexit_patcher.start()

        # First create a real in memory NetCDF file, before patching the class
        self.mock_file = self._create_mock_netcdf_file()
        self.dataset_patcher = mock.patch("smrf.data.read_netcdf.netCDF4.Dataset")
        self.mock_dataset = self.dataset_patcher.start()
        self.mock_dataset.return_value = self.mock_file

    def tearDown(self):
        self.logger_patcher.stop()
        self.atexit_patcher.stop()
        self.dataset_patcher.stop()

        if self.mock_file.isopen():
            self.mock_file.close()

    def test_init(self):
        reader = ReadNetCDF(self.test_file, self.time_zone)

        self.mock_dataset.assert_called_once_with(self.test_file, "r")

        self.mock_atexit.assert_called_once_with(reader.close)

        # Verify attributes were set correctly
        self.assertEqual(reader.time_zone, self.time_zone)
        self.assertIsNotNone(reader.dates)
        self.assertEqual(len(reader.dates), 3)
        self.assertEqual(len(reader.variables), 2)

    def test_load_timesteps(self):
        reader = ReadNetCDF(self.test_file, self.time_zone)

        self.assertEqual(len(reader.dates), 3)
        self.assertIsInstance(reader.dates[0], float)
        self.assertIsInstance(reader.dates[1], float)
        self.assertIsInstance(reader.dates[2], float)
        self.assertTrue(reader.dates[0] < reader.dates[1] < reader.dates[2])

    def test_load(self):
        reader = ReadNetCDF(self.test_file, self.time_zone)

        timestep = datetime.fromtimestamp(reader.dates[0], tz=self.time_zone)
        result = reader.load("temperature", timestep)

        self.assertEqual(result, 10.5)

    def test_close_file(self):
        reader = ReadNetCDF(self.test_file, self.time_zone)
        reader.close()

        self.assertFalse(self.mock_file.isopen())

    @staticmethod
    def _create_mock_netcdf_file():
        """
        Helper method to create a mock NetCDF file in memory with time and
        optional variable data.

        Returns:
            NetCDF4.Dataset: Mocked NetCDF file
        """
        data = [10.5, 11.2, 12.1]
        variable = {"temperature": data}

        ds = nc.Dataset("test_data.nc", mode="w", diskless=True, persist=False)

        time_values = np.array([0.0, 1.0, 2.0], dtype="f4")
        ds.createDimension("time", len(time_values))
        time_var = ds.createVariable("time", "f4", ("time",))
        time_var[:] = time_values
        time_var.units = "hours since 2024-01-01 00:00:00"
        time_var.calendar = "standard"

        # Custom variables
        for var_name, data in variable.items():
            ds.createDimension(var_name, len(data))
            var = ds.createVariable(var_name, "f4", (var_name,))
            var[:] = np.array(data, dtype="f4")

        return ds
