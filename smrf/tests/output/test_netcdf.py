import os

import netCDF4 as nc
import numpy as np
import numpy.testing as npt

from smrf.distribute import AirTemperature, CloudFactor
from smrf.framework.model_framework import SMRF
from smrf.output.output_netcdf import OutputNetcdf
from smrf.tests.smrf_test_case import SMRFTestCase
from smrf.version import __version__ as SMRF_VERSION


class TestOutputNetCDF(SMRFTestCase):
    def setUp(self):
        self.smrf = SMRF(self.base_config_copy())
        self.smrf.load_topo()
        module = AirTemperature()

        self.variable_dict = {"air_temp": module}

        self.writer = OutputNetcdf(
            self.variable_dict,
            self.smrf.topo,
            self.smrf.config["time"],
            self.smrf.config["output"],
        )
        # Start each test with a new file, since the output folder is shared
        self.addCleanup(os.remove, self.writer.file_name("air_temp"))

    def read_air_temp(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Times and values in the air_temp output file.
        """
        with nc.Dataset(self.writer.file_name("air_temp")) as out_file:
            return out_file.variables["time"][:], out_file.variables["air_temp"][:]

    def test_property_output_variables(self):
        self.assertEqual(self.variable_dict, self.writer.output_variables)

    def test_property_out_config(self):
        self.assertEqual(self.smrf.config["output"], self.writer.out_config)

    def test_property_topo_x_y(self):
        npt.assert_equal(self.smrf.topo.x, self.writer.topo_x)
        npt.assert_equal(self.smrf.topo.y, self.writer.topo_y)

    def test_netcdf_smrf_version(self):
        with nc.Dataset(self.writer.file_name("air_temp")) as out_file:
            self.assertEqual(SMRF_VERSION, out_file.getncattr("SMRF_version"))
            self.assertTrue(out_file.variables["air_temp"].dtype == np.float32)

    def test_netcdf_precision(self):
        self.writer.out_config["netcdf_output_precision"] = "double"

        writer = OutputNetcdf(
            {
                "cloud_factor": CloudFactor(
                    {
                        "time": {
                            "time_zone": "utc",
                            "start_date": "2025-10-01 00:00",
                        },
                    }
                )
            },
            self.smrf.topo,
            self.smrf.config["time"],
            self.smrf.config["output"],
        )

        with nc.Dataset(writer.file_name("cloud_factor")) as out_file:
            self.assertTrue(out_file.variables["cloud_factor"].dtype == np.float64)

    def test_flush(self):
        array = np.full((self.smrf.topo.ny, self.smrf.topo.nx), 5.0)
        self.variable_dict["air_temp"].air_temp = array
        self.writer.output(self.smrf.date_time[0])
        self.writer.flush()

        file_times, data = self.read_air_temp()
        npt.assert_equal(file_times, [0])
        npt.assert_allclose(data, [array], atol=self.VARIABLE_TOLERANCE)

    def test_output_none(self):
        self.writer.output(self.smrf.date_time[0])
        self.writer.flush()

        _, data = self.read_air_temp()
        npt.assert_equal(data, np.zeros((1, self.smrf.topo.ny, self.smrf.topo.nx)))

    def test_output_mask(self):
        self.writer.out_config["mask_output"] = True
        array = np.full((self.smrf.topo.ny, self.smrf.topo.nx), 5.0)
        self.variable_dict["air_temp"].air_temp = array
        self.writer.output(self.smrf.date_time[0])
        self.writer.flush()

        _, data = self.read_air_temp()
        npt.assert_allclose(
            data, [array * self.smrf.topo.mask], atol=self.VARIABLE_TOLERANCE
        )
