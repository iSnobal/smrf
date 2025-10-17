import os

import netCDF4 as nc
import numpy as np
import numpy.testing as npt

from unittest.mock import MagicMock

from smrf import __version__
from smrf.output.output_netcdf import OutputNetcdf
from smrf.tests.smrf_test_case import SMRFTestCase

class TestOutputNetCDF(SMRFTestCase):
    def setUp(self):
        super().setUpClass()
        self.smrf = self.smrf_instance
        self.smrf.load_topo()

        self.variable_dict = {
            "air_temp": {
                "variable": "air_temp",
                "module": "air_temp",
                "out_location": os.path.join(
                    self.smrf.config["output"]["out_location"], "air_temp"
                ),
                "nc_attributes": {
                    "units": "degree_Celsius",
                    "standard_name": "air_temperature",
                    "long_name": "Air temperature",
                },
            }
        }

        self.writer = OutputNetcdf(
            self.variable_dict,
            self.smrf.topo,
            self.smrf.config["time"],
            self.smrf.config["output"],
        )

    def test_property_variables_info(self):
        self.assertEqual(self.variable_dict, self.writer.variables_info)

    def test_property_out_config(self):
        self.assertEqual(self.smrf.config["output"], self.writer.out_config)

    def test_property_topo_x_y(self):
        npt.assert_equal(self.smrf.topo.x, self.writer.topo_x)
        npt.assert_equal(self.smrf.topo.y, self.writer.topo_y)

    def test_netcdf_smrf_version(self):
        with nc.Dataset(self.writer.variables_info["air_temp"]["out_location"]) as out_file:
            self.assertEqual(__version__, out_file.getncattr("SMRF_version"))
            self.assertTrue(out_file.variables["air_temp"].dtype == np.float32)

    def test_netcdf_precision(self):
        self.smrf.config["output"]["netcdf_output_precision"] = "double"

        writer = OutputNetcdf(
            self.variable_dict,
            self.smrf.topo,
            self.smrf.config["time"],
            self.smrf.config["output"],
        )

        with nc.Dataset(writer.variables_info["air_temp"]["out_location"]) as out_file:
            self.assertTrue(out_file.variables["air_temp"].dtype == np.float64)
