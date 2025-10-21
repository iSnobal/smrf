import netCDF4 as nc
import numpy as np
import numpy.testing as npt

from smrf import __version__
from smrf.framework.model_framework import SMRF
from smrf.distribute import AirTemperature, CloudFactor
from smrf.output.output_netcdf import OutputNetcdf
from smrf.tests.smrf_test_case import SMRFTestCase


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

    def test_property_output_variables(self):
        self.assertEqual(self.variable_dict, self.writer.output_variables)

    def test_property_out_config(self):
        self.assertEqual(self.smrf.config["output"], self.writer.out_config)

    def test_property_topo_x_y(self):
        npt.assert_equal(self.smrf.topo.x, self.writer.topo_x)
        npt.assert_equal(self.smrf.topo.y, self.writer.topo_y)

    def test_netcdf_smrf_version(self):
        with nc.Dataset(self.writer.file_name("air_temp")) as out_file:
            self.assertEqual(__version__, out_file.getncattr("SMRF_version"))
            self.assertTrue(out_file.variables["air_temp"].dtype == np.float32)

    def test_netcdf_precision(self):
        self.writer.out_config["netcdf_output_precision"] = "double"

        writer = OutputNetcdf(
            {"cloud_factor": CloudFactor({})},
            self.smrf.topo,
            self.smrf.config["time"],
            self.smrf.config["output"],
        )

        with nc.Dataset(writer.file_name("cloud_factor")) as out_file:
            self.assertTrue(out_file.variables["cloud_factor"].dtype == np.float64)
