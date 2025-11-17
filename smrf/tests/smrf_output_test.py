import netCDF4
import numpy as np
import numpy.testing as npt
import netCDF4 as nc


class CheckSMRFOutputs(object):
    """
    Check the SMRF test case for all the variables. To be used as a
    mixin for tests to avoid these tests running more than once.

    Example:
        TestSomethingNew(CheckSMRFOutputs, SMRFTestCase)
    """

    def test_air_temp(self):
        self.compare_netcdf_files('air_temp.nc')

    def test_precip_temp(self):
        self.compare_netcdf_files("precip_temp.nc")

        # Check mask of temperature to match precipitation
        with nc.Dataset(self.output_dir.joinpath("precip_temp.nc")) as temperature:
            with netCDF4.Dataset(self.output_dir.joinpath("precip.nc")) as precip:
                npt.assert_equal(
                    precip.variables["precip"] == np.nan,
                    temperature.variables["precip_temp"] == np.nan,
                )

    def test_net_solar(self):
        self.compare_netcdf_files('net_solar.nc')

    def test_percent_snow(self):
        self.compare_netcdf_files('percent_snow.nc')

    def test_precip(self):
        self.compare_netcdf_files('precip.nc')

    def test_thermal(self):
        self.compare_netcdf_files('thermal.nc')

    def test_wind_speed(self):
        self.compare_netcdf_files('wind_speed.nc')

    def test_wind_direction(self):
        self.compare_netcdf_files('wind_direction.nc')

    def test_snow_density(self):
        self.compare_netcdf_files('snow_density.nc')

    def test_vapor_pressure(self):
        self.compare_netcdf_files('vapor_pressure.nc')
