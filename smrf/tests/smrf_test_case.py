import os
import shutil
import unittest
from pathlib import Path

import netCDF4 as nc
import numpy.testing as npt
from inicheck.config import UserConfig

import smrf
from inicheck.tools import get_user_config


class SMRFTestCase(unittest.TestCase):
    """
    The base test case for SMRF that will load in the configuration file
    and store as the base config. Also will remove the output
    directory upon tear down.

    Runs the short simulation over Reynolds Mountain East (RME)
    """
    DISTRIBUTION_VARIABLES = frozenset([
        'air_temp',
        'cloud_factor',
        'precip',
        'thermal',
        'vapor_pressure',
        'wind',
    ])

    BASE_INI_FILE_NAME = 'config.ini'

    test_dir = Path(smrf.__file__).parent.joinpath('tests')
    basin_dir = test_dir.joinpath('basins', 'RME')
    config_file = os.path.join(basin_dir, BASE_INI_FILE_NAME)
    gold_dir = basin_dir.joinpath("gold_hrrr")

    @classmethod
    def topo_nc(cls):
        return cls.basin_dir.joinpath("topo", "topo.nc")

    @classmethod
    def base_config_copy(cls) -> UserConfig:
        """
        Return a copy of the default test config to manipulate for specific test
        cases

        :return: UserConfig object
        """
        return get_user_config(cls.config_file, modules='smrf')

    @classmethod
    def setUpClass(cls):
        """
        Set up basic structure to run SMRF by parsing the `.ini` file, create
        output folder logic, and create a copy of the unparsed `.ini` file for. The latter
        can be used to configure a run differently, without having to redefine a whole
        file.
        """
        cls.base_config = get_user_config(cls.config_file, modules='smrf')
        cls.create_output_dir()

    @classmethod
    def tearDownClass(cls):
        cls.remove_output_dir()
        delattr(cls, 'output_dir')

    # START - Required test setup folder structure methods

    @classmethod
    def create_output_dir(cls):
        folder = os.path.join(cls.base_config.cfg['output']['out_location'])

        # Remove any potential files to ensure fresh run
        if os.path.isdir(folder):
            shutil.rmtree(folder)

        os.makedirs(folder)
        cls.output_dir = Path(folder)

    @classmethod
    def remove_output_dir(cls):
        if hasattr(cls, 'output_dir') and \
                os.path.exists(cls.output_dir):
            shutil.rmtree(cls.output_dir, ignore_errors=True)

    # END

    def compare_hrrr_gold(self):
        """
        Compare the model results with the gold standard
        """
        [
            self.compare_netcdf_files(file_name.name)
            for file_name in self.gold_dir.glob('*.nc')
        ]

    def compare_netcdf_files(self, output_file):
        """
        Compare two netcdf files to ensure that they are identical. The
        tests will compare the attributes of each variable and ensure that
        the values are exact
        """

        with nc.Dataset(self.gold_dir.joinpath(output_file)) as gold:
            with nc.Dataset(self.output_dir.joinpath(output_file)) as test:
                # See AWSM issue #11
                self.compare_file_variables(gold, test, 0.005)

    def compare_file_variables(self, gold, test, tolerance=1e-10):
        npt.assert_equal(
            gold.variables['time'][:],
            test.variables['time'][:],
            err_msg="Time steps did not match gold standard"
        )

        for variable, _v in gold.variables.items():
            # Time was already compared
            if variable =='time':
                continue

            for attribute in gold.variables[variable].ncattrs():
                self.assertEqual(
                    getattr(gold.variables[variable], attribute),
                    getattr(test.variables[variable], attribute),
                    msg="Variable `{0}` attribute `{1}` did not match gold standard".format(
                        variable, attribute
                    ),
                )

            if variable in ['x', 'y']:
                npt.assert_equal(
                    gold.variables[variable][:],
                    test.variables[variable][:],
                    err_msg=f"Coordinate {variable} did not match gold standard",
                )
            elif variable == 'projection':
                self.assertEqual(
                    gold.variables[variable].spatial_ref,
                    test.variables[variable].spatial_ref,
                    msg="Spatial reference did not match gold standard"
                )
            else:
                for time_slice in range(len(gold.variables[variable])):
                    npt.assert_allclose(
                        gold.variables[variable][time_slice][time_slice, ...],
                        test.variables[variable][time_slice][time_slice, ...],
                        rtol=tolerance,
                        err_msg=f"Variable: {variable} at time slice {time_slice} did not match gold standard",
                    )
