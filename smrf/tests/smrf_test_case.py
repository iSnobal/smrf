import os
import shutil
import unittest
from pathlib import Path

import netCDF4 as nc
import numpy as np
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
            shutil.rmtree(cls.output_dir)

    # END

    @staticmethod
    def assert_gold_equal(gold, not_gold, error_msg):
        """Compare two arrays

        Arguments:
            gold {array} -- gold array
            not_gold {array} -- not gold array
            error_msg {str} -- error message to display
        """

        if os.getenv('NOT_ON_GOLD_HOST') is None:
            np.allclose(
                not_gold,
                gold,
                atol=0,
                rtol=1e-4
            )
        else:
            np.testing.assert_almost_equal(
                not_gold,
                gold,
                decimal=3,
                err_msg=error_msg
            )

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

        gold = nc.Dataset(self.gold_dir.joinpath(output_file))
        test = nc.Dataset(self.output_dir.joinpath(output_file))

        np.testing.assert_equal(
            gold.variables['time'][:],
            test.variables['time'][:],
            err_msg="Time steps did not match: \nGOLD {0} \n TEST {1}".format(
                gold.variables['time'], test.variables['time']
            )
        )

        # go through all variables and compare everything including
        # the attributes and data
        for var_name, v in gold.variables.items():

            # compare the dimensions
            for att in gold.variables[var_name].ncattrs():
                self.assertEqual(
                    getattr(gold.variables[var_name], att),
                    getattr(test.variables[var_name], att),
                    msg="Variable `{0}` attribute `{1}` did not match gold standard in file {2}".format(
                        var_name, att, output_file
                    ),
                )

            # only compare those that are floats
            if gold.variables[var_name].datatype != np.dtype('S1'):
                error_msg = "Variable: {0} did not match gold standard". \
                    format(var_name)
                self.assert_gold_equal(
                    gold.variables[var_name][:],
                    test.variables[var_name][:],
                    error_msg
                )

        gold.close()
        test.close()
