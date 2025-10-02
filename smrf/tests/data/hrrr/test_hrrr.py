import netCDF4 as nc
import numpy.testing as npt
from inicheck.tools import cast_all_variables
from smrf.framework.model_framework import run_smrf
from smrf.tests.smrf_test_case_lakes import SMRFTestCase, SMRFTestCaseLakes


class TestLoadHRRR(SMRFTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        config = cls.base_config_copy()
        del config.raw_cfg['csv']

        adj_config = {
            'gridded': {
                'data_type': 'hrrr_grib',
                'hrrr_directory': './gridded/hrrr_test/',
            },
            'time': {
                'start_date': '2018-07-22 16:00',
                'end_date': '2018-07-22 20:00',
                'time_zone': 'utc'
            },
            'system': {
                'log_file': './output/test.log'
            },
            'air_temp': {
                'grid_local': True,
                'grid_local_n': 25
            },
            'vapor_pressure': {
                'grid_local': True,
                'grid_local_n': 25
            },
            'precip': {
                'grid_local': True,
                'grid_local_n': 25,
                'precip_temp_method': 'dew_point'
            },
            'wind': {
                'wind_model': 'interp'
            },
            'thermal': {
                'correct_cloud': True,
                'correct_veg': True
            },
            'albedo': {
                'grain_size': 300.0,
                'max_grain': 2000.0
            }
        }
        config.raw_cfg.update(adj_config)

        # set the distribution to grid, thermal defaults will be fine
        for v in cls.DIST_VARIABLES:
            config.raw_cfg[v]['distribution'] = 'grid'
            config.raw_cfg[v]['grid_mask'] = 'False'

        config.apply_recipes()
        config = cast_all_variables(config, config.mcfg)

        cls.config = config
        cls.gold_dir = cls.basin_dir.joinpath('gold_hrrr')

    def test_load_timestep(self):
        run_smrf(self.config)

        self.compare_hrrr_gold()


class TestHrrrThermal(SMRFTestCaseLakes):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        config = cls.base_config_copy()
        config.raw_cfg['output']['variables'] = ['hrrr_thermal']
        config.apply_recipes()
        config = cast_all_variables(config, config.mcfg)

        cls.config = config

    def test_load(self):
        run_smrf(self.config)

        nc_variable = 'thermal'

        with nc.Dataset(self.gold_dir.joinpath('thermal_hrrr.nc')) as gold:
            with nc.Dataset(self.output_dir.joinpath('thermal.nc')) as test:
                npt.assert_equal(
                    gold.variables['time'][:],
                    test.variables['time'][:],
                    err_msg="Time steps did not match for {}".format(nc_variable)
                )

                for att in gold.variables[nc_variable].ncattrs():
                    self.assertEqual(
                        getattr(gold.variables[nc_variable], att),
                        getattr(test.variables[nc_variable], att)
                    )

                npt.assert_array_equal(
                    test.variables[nc_variable][:],
                    gold.variables[nc_variable][:],
                    "Variable: {0} did not match gold standard".format(nc_variable)
                )
