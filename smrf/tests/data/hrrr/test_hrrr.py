import netCDF4 as nc

from inicheck.tools import cast_all_variables
from smrf.framework.model_framework import run_smrf
from smrf.tests.smrf_test_case_lakes import SMRFTestCase, SMRFTestCaseLakes


class TestLoadHRRR(SMRFTestCase):
    def setUp(self):
        config = self.base_config_copy()
        del config.raw_cfg["csv"]

        adj_config = {
            "gridded": {
                "data_type": "hrrr_grib",
                "hrrr_directory": "./gridded/hrrr_test/",
            },
            "time": {
                "start_date": "2018-07-22 16:00",
                "end_date": "2018-07-22 20:00",
                "time_zone": "utc",
            },
            "system": {"log_file": "./output/test.log"},
            "air_temp": {"grid_local": True, "grid_local_n": 25},
            "vapor_pressure": {"grid_local": True, "grid_local_n": 25},
            "precip": {
                "grid_local": True,
                "grid_local_n": 25,
                "precip_temp_method": "dew_point",
            },
            "wind": {"wind_model": "interp"},
            "thermal": {"correct_cloud": True, "correct_veg": True},
            "albedo": {"grain_size": 300.0, "max_grain": 2000.0},
        }
        config.raw_cfg.update(adj_config)

        # set the distribution to grid, thermal defaults will be fine
        for v in self.DISTRIBUTION_VARIABLES:
            config.raw_cfg[v]["distribution"] = "grid"
            config.raw_cfg[v]["grid_mask"] = "False"

        config.apply_recipes()
        config = cast_all_variables(config, config.mcfg)

        self.config = config

    def test_load_timestep(self):
        run_smrf(self.config)

        self.compare_hrrr_gold()


class TestHrrrThermal(SMRFTestCaseLakes):
    def setUp(self):
        config = self.base_config_copy()
        config.cfg["output"]["variables"] = ["hrrr_thermal"]
        config = cast_all_variables(config, config.mcfg)

        self.config = config

    def test_load(self):
        run_smrf(self.config)

        with nc.Dataset(self.gold_dir.joinpath("thermal_hrrr.nc")) as gold:
            with nc.Dataset(self.output_dir.joinpath("thermal.nc")) as test:
                self.compare_file_variables(gold, test)


class TestHrrrSolar(SMRFTestCaseLakes):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        config = cls.base_config_copy()
        config.cfg["output"]["variables"] = ["hrrr_solar", "solar_k"]
        config = cast_all_variables(config, config.mcfg)

        cls.config = config

    def test_load(self):
        run_smrf(self.config)

        with nc.Dataset(self.gold_dir.joinpath("net_solar_hrrr.nc")) as gold:
            with nc.Dataset(self.output_dir.joinpath("net_solar.nc")) as test:
                self.compare_file_variables(gold, test)

        self.compare_netcdf_files(self.gold_dir.joinpath("hrrr_solar.nc"))
        self.compare_netcdf_files(self.gold_dir.joinpath("solar_k.nc"))
