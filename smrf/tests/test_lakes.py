from smrf.framework.model_framework import run_smrf
from smrf.tests.smrf_output_test import CheckSMRFOutputs
from smrf.tests.smrf_test_case_lakes import SMRFTestCaseLakes
from inicheck.tools import cast_all_variables
import netCDF4 as nc


class TestLakes(CheckSMRFOutputs, SMRFTestCaseLakes):
    """
    Integration test using test data from basins -> Lakes
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.smrf = run_smrf(cls.base_config)

class TestLakesExternalAlbedo(SMRFTestCaseLakes):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        config = cls.base_config_copy()
        config.cfg["output"]["variables"] = ["hrrr_solar"]
        config.cfg["albedo"]["source_files"] = str(cls.input_dir / "forcing")
        config.cfg["solar"]["correct_veg"] = False
        config = cast_all_variables(config, config.mcfg)

        cls.config = config

    def test_run(self):
        run_smrf(self.config)

        with nc.Dataset(self.gold_dir.joinpath("net_solar_hrrr_albedo.nc")) as gold:
            with nc.Dataset(self.output_dir.joinpath("net_solar.nc")) as test:
                self.compare_file_variables(gold, test)

        self.compare_netcdf_files(self.gold_dir.joinpath("hrrr_solar.nc"))
