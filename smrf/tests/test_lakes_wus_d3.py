import os

from smrf.framework.model_framework import run_smrf
from smrf.tests.smrf_output_test import CheckSMRFOutputs
from smrf.tests.smrf_test_case_lakes import SMRFTestCaseLakes


class TestLakesWUS_D3(CheckSMRFOutputs, SMRFTestCaseLakes):
    """
    Integration test using the WUS-D3 hourly output in basins -> Lakes -> WUS-D3
    """

    config_file = os.path.join(SMRFTestCaseLakes.basin_dir, "config_wus_d3.ini")
    gold_dir = SMRFTestCaseLakes.basin_dir.joinpath("gold_wus_d3")

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        cls.smrf = run_smrf(cls.base_config)
