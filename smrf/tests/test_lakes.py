from smrf.framework.model_framework import run_smrf
from smrf.tests.smrf_output_test import CheckSMRFOutputs
from smrf.tests.smrf_test_case_lakes import SMRFTestCaseLakes


class TestLakes(CheckSMRFOutputs, SMRFTestCaseLakes):
    """
    Integration test using test data from basins -> Lakes
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.smrf = run_smrf(cls.base_config)
