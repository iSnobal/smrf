from smrf.framework.model_framework import run_smrf
from smrf.tests.check_mixin import CheckSMRFOutputs
from smrf.tests.smrf_test_case_lakes import SMRFTestCaseLakes


class TestLakes(CheckSMRFOutputs, SMRFTestCaseLakes):
    """
    Integration test for SMRF
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.smrf = run_smrf(cls.run_config)
