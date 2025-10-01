from smrf.framework.model_framework import run_smrf
from smrf.tests.check_mixin import CheckSMRFOutputs
from smrf.tests.smrf_test_case import SMRFTestCase


class TestThreadedRME(CheckSMRFOutputs, SMRFTestCase):
    """
    Integration test for SMRF.
    Runs the short simulation over reynolds mountain east.
    """
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.gold_dir = cls.basin_dir.joinpath('gold')

        run_smrf(cls.run_config)
