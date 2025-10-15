from smrf.framework.model_framework import run_smrf
from smrf.tests.smrf_output_test import CheckSMRFOutputs
from smrf.tests.smrf_test_case import SMRFTestCase


class TestRME(CheckSMRFOutputs, SMRFTestCase):
    """
    Integration test using test data from basins -> RME
    """
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.gold_dir = cls.basin_dir.joinpath('gold')

        run_smrf(cls.run_config)
