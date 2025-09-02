from inicheck.tools import cast_all_variables

from smrf.framework.model_framework import run_smrf
from smrf.tests.smrf_test_case_lakes import SMRFTestCaseLakes
from smrf.tests.check_mixin import CheckSMRFOutputs


class TestLakes(CheckSMRFOutputs, SMRFTestCaseLakes):
    """
    Integration test for SMRF without threading.
        - serial simulation
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.smrf = run_smrf(cls.run_config)
