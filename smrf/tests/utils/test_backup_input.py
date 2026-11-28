import os

from smrf.framework.model_framework import SMRF
from smrf.tests.smrf_test_case import SMRFTestCase
from smrf.utils.utils import backup_input
from smrf.version import __version__ as SMRF_VERSION


class TestBackupInput(SMRFTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.smrf = SMRF(cls.config_file)
        cls.smrf.load_topo()
        cls.smrf.create_distribution()
        cls.smrf.load_data()

    def test_backup_version(self):
        backup_input(self.smrf.data, self.smrf.ucfg)

        backup_config = os.path.join(
            self.smrf.config["output"]["out_location"],
            "input_backup",
            "backup_config.ini",
        )

        with open(backup_config) as f:
            lines = f.readlines()
            self.assertTrue(SMRF_VERSION in lines[1])
