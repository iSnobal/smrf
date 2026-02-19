import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.testing as npt
import pandas as pd

from smrf.distribute import ThermalHRRR
from smrf.envphys.constants import EMISS_TERRAIN, FREEZE, STEF_BOLTZ
from smrf.tests.distribute import SKY_VIEW_FACTOR_MOCK, TOPO_MOCK
from smrf.tests.smrf_config import SMRFConfig

RAW_DATA_MOCK = np.ones((1, 1))
AIR_TEMP_MOCK = np.ones((1, 1))

DATA_MOCK = MagicMock(thermal=RAW_DATA_MOCK)

CONFIG = {
    "time": {
        "start_date": "2025-09-20 00:00",
        "time_zone": "utc",
    },
    "thermal": {
        "correct_veg": False,
    },
}


class TestThermalHRRR(unittest.TestCase, SMRFConfig):
    def test_distribute(self):
        self.subject = ThermalHRRR(config=CONFIG, topo=TOPO_MOCK)

        result = (SKY_VIEW_FACTOR_MOCK * RAW_DATA_MOCK) + (
            1 - SKY_VIEW_FACTOR_MOCK
        ) * EMISS_TERRAIN * STEF_BOLTZ * (AIR_TEMP_MOCK + FREEZE) ** 4

        self.subject.initialize(pd.DataFrame())
        self.subject.distribute("2025-09-20", RAW_DATA_MOCK, AIR_TEMP_MOCK)

        npt.assert_equal(result, self.subject.thermal)

    @patch("smrf.distribute.thermal_hrrr.vegetation")
    def test_distribute_vegetation(self, mock_vegetation):
        config = self._copy_config(CONFIG)
        config["thermal"]["correct_veg"] = True
        self.subject = ThermalHRRR(config=config, topo=TOPO_MOCK)

        self.subject.initialize(pd.DataFrame())
        self.subject.distribute("2025-09-20=9", RAW_DATA_MOCK, AIR_TEMP_MOCK)

        mock_vegetation.thermal_correct_canopy.assert_called_once()
