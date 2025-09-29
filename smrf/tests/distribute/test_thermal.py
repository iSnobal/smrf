import unittest
from unittest.mock import MagicMock

import numpy as np
from smrf.data import Topo
from smrf.distribute import ThermalHRRR
from smrf.envphys.constants import EMISS_TERRAIN, STEF_BOLTZ

SKY_VIEW_FACTOR_MOCK = np.ones((1, 1))
RAW_DATA_MOCK = np.ones((1, 1))
AIR_TEMP_MOCK = np.ones((1, 1))

TOPO_MOCK = MagicMock(spec=Topo, sky_view_factor=SKY_VIEW_FACTOR_MOCK, instance=True)
DATA_MOCK = MagicMock(thermal=RAW_DATA_MOCK)


class TestThermal(unittest.TestCase):
    def setUp(self):
        self.subject = ThermalHRRR()

    def test_initialize(self):
        self.subject.initialize(TOPO_MOCK, DATA_MOCK)

        self.assertEqual(SKY_VIEW_FACTOR_MOCK, self.subject._sky_view_factor)
        self.assertEqual(RAW_DATA_MOCK, self.subject._forcing_data)

    def test_distribute(self):
        result = (SKY_VIEW_FACTOR_MOCK * RAW_DATA_MOCK) + (
            1 - SKY_VIEW_FACTOR_MOCK
        ) * EMISS_TERRAIN * STEF_BOLTZ * AIR_TEMP_MOCK**4

        self.subject.initialize(TOPO_MOCK, DATA_MOCK)
        self.subject.distribute("2025-09-20=9", AIR_TEMP_MOCK)

        self.assertEqual(result, self.subject.thermal)
