import unittest
from unittest.mock import MagicMock

import numpy as np
import numpy.testing as npt

from smrf.data import Topo
from smrf.distribute.albedo import Albedo
from smrf.envphys.solar.net_solar import NetSolar

TOPO_MOCK = MagicMock(spec=Topo, instance=True)
SOLAR_1 = np.array([[400, 500], [350, 440]]).astype(np.float32, order="C", copy=False)
SOLAR_2 = np.array([[300, 400], [250, 340]]).astype(np.float32, order="C", copy=False)
ALBEDO_1 = np.array([[0.85, 0.9]]).astype(np.float32, order="C", copy=False)
ALBEDO_2 = np.array([[0.85, 0.75]]).astype(np.float32, order="C", copy=False)


class TestNetSolar(unittest.TestCase):
    def setUp(self):
        config = {
            "time": {
                "start_date": "2025-10-01 00:00",
                "time_zone": "utc",
            },
            "albedo": {
                "decay_method": "date_method",
            },
        }
        self.albedo = Albedo(config=config, topo=TOPO_MOCK)

    def test_broadband_albedo(self):
        self.albedo.albedo = ALBEDO_1

        result = NetSolar.broadband_albedo(SOLAR_1, self.albedo)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, SOLAR_1.shape)

        expected = SOLAR_1 * (1 - ALBEDO_1)
        npt.assert_equal(expected, result)

    def test_broadband_from_vis_ir(self):
        self.albedo.albedo_vis = ALBEDO_1
        self.albedo.albedo_ir = ALBEDO_2

        result = NetSolar.broadband_from_vis_ir(SOLAR_1, self.albedo)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, SOLAR_1.shape)

        expected = SOLAR_1 * (1 - (0.54 * ALBEDO_1 + 0.46 * ALBEDO_2))
        npt.assert_equal(expected, result)

    def test_albedo_diffuse_and_direct(self):
        self.albedo.albedo_direct = ALBEDO_1
        self.albedo.albedo_diffuse = ALBEDO_2

        result = NetSolar.albedo_diffuse_and_direct(SOLAR_1, SOLAR_2, self.albedo)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, SOLAR_1.shape)

        expected = SOLAR_1 * (1 - ALBEDO_1) + SOLAR_2 * (1 - ALBEDO_2)
        npt.assert_equal(expected, result)
