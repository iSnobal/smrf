import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import pandas as pd
from smrf.data import Topo
from smrf.distribute import Solar

SKY_VIEW_FACTOR_MOCK = np.ones((1, 2, 4))
TOPO_MOCK = MagicMock(spec=Topo, sky_view_factor=SKY_VIEW_FACTOR_MOCK, instance=True)

DATETIME = pd.to_datetime("2025-11-01 00:00:00")
COS_Z = np.cos(np.radians(10))
AZIMUTH = 100
CLOUD_FACTOR = np.array([[0.0, 0.5, 1.0]])
ILLUMINATION_MOCK = np.array([[15.0, 20.0, 18.0]])
ALBEDO_MOCK = MagicMock(
    albedo_vis=np.array([[0.85, 0.75, 0.9]]),
    albedo_ir=np.array([[0.85, 0.75, 0.9]]),
)
MOCK_SOLAR = (
    np.array([[150.0, 200.0, 180.0]]),
    np.array([[120.0, 150.0, 110.0]]),
    np.array([[22.0, 25.0, 21.0]])
)


class TestSolar(unittest.TestCase):
    def setUp(self):
        self.subject = Solar(
            config={
                'solar': {
                    'correct_cloud': False,
                    'correct_veg': False,
                }
            },
            topo=TOPO_MOCK
        )

    def test_initialize(self):
        self.assertIsNone(self.subject.vis_beam)
        self.assertIsNone(self.subject.vis_diffuse)
        self.assertIsNone(self.subject.ir_beam)
        self.assertIsNone(self.subject.ir_diffuse)
        self.assertIsNone(self.subject.cloud_factor)

        self.assertIsNone(self.subject.clear_vis_beam)
        self.assertIsNone(self.subject.clear_vis_diffuse)
        self.assertIsNone(self.subject.clear_ir_beam)
        self.assertIsNone(self.subject.clear_ir_diffuse)

        self.assertIsNone(self.subject.cloud_vis_beam)
        self.assertIsNone(self.subject.cloud_vis_diffuse)
        self.assertIsNone(self.subject.cloud_ir_beam)
        self.assertIsNone(self.subject.cloud_ir_diffuse)

        self.assertIsNone(self.subject.veg_vis_beam)
        self.assertIsNone(self.subject.veg_vis_diffuse)
        self.assertIsNone(self.subject.veg_ir_beam)
        self.assertIsNone(self.subject.veg_ir_diffuse)

        self.assertIsNone(self.subject.net_solar)

    @patch.object(Solar, "calc_stoporad")
    def test_distribute(self, toporad_mock):
        toporad_mock.return_value = MOCK_SOLAR

        self.subject.distribute(
            DATETIME,
            CLOUD_FACTOR,
            ILLUMINATION_MOCK,
            COS_Z,
            AZIMUTH,
            ALBEDO_MOCK,
        )

        assert toporad_mock.call_count == 2

        self.assertIsNotNone(self.subject.vis_beam)
        self.assertIsNotNone(self.subject.vis_diffuse)
        self.assertIsNotNone(self.subject.ir_beam)
        self.assertIsNotNone(self.subject.ir_diffuse)
        self.assertIsNotNone(self.subject.cloud_factor)

        self.assertIsNotNone(self.subject.clear_vis_beam)
        self.assertIsNotNone(self.subject.clear_vis_diffuse)
        self.assertIsNotNone(self.subject.clear_ir_beam)
        self.assertIsNotNone(self.subject.clear_ir_diffuse)

        self.assertIsNotNone(self.subject.net_solar)

        self.assertIsNone(self.subject.cloud_vis_beam)
        self.assertIsNone(self.subject.cloud_vis_diffuse)
        self.assertIsNone(self.subject.cloud_ir_beam)
        self.assertIsNone(self.subject.cloud_ir_diffuse)

        self.assertIsNone(self.subject.veg_vis_beam)
        self.assertIsNone(self.subject.veg_vis_diffuse)
        self.assertIsNone(self.subject.veg_ir_beam)
        self.assertIsNone(self.subject.veg_ir_diffuse)

    def test_distribute_sun_is_down(self):
        self.subject.distribute(
            DATETIME,
            CLOUD_FACTOR,
            ILLUMINATION_MOCK,
            0,
            AZIMUTH,
            ALBEDO_MOCK,
        )

        self.assertIsNone(self.subject.vis_beam)
        self.assertIsNone(self.subject.vis_diffuse)
        self.assertIsNone(self.subject.ir_beam)
        self.assertIsNone(self.subject.ir_diffuse)
        self.assertIsNone(self.subject.cloud_factor)

        self.assertIsNone(self.subject.clear_vis_beam)
        self.assertIsNone(self.subject.clear_vis_diffuse)
        self.assertIsNone(self.subject.clear_ir_beam)
        self.assertIsNone(self.subject.clear_ir_diffuse)

        self.assertIsNone(self.subject.cloud_vis_beam)
        self.assertIsNone(self.subject.cloud_vis_diffuse)
        self.assertIsNone(self.subject.cloud_ir_beam)
        self.assertIsNone(self.subject.cloud_ir_diffuse)

        self.assertIsNone(self.subject.veg_vis_beam)
        self.assertIsNone(self.subject.veg_vis_diffuse)
        self.assertIsNone(self.subject.veg_ir_beam)
        self.assertIsNone(self.subject.veg_ir_diffuse)

        self.assertIsNone(self.subject.net_solar)

    @patch.object(Solar, "calc_stoporad")
    def test_distribute_sun_up_to_down(self, toporad_mock):
        """
        Simulate a daily run where the sun was up and down and ensure we don't keep
        any "leftover" values from previous hours
        """
        toporad_mock.return_value = MOCK_SOLAR

        self.subject.distribute(
            DATETIME,
            CLOUD_FACTOR,
            ILLUMINATION_MOCK,
            COS_Z,
            AZIMUTH,
            ALBEDO_MOCK,
        )

        self.assertIsNotNone(self.subject.vis_beam)
        self.assertIsNotNone(self.subject.vis_diffuse)
        self.assertIsNotNone(self.subject.ir_beam)
        self.assertIsNotNone(self.subject.ir_diffuse)
        self.assertIsNotNone(self.subject.cloud_factor)

        self.assertIsNotNone(self.subject.clear_vis_beam)
        self.assertIsNotNone(self.subject.clear_vis_diffuse)
        self.assertIsNotNone(self.subject.clear_ir_beam)
        self.assertIsNotNone(self.subject.clear_ir_diffuse)

        self.assertIsNotNone(self.subject.net_solar)

        # Distribute after sunset
        self.subject.distribute(
            DATETIME,
            CLOUD_FACTOR,
            ILLUMINATION_MOCK,
            0,
            AZIMUTH,
            ALBEDO_MOCK,
        )

        self.assertIsNone(self.subject.vis_beam)
        self.assertIsNone(self.subject.vis_diffuse)
        self.assertIsNone(self.subject.ir_beam)
        self.assertIsNone(self.subject.ir_diffuse)
        self.assertIsNone(self.subject.cloud_factor)

        self.assertIsNone(self.subject.veg_vis_beam)
        self.assertIsNone(self.subject.veg_vis_diffuse)
        self.assertIsNone(self.subject.veg_ir_beam)
        self.assertIsNone(self.subject.veg_ir_diffuse)

        self.assertIsNone(self.subject.net_solar)

        assert toporad_mock.call_count == 2
