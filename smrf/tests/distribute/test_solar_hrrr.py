import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.testing as npt

import pandas as pd
from smrf.data import Topo
from smrf.distribute import SolarHRRR

RAW_DATA_MOCK = np.array([[20, 20]])

SKY_VIEW_FACTOR_MOCK = np.ones((1, 2))
TOPO_MOCK = MagicMock(spec=Topo, sky_view_factor=SKY_VIEW_FACTOR_MOCK, instance=True)

DATETIME = pd.to_datetime("2025-11-01 00:00:00")
DATA_MOCK = {
    SolarHRRR.DSWRF: RAW_DATA_MOCK,
    SolarHRRR.VBDSF: RAW_DATA_MOCK,
    SolarHRRR.VDDSF: RAW_DATA_MOCK,
}
COS_Z = np.cos(np.radians(10))
AZIMUTH = 100
ILLUMINATION_MOCK = RAW_DATA_MOCK
ALBEDO_MOCK = RAW_DATA_MOCK


class TestSolarHRRR(unittest.TestCase):
    def setUp(self):
        self.subject = SolarHRRR(config={}, topo=TOPO_MOCK)

    @patch("smrf.distribute.solar_hrrr.mask_for_shade")
    def test_distribute(self, shade_mock):
        shade_mock.return_value = ILLUMINATION_MOCK, RAW_DATA_MOCK

        self.subject.distribute(
            DATETIME,
            DATA_MOCK,
            COS_Z,
            AZIMUTH,
            ILLUMINATION_MOCK,
            ALBEDO_MOCK,
            ALBEDO_MOCK,
        )

        shade_mock.assert_called_once_with(COS_Z, AZIMUTH, ILLUMINATION_MOCK, TOPO_MOCK)

        ghi_vis = DATA_MOCK[SolarHRRR.VBDSF] * COS_Z + DATA_MOCK[SolarHRRR.VDDSF]
        npt.assert_equal(ghi_vis, self.subject.solar_ghi_vis)

        k = DATA_MOCK[SolarHRRR.VDDSF] / ghi_vis
        npt.assert_equal(k, self.subject.solar_k)

        dhi = RAW_DATA_MOCK * k
        npt.assert_equal(dhi, self.subject.solar_dhi)

        dni = (RAW_DATA_MOCK * ( 1 - k )) / COS_Z
        npt.assert_equal(dni, self.subject.solar_dni)

        solar = dni * ILLUMINATION_MOCK + dhi * SKY_VIEW_FACTOR_MOCK
        npt.assert_equal(solar, self.subject.hrrr_solar)

        net_solar = solar * ( 1- (0.54 * ALBEDO_MOCK + 0.46 * ALBEDO_MOCK))
        npt.assert_equal(net_solar, self.subject.net_solar)

    def test_distribute_sun_is_down(self):
        self.subject.distribute(
            DATETIME,
            DATA_MOCK,
            0,
            AZIMUTH,
            ILLUMINATION_MOCK,
            ALBEDO_MOCK,
            ALBEDO_MOCK,
        )

        empty = np.zeros_like(SKY_VIEW_FACTOR_MOCK)

        npt.assert_equal(empty, self.subject.solar_ghi_vis)
        npt.assert_equal(empty, self.subject.solar_k)
        npt.assert_equal(empty, self.subject.solar_dhi)
        npt.assert_equal(empty, self.subject.solar_dni)
        npt.assert_equal(empty, self.subject.hrrr_solar)
        npt.assert_equal(empty, self.subject.net_solar)

    def test_output_variables(self):
        for variable in self.subject.OUTPUT_VARIABLES.keys():
            self.assertTrue(
                hasattr(self.subject, variable),
                msg=f"SolarHRRR is missing attribute {variable}",
            )
