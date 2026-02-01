import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.testing as npt
import pandas as pd

from smrf.data import Topo
from smrf.distribute import SolarHRRR

SKY_VIEW_FACTOR_MOCK = np.array([[1.0, 1.0]])
TOPO_MOCK = MagicMock(
    spec=Topo,
    sky_view_factor=SKY_VIEW_FACTOR_MOCK,
    veg_height=np.array([[5.0, 10.0]]),
    veg_k=np.array([[0.8, 0.1]]),
    veg_tau=np.array([[0.6, 0.7]]),
    instance=True,
)

DATETIME = pd.to_datetime("2025-11-01 00:00:00")
DATA_MOCK = {
    SolarHRRR.DSWRF: np.array([[20.0, 19.0]]),
    SolarHRRR.VBDSF: np.array([[16.0, 18.0]]),
    SolarHRRR.VDDSF: np.array([[5.0, 10.0]]),
}
COS_Z = np.cos(np.radians(10))
AZIMUTH = 100
ILLUMINATION_MOCK = np.array([[40.0, 50.0]])
ALBEDO_MOCK = MagicMock(
    albedo_vis=np.array([[0.85, 0.9]]).astype(np.float32, order="C", copy=False),
    albedo_ir=np.array([[0.85, 0.75]]).astype(np.float32, order="C", copy=False),
)


class TestSolarHRRR(unittest.TestCase):
    def setUp(self):
        self.subject = SolarHRRR(
            config={
                "solar": {
                    "correct_veg": False,
                }
            },
            topo=TOPO_MOCK,
        )

    @patch("smrf.distribute.solar_hrrr.vegetation")
    @patch("smrf.distribute.solar_hrrr.mask_for_shade")
    def test_distribute(self, shade_mock, vegetation_mock):
        shade_mock.return_value = ILLUMINATION_MOCK, np.array([1, 1])

        self.subject.distribute(
            DATETIME,
            DATA_MOCK,
            COS_Z,
            AZIMUTH,
            ILLUMINATION_MOCK,
            ALBEDO_MOCK,
        )

        shade_mock.assert_called_once_with(COS_Z, AZIMUTH, ILLUMINATION_MOCK, TOPO_MOCK)

        ghi_vis = DATA_MOCK[SolarHRRR.VBDSF] * COS_Z + DATA_MOCK[SolarHRRR.VDDSF]
        npt.assert_equal(ghi_vis, self.subject.solar_ghi_vis)

        k = DATA_MOCK[SolarHRRR.VDDSF] / ghi_vis
        npt.assert_equal(k, self.subject.solar_k)

        dhi = DATA_MOCK[SolarHRRR.DSWRF] * k
        npt.assert_equal(dhi, self.subject.solar_dhi)

        dni = (DATA_MOCK[SolarHRRR.DSWRF] * (1 - k)) / COS_Z
        npt.assert_equal(dni, self.subject.solar_dni)

        direct = dni * ILLUMINATION_MOCK
        npt.assert_equal(direct, self.subject.direct)

        diffuse = dhi * SKY_VIEW_FACTOR_MOCK
        npt.assert_equal(diffuse, self.subject.diffuse)

        solar = direct.astype(np.float32, order="C", copy=False) + diffuse.astype(
            np.float32, order="C", copy=False
        )
        npt.assert_equal(solar, self.subject.hrrr_solar)

        net_solar = solar * (
            1 - (0.54 * ALBEDO_MOCK.albedo_vis + 0.46 * ALBEDO_MOCK.albedo_ir)
        )
        npt.assert_equal(net_solar, self.subject.net_solar)

        vegetation_mock.solar_veg_beam.assert_not_called()
        vegetation_mock.solar_veg_diffuse.assert_not_called()

    @patch("smrf.distribute.solar_hrrr.vegetation")
    def test_distribute_with_vegetation(self, vegetation_mock):
        # Simulate the Toposplit call in distribute which sets the necessary attributes
        direct = np.array([[5., 5.]])
        diffuse = np.array([[2., 2.]])
        self.subject.direct = direct
        self.subject.diffuse = diffuse

        self.subject.correct_vegetation(ILLUMINATION_MOCK)

        vegetation_mock.solar_veg_beam.assert_called_once_with(
            direct,
            self.subject.veg_height,
            ILLUMINATION_MOCK,
            self.subject.veg_k,
        )
        vegetation_mock.solar_veg_diffuse.assert_called_once_with(
            diffuse, self.subject.veg_tau
        )

    def test_distribute_sun_is_down(self):
        self.subject.distribute(
            DATETIME,
            DATA_MOCK,
            0,
            AZIMUTH,
            ILLUMINATION_MOCK,
            ALBEDO_MOCK,
        )

        empty = np.zeros_like(SKY_VIEW_FACTOR_MOCK)

        npt.assert_equal(empty, self.subject.solar_ghi_vis)
        npt.assert_equal(empty, self.subject.solar_k)
        npt.assert_equal(empty, self.subject.solar_dhi)
        npt.assert_equal(empty, self.subject.solar_dni)
        npt.assert_equal(empty, self.subject.hrrr_solar)
        npt.assert_equal(empty, self.subject.net_solar)

    @patch("smrf.distribute.solar_hrrr.mask_for_shade")
    def test_below_threshold(self, shade_mock):
        shade_mock.return_value = ILLUMINATION_MOCK, np.array([1, 1])

        self.subject.distribute(
            DATETIME,
            {
                SolarHRRR.DSWRF: np.array([[0.0, 19.0]]),
                SolarHRRR.VBDSF: np.array([[6.0, -1.0]]),
                SolarHRRR.VDDSF: np.array([[5.0, 10.0]]),
            },
            COS_Z,
            AZIMUTH,
            ILLUMINATION_MOCK,
            ALBEDO_MOCK,
        )

        empty = np.zeros_like(SKY_VIEW_FACTOR_MOCK)

        npt.assert_equal(empty, self.subject.solar_ghi_vis)
        npt.assert_equal(empty, self.subject.solar_k)
        npt.assert_equal(empty, self.subject.solar_dhi)
        npt.assert_equal(empty, self.subject.solar_dni)
        npt.assert_equal(empty, self.subject.direct)
        npt.assert_equal(empty, self.subject.diffuse)
        npt.assert_equal(empty, self.subject.hrrr_solar)
        npt.assert_equal(empty, self.subject.net_solar)

    def test_output_variables(self):
        for variable in self.subject.OUTPUT_VARIABLES.keys():
            self.assertTrue(
                hasattr(self.subject, variable),
                msg=f"SolarHRRR is missing attribute {variable}",
            )
