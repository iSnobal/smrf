import unittest
from unittest.mock import patch

import numpy as np
import numpy.testing as npt
import pandas as pd

from smrf.distribute import Albedo, SolarHRRR
from smrf.tests.distribute import SKY_VIEW_FACTOR_MOCK, topo_mock

DATETIME = pd.to_datetime("2025-11-01 00:00:00")
DATA_MOCK = {
    SolarHRRR.DSWRF: np.array([[20.0, 19.0]]),
    SolarHRRR.VBDSF: np.array([[16.0, 18.0]]),
    SolarHRRR.VDDSF: np.array([[5.0, 10.0]]),
}
COS_Z = np.cos(np.radians(10))
AZIMUTH = 100
ILLUMINATION_MOCK = np.array([[40.0, 50.0]])
ALBEDO_1 = np.array([[0.85, 0.9]]).astype(np.float32, order="C", copy=False)
ALBEDO_2 = np.array([[0.85, 0.75]]).astype(np.float32, order="C", copy=False)
COMPONENT_FLOOR = 0.1  # Mirrors COMPONENT_FLOOR in smrf/envphys/solar/toposplit.pyx
BELOW_FLOOR = COMPONENT_FLOOR - 0.01
ABOVE_FLOOR = COMPONENT_FLOOR + 0.01


class TestSolarHRRR(unittest.TestCase):
    def setUp(self):
        config = {
            "time": {
                "start_date": "2025-10-01 00:00",
                "time_zone": "utc",
            },
            "albedo": {
                "decay_method": "date_method",
            },
            "solar": {
                "correct_veg": False,
            },
        }
        self.subject = SolarHRRR(
            config=config,
            topo=topo_mock(),
        )
        self.albedo = Albedo(config=config, topo=topo_mock())
        self.albedo.albedo_vis = ALBEDO_1
        self.albedo.albedo_ir = ALBEDO_2

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
            self.albedo,
        )

        shade_mock.assert_called_once_with(
            COS_Z, AZIMUTH, ILLUMINATION_MOCK, self.subject.topo
        )

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
            1 - (0.54 * self.albedo.albedo_vis + 0.46 * self.albedo.albedo_ir)
        )
        npt.assert_equal(net_solar, self.subject.net_solar)

        vegetation_mock.solar_veg_beam.assert_not_called()
        vegetation_mock.solar_veg_diffuse.assert_not_called()

    @patch("smrf.distribute.solar_hrrr.vegetation")
    def test_distribute_with_vegetation(self, vegetation_mock):
        # Simulate the Toposplit call in distribute which sets the necessary attributes
        direct = np.array([[5.0, 5.0]])
        diffuse = np.array([[2.0, 2.0]])
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
            self.albedo,
        )

        empty = np.zeros_like(SKY_VIEW_FACTOR_MOCK)

        npt.assert_equal(empty, self.subject.solar_ghi_vis)
        npt.assert_equal(empty, self.subject.solar_k)
        npt.assert_equal(empty, self.subject.solar_dhi)
        npt.assert_equal(empty, self.subject.solar_dni)
        npt.assert_equal(empty, self.subject.hrrr_solar)
        npt.assert_equal(empty, self.subject.net_solar)

    @patch("smrf.distribute.solar_hrrr.mask_for_shade")
    def test_dswrf_below_threshold(self, shade_mock):
        """
        When DSWRF is below the minimum radiation threshold, the pixel is zeroed,
        regardless of VBDSF/VDDSF.
        """
        shade_mock.return_value = ILLUMINATION_MOCK, np.array([1, 1])

        data = {
            SolarHRRR.DSWRF: np.full_like(SKY_VIEW_FACTOR_MOCK, 0.0),
            SolarHRRR.VBDSF: np.full_like(SKY_VIEW_FACTOR_MOCK, 6.0),
            SolarHRRR.VDDSF: np.full_like(SKY_VIEW_FACTOR_MOCK, 5.0),
        }

        self.subject.distribute(
            DATETIME,
            data,
            COS_Z,
            AZIMUTH,
            ILLUMINATION_MOCK,
            self.albedo,
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

    @patch("smrf.distribute.solar_hrrr.mask_for_shade")
    def test_direct_below_floor_is_zeroed(self, shade_mock):
        """
        VBDSF below COMPONENT_FLOOR (a negative interpolation artifact in
        pixel 0, a small positive value in pixel 1) is set to 0. The pixel is
        still computed, as DSWRF and ghi_vis are above min_value, and is fully
        diffuse: k is 1.0 and dni/direct are 0.
        NB: data are dummy test values, not necessarily physically meaningful.
        """
        shade_mock.return_value = ILLUMINATION_MOCK, np.array([1, 1])

        data = {
            SolarHRRR.DSWRF: np.full_like(SKY_VIEW_FACTOR_MOCK, 19.0),
            SolarHRRR.VBDSF: np.array([[-1.0, BELOW_FLOOR]]),
            SolarHRRR.VDDSF: np.full_like(SKY_VIEW_FACTOR_MOCK, 10.0),
        }

        self.subject.distribute(
            DATETIME,
            data,
            COS_Z,
            AZIMUTH,
            ILLUMINATION_MOCK,
            self.albedo,
        )

        empty = np.zeros_like(SKY_VIEW_FACTOR_MOCK)
        ghi_vis = np.full_like(SKY_VIEW_FACTOR_MOCK, 10.0)
        k = np.full_like(SKY_VIEW_FACTOR_MOCK, 1.0)
        dhi = np.full_like(SKY_VIEW_FACTOR_MOCK, 19.0)

        npt.assert_equal(ghi_vis, self.subject.solar_ghi_vis)
        npt.assert_equal(k, self.subject.solar_k)
        npt.assert_equal(dhi, self.subject.solar_dhi)
        npt.assert_equal(empty, self.subject.solar_dni)
        npt.assert_equal(empty, self.subject.direct)
        npt.assert_equal(dhi * SKY_VIEW_FACTOR_MOCK, self.subject.diffuse)

    @patch("smrf.distribute.solar_hrrr.mask_for_shade")
    def test_diffuse_below_floor_is_zeroed(self, shade_mock):
        """
        VDDSF below COMPONENT_FLOOR (a negative interpolation artifact in
        pixel 0, a small positive value in pixel 1) is set to 0. The pixel is
        still computed, as DSWRF and ghi_vis are above min_value, and is fully
        direct: k is 0.0 and dhi/diffuse are 0.
        NB: data are dummy test values, not necessarily physically meaningful.
        """
        shade_mock.return_value = ILLUMINATION_MOCK, np.array([1, 1])

        data = {
            SolarHRRR.DSWRF: np.full_like(SKY_VIEW_FACTOR_MOCK, 19.0),
            SolarHRRR.VBDSF: np.full_like(SKY_VIEW_FACTOR_MOCK, 16.0),
            SolarHRRR.VDDSF: np.array([[-1.0, BELOW_FLOOR]]),
        }

        self.subject.distribute(
            DATETIME,
            data,
            COS_Z,
            AZIMUTH,
            ILLUMINATION_MOCK,
            self.albedo,
        )

        empty = np.zeros_like(SKY_VIEW_FACTOR_MOCK)
        ghi_vis = np.full_like(SKY_VIEW_FACTOR_MOCK, 16.0 * COS_Z)
        dni = np.full_like(SKY_VIEW_FACTOR_MOCK, 19.0 / COS_Z)

        npt.assert_equal(ghi_vis, self.subject.solar_ghi_vis)
        npt.assert_equal(empty, self.subject.solar_k)
        npt.assert_equal(empty, self.subject.solar_dhi)
        npt.assert_equal(dni, self.subject.solar_dni)
        npt.assert_equal(dni * ILLUMINATION_MOCK, self.subject.direct)
        npt.assert_equal(empty, self.subject.diffuse)

    @patch("smrf.distribute.solar_hrrr.mask_for_shade")
    def test_components_above_floor_are_kept(self, shade_mock):
        """
        Small components just above COMPONENT_FLOOR are not zeroed, so k stays
        strictly between 0 and 1. Pixel 0 has a small VDDSF, pixel 1 has a
        small VBDSF.
        NB: data are dummy test values, not necessarily physically meaningful.
        """
        shade_mock.return_value = ILLUMINATION_MOCK, np.array([1, 1])

        data = {
            SolarHRRR.DSWRF: np.full_like(SKY_VIEW_FACTOR_MOCK, 19.0),
            SolarHRRR.VBDSF: np.array([[16.0, ABOVE_FLOOR]]),
            SolarHRRR.VDDSF: np.array([[ABOVE_FLOOR, 10.0]]),
        }

        self.subject.distribute(
            DATETIME,
            data,
            COS_Z,
            AZIMUTH,
            ILLUMINATION_MOCK,
            self.albedo,
        )

        k = np.array(
            [
                [
                    ABOVE_FLOOR / (16.0 * COS_Z + ABOVE_FLOOR),
                    10.0 / (ABOVE_FLOOR * COS_Z + 10.0),
                ]
            ]
        )
        npt.assert_allclose(k, self.subject.solar_k)

    @patch("smrf.distribute.solar_hrrr.mask_for_shade")
    def test_small_fractional_subcomponents(self, shade_mock):
        """
        When both VBDSF/VDDSF are small (e.g., 1e-4) and below COMPONENT_FLOOR,
        they are set to 0, giving ghi_vis of 0 and zeroing the pixel even though
        DSWRF is above the min_value. The ghi_vis guard prevents a 0/0 in the k
        calculation, so no NaN/inf values are produced.
        NB: data are dummy test values, not necessarily physically meaningful.
        """
        shade_mock.return_value = ILLUMINATION_MOCK, np.array([1, 1])

        data = {
            SolarHRRR.DSWRF: np.full_like(SKY_VIEW_FACTOR_MOCK, 19.0),
            SolarHRRR.VBDSF: np.full_like(SKY_VIEW_FACTOR_MOCK, 1e-4),
            SolarHRRR.VDDSF: np.full_like(SKY_VIEW_FACTOR_MOCK, 1e-4),
        }

        self.subject.distribute(
            DATETIME,
            data,
            COS_Z,
            AZIMUTH,
            ILLUMINATION_MOCK,
            self.albedo,
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
