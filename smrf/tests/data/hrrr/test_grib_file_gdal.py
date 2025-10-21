import unittest
from datetime import datetime, timezone

import numpy as np
from osgeo import gdal
from smrf.data.hrrr.grib_file_gdal import GribFileGdal
from smrf.data.load_topo import Topo
from smrf.tests.smrf_test_case_lakes import SMRFTestCaseLakes


class TestGribFile(unittest.TestCase):
    TOPO_NC = Topo(
        {
            "filename": SMRFTestCaseLakes.topo_nc().as_posix(),
            "northern_hemisphere": True,
            "gradient_method": "gradient_d8",
            "sky_view_factor_angles": 72,
        }
    )
    HRRR_INPUT = SMRFTestCaseLakes.input_dir.joinpath(
        "hrrr.20191001", "hrrr.t14z.wrfsfcf01.grib2"
    )

    def setUp(self):
        self.grib_gdal = GribFileGdal(topo=self.TOPO_NC, resample_method="cubic")

    def test_init(self):
        self.assertEqual(self.TOPO_NC, self.grib_gdal.topo)
        self.assertEqual(gdal.GRA_Cubic, self.grib_gdal.resample_method)

    def test_get_grib_metadata(self):
        band_map, valid_time = self.grib_gdal.get_grib_metadata(self.HRRR_INPUT)

        self.assertEqual(
            {
                "APCP01": 6,
                "DSWRF": 8,
                "HGT": 1,
                "RH": 3,
                "TCDC": 7,
                "TMP": 2,
                "UGRD": 4,
                "VGRD": 5,
            },
            band_map,
        )
        self.assertEqual(
            {
                "APCP01": datetime(2019, 10, 1, 15, 0, tzinfo=timezone.utc),
                "DSWRF": datetime(2019, 10, 1, 15, 0, tzinfo=timezone.utc),
                "HGT": datetime(2019, 10, 1, 15, 0, tzinfo=timezone.utc),
                "RH": datetime(2019, 10, 1, 15, 0, tzinfo=timezone.utc),
                "TCDC": datetime(2019, 10, 1, 15, 0, tzinfo=timezone.utc),
                "TMP": datetime(2019, 10, 1, 15, 0, tzinfo=timezone.utc),
                "UGRD": datetime(2019, 10, 1, 15, 0, tzinfo=timezone.utc),
                "VGRD": datetime(2019, 10, 1, 15, 0, tzinfo=timezone.utc),
            },
            valid_time,
        )

    def test_warp_and_cut(self):
        with self.grib_gdal.gdal_warp_and_cut(
            self.HRRR_INPUT, [(1, 1), (2, 2), (3, 3)]
        ) as dataset:
            self.assertEqual(3, dataset.RasterCount)
            self.assertEqual(
                (319975.0, 50.0, 0.0, 4166675.0, 0.0, -50.0), dataset.GetGeoTransform()
            )

    def test_load(self):
        hrrr_variable = "DSWRF"
        data = self.grib_gdal.load([hrrr_variable], self.HRRR_INPUT)

        self.assertTrue(hrrr_variable in data)
        self.assertIsInstance(data[hrrr_variable], np.ndarray)
