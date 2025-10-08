import unittest
from unittest import mock

import netCDF4 as nc
import numpy as np
from smrf.data import Topo
from smrf.tests.smrf_test_case import SMRFTestCase


class TestLoadTopo(unittest.TestCase):
    TOPO_CONFIG = {
        "filename": SMRFTestCase.topo_nc(),
        "northern_hemisphere": True,
        "gradient_method": "gradient_d8",
        "sky_view_factor_angles": 72,
    }

    @classmethod
    def setUp(cls):
        cls.ds = nc.Dataset(cls.TOPO_CONFIG['filename'])
        cls.topo = Topo(cls.TOPO_CONFIG)

    @classmethod
    def tearDown(cls):
        cls.ds.close()

    def test_read_topo_images(self):
        self.assertEqual(
            [
                "burn_mask",
                "dem",
                "mask",
                "veg_height",
                "veg_k",
                "veg_tau",
                "veg_type",
            ],
            Topo.IMAGES
        )

    def test_topo_image_as_variables(self):
        for variable in Topo.IMAGES:
            self.assertTrue(hasattr(self.topo, variable))

    def test_no_burn_mask_required(self):
        self.assertIsNone(self.topo.burn_mask)

    @mock.patch.object(Topo, 'readNetCDF')
    @mock.patch.object(Topo, 'gradient')
    def test_init(self, gradient, read_nc):
        topo = Topo(self.TOPO_CONFIG)
        self.assertEqual(self.TOPO_CONFIG, topo.topoConfig)
        gradient.assert_called_once()
        read_nc.assert_called_once()

    def test_topo_gdal_attributes(self):
        self.assertEqual("EPSG:32611", self.topo.gdal_attributes.srs)
        self.assertEqual(
            [519650.0, 4767630.0, 520450.0, 4768480.0],
            self.topo.gdal_attributes.outputBounds,
        )
        self.assertEqual(50.0, self.topo.gdal_attributes.xRes)
        self.assertEqual(50.0, self.topo.gdal_attributes.yRes)

    def test_attribute_file(self):
        self.assertEqual(self.TOPO_CONFIG['filename'], self.topo.file)

    def test_center_calc_masked(self):
        """
        Test the basin center calculation using the basin mask
        """
        cx, cy = self.topo.get_center(self.ds, mask_name='mask')
        np.testing.assert_almost_equal(cx, 520033.7187500, 7)
        np.testing.assert_almost_equal(cy, 4768035.0, 7)

    def test_center_calc_domain(self):
        """
        Test the basin center calculation for the entire basin domain
        """
        cx, cy = self.topo.get_center(self.ds, mask_name=None)
        np.testing.assert_almost_equal(cx, 520050.0, 7)
        np.testing.assert_almost_equal(cy, 4768055.0, 7)

    def test_auto_calc_lat_lon(self):
        """
        Test calculating the basin lat long correctly
        """
        # Original RME
        # basin_lon:                     -116.7547
        # basin_lat:                     43.067
        np.testing.assert_almost_equal(
            self.topo.basin_lat, 43.06475372378507, 7)
        np.testing.assert_almost_equal(
            self.topo.basin_long, -116.75395420397061, 7)

    def test_projection_attributes(self):
        """
        Confirm that this class has important projection attributes
        """
        # Attribute directly used in load Grid as attributes from topo class
        important = ['basin_lat', 'basin_long', 'zone_number',
                     'northern_hemisphere']

        for at in important:
            self.assertTrue(hasattr(self.topo, at))
