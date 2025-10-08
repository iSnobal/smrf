import unittest
from unittest import mock
from sys import platform

import netCDF4 as nc
import numpy as np
import numpy.testing as npt
from netCDF4 import Dataset
from topocalc.viewf import viewf

from smrf.data.load_topo import Topo
from smrf.tests.smrf_test_case import SMRFTestCase

TOPO_CONFIG = {
    "filename": SMRFTestCase.topo_nc(),
    "northern_hemisphere": True,
    "gradient_method": "gradient_d8",
    "sky_view_factor_angles": 72,
}


class TestLoadTopo(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.ds = nc.Dataset(TOPO_CONFIG["filename"])
        cls.topo = Topo(TOPO_CONFIG)

    @classmethod
    def tearDown(cls):
        cls.ds.close()

<<<<<<< HEAD
    @mock.patch.object(Topo, "gradient")
    @mock.patch.object(Topo, "readNetCDF")
    def test_init(self, read_nc, gradient):
        topo = Topo(TOPO_CONFIG)
        self.assertEqual(TOPO_CONFIG, topo.topoConfig)
=======
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
>>>>>>> 190ae1b (Topo - Read in a burn mask if present.)
        read_nc.assert_called_once()
        gradient.assert_called_once()

    def test_topo_gdal_attributes(self):
        self.assertEqual("EPSG:32611", self.topo.gdal_attributes.srs)
        self.assertEqual(
            [519650.0, 4767630.0, 520450.0, 4768480.0],
            self.topo.gdal_attributes.outputBounds,
        )
        self.assertEqual(50.0, self.topo.gdal_attributes.xRes)
        self.assertEqual(50.0, self.topo.gdal_attributes.yRes)

    def test_attribute_file(self):
        self.assertEqual(TOPO_CONFIG["filename"], self.topo.file)

    def test_center_calc_masked(self):
        """
        Test the basin center calculation using the basin mask
        """
        cx, cy = self.topo.get_center(self.ds, mask_name="mask")
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
        np.testing.assert_almost_equal(self.topo.basin_lat, 43.06475372378507, 7)
        np.testing.assert_almost_equal(self.topo.basin_long, -116.75395420397061, 7)

    def test_projection_attributes(self):
        """
        Confirm that this class has important projection attributes
        """
        important = ["basin_lat", "basin_long", "zone_number", "northern_hemisphere"]

        for at in important:
            self.assertTrue(hasattr(self.topo, at))


class TestMissingSkyViewFactor(SMRFTestCase):
    def setUp(self):
        super().setUp()
        self.test_topo_path = self.output_dir.joinpath("test_topo_svf.nc")

        # Create a skeleton NetCDF copy of the topo file.
        # Only the DEM and necessary dimensions/projection are copied.
        with Dataset(TOPO_CONFIG["filename"], "r") as src:
            with Dataset(self.test_topo_path, "w") as dst:
                for name, dimension in src.dimensions.items():
                    dst.createDimension(
                        name, (len(dimension) if not dimension.isunlimited() else None)
                    )

                for var_name in ["dem", "projection", "x", "y", "mask"]:
                    if var_name in src.variables:
                        var = src.variables[var_name]
                        out_var = dst.createVariable(
                            var_name, var.datatype, var.dimensions
                        )
                        out_var.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
                        out_var[:] = var[:]

        self.config = TOPO_CONFIG.copy()
        self.config["filename"] = str(self.test_topo_path)

        # The __init__ triggers calculate_sky_view_factor if vars are missing
        self.subject = Topo(self.config)

    def test_calculate_sky_view_factor_on_disk(self):
        sky_view_factor, terrain_config_factor = viewf(
            self.subject.dem, self.subject.dx, TOPO_CONFIG["sky_view_factor_angles"]
        )

        with Dataset(self.test_topo_path, "r") as test:
            for var in ["sky_view_factor", "terrain_config_factor", "slope"]:
                self.assertIn(var, test.variables)
                self.assertEqual(
                    test.variables[var].shape, (self.subject.ny, self.subject.nx)
                )
                self.assertEqual(test.variables[var].dtype, np.float64)

            npt.assert_allclose(
                test.variables["sky_view_factor"][:], sky_view_factor, rtol=1e-7
            )
            npt.assert_allclose(
                test.variables["terrain_config_factor"][:],
                terrain_config_factor,
                rtol=1e-7,
            )

            npt.assert_allclose(
                test.variables["slope"][:],
                np.degrees(self.subject.slope_radians),
                rtol=1e-7,
            )

            self.assertEqual(
                test.variables["sky_view_factor"].getncattr("long_name"),
                f"Sky view factor for {self.config['sky_view_factor_angles']} angles",
            )

    def test_topo_gold(self):
        """
        This test acts as a warning system to any changes to the TopoCalc sky_view_factor
        method.
        """
        with Dataset(TOPO_CONFIG["filename"], "r") as gold:
            with Dataset(self.test_topo_path, "r") as test:
                self.assertIn("slope", test.variables)
                self.assertIn("sky_view_factor", test.variables)
                self.assertIn("terrain_config_factor", test.variables)

                # OSX test fails on slope check while linux does not
                if platform == "darwin":
                    npt.assert_allclose(
                        test.variables["slope"][:],
                        gold.variables["slope"][:],
                        atol=1e-6,
                    )
                else:
                    npt.assert_array_equal(
                        test.variables["slope"][:],
                        gold.variables["slope"][:],
                    )

                npt.assert_allclose(
                    test.variables["sky_view_factor"][:],
                    gold.variables["sky_view_factor"][:],
                    atol=0.025,
                )
                npt.assert_allclose(
                    test.variables["terrain_config_factor"][:],
                    gold.variables["terrain_config_factor"][:],
                    atol=0.025,
                )
