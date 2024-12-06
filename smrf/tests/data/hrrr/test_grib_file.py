import unittest
from pathlib import Path

import mock
import xarray

import smrf
from smrf.data.hrrr.grib_file import GribFile
from smrf.data.hrrr.grib_file_variables import (HRRR_HAG_10, HRRR_HAG_2,
                                                HRRR_SURFACE)

# RME
# BBOX = [-116.85837324, 42.96134124, -116.64913327, 43.16852535]
# Lakes test basin
BBOX = [-119.13778957, 37.4541464, -118.85206348, 37.73084705]
LOGGER = mock.Mock(name='Logger')

HRRR_FILE_DIR = Path(smrf.__file__).parent.joinpath(
    'tests', 'basins', 'Lakes', 'input'
)
HRRR_DAY_FOLDER = 'hrrr.20191001'


class TestGribFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.subject = GribFile(LOGGER)
        cls.subject.bbox = BBOX

    def test_set_bbox(self):
        self.assertEqual(
            BBOX,
            self.subject.bbox
        )

    def test_log_name(self):
        self.assertEqual(
            LOGGER,
            self.subject.log,
        )

    def test_file_suffix(self):
        self.assertEqual('grib2', GribFile.SUFFIX)

    def test_cell_size(self):
        self.assertEqual(3000, GribFile.CELL_SIZE)

    def test_variables(self):
        self.assertEqual(
            [HRRR_SURFACE, HRRR_HAG_2, HRRR_HAG_10],
            GribFile.HRRR_VARIABLES
        )

    def test_load_first_forecast_hour(self):
        hrrr_day = HRRR_FILE_DIR.joinpath(HRRR_DAY_FOLDER)

        data = self.subject.load(
            file=hrrr_day.joinpath('hrrr.t15z.wrfsfcf01.grib2'),
            sixth_hour_file=False,
        )

        self.assertIsInstance(data, xarray.Dataset)

    def test_maps_variables(self):
        hrrr_day = HRRR_FILE_DIR.joinpath(HRRR_DAY_FOLDER)

        data = self.subject.load(
            file=hrrr_day.joinpath('hrrr.t15z.wrfsfcf01.grib2'),
            sixth_hour_file=False,
        )
        requested_variables = (
            list(HRRR_SURFACE.smrf_map.values()) +
            list(HRRR_HAG_2.smrf_map.values())
        )

        # Number of returned Datasets (one per variable) matches the requested
        self.assertCountEqual(data.data_vars, requested_variables)
        self.assertEqual(
            sorted(list(data.data_vars)),
            sorted(requested_variables)
        )

    @mock.patch('xarray.open_dataset')
    def test_not_all_variables_found(self, xarray_patch):
        xarray_patch.return_value = xarray.Dataset()
        hrrr_day = HRRR_FILE_DIR.joinpath(HRRR_DAY_FOLDER)

        with self.assertRaises(Exception):
            self.subject.load(
                file=hrrr_day.joinpath('hrrr.t15z.wrfsfcf01.grib2'),
                sixth_hour_file=False,
            )

    def test_load_sixth_forecast_hour(self):
        hrrr_day = HRRR_FILE_DIR.joinpath(HRRR_DAY_FOLDER)

        data = self.subject.load(
            file=hrrr_day.joinpath('hrrr.t15z.wrfsfcf01.grib2'),
            sixth_hour_file=hrrr_day.joinpath('hrrr.t15z.wrfsfcf06.grib2'),
            sixth_hour_variables=['precip_int'],
        )

        requested_variables = (
            list(HRRR_SURFACE.smrf_map.values()) +
            list(HRRR_HAG_2.smrf_map.values())
        )

        # Number of returned Datasets (one per variable) matches the requested
        self.assertCountEqual(data.data_vars, requested_variables)
        self.assertEqual(
            sorted(list(data.data_vars)),
            sorted(requested_variables)
        )
