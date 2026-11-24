import unittest
from pathlib import Path

from unittest import mock
import xarray

import smrf
from smrf.data.hrrr.grib_file_xarray import GribFileXarray
from smrf.data.hrrr.grib_file_variables import (
    FIRST_HOUR,
    HRRR_HAG_10,
    HRRR_HAG_2,
    HRRR_SURFACE,
    SIXTH_HOUR,
)
from smrf.tests.smrf_test_case_lakes import SMRFTestCaseLakes

# RME
# BBOX = [-116.85837324, 42.96134124, -116.64913327, 43.16852535]
# Lakes test basin
BBOX = SMRFTestCaseLakes.BBOX
LOGGER = mock.Mock(name='Logger')

HRRR_FILE_DIR = Path(smrf.__file__).parent.joinpath(
    'tests', 'basins', 'Lakes', 'input'
)
HRRR_DAY_FOLDER = 'hrrr.20191001'


class TestGribFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.subject = GribFileXarray(LOGGER)
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

    def test_cell_size(self):
        self.assertEqual(3000, GribFileXarray.CELL_SIZE)

    def test_variables(self):
        self.assertEqual(
            [HRRR_SURFACE, HRRR_HAG_2, HRRR_HAG_10],
            GribFileXarray.HRRR_VARIABLES,
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

    def test_crops_to_bbox(self):
        hrrr_day = HRRR_FILE_DIR.joinpath(HRRR_DAY_FOLDER)

        with mock.patch.object(self.subject, "crop_to_bbox") as mock_crop:
            self.subject.load(
                file=hrrr_day.joinpath('hrrr.t15z.wrfsfcf01.grib2'),
                sixth_hour_file=False,
            )

            mock_crop.assert_called_once()

    @mock.patch.object(GribFileXarray, 'load_variable_level')
    def test_all_variables_from_sixth_hour_skip_first_hour(
        self, mock_load_variable
    ):
        sixth_hour_variables = (
            list(HRRR_SURFACE.smrf_map.values()) +
            list(HRRR_HAG_2.smrf_map.values())
        )
        combined = xarray.Dataset(
            {
                variable: ('time', [1.0])
                for variable in sixth_hour_variables
            }
        )
        mock_load_variable.return_value = xarray.Dataset()

        with mock.patch(
            'xarray.combine_by_coords',
            return_value=combined,
        ), mock.patch.object(
            self.subject,
            'crop_to_bbox',
            return_value=combined,
        ):
            self.subject.load(
                file='first-hour.grib2',
                sixth_hour_file='sixth-hour.grib2',
                sixth_hour_variables=sixth_hour_variables,
            )

        self.assertEqual(2, mock_load_variable.call_count)
        for call in mock_load_variable.call_args_list:
            self.assertEqual('sixth-hour.grib2', call.args[0])
            self.assertEqual(SIXTH_HOUR['stepRange'], call.args[1]['stepRange'])

    @mock.patch.object(GribFileXarray, 'load_variable_level')
    def test_mixed_first_and_sixth_hour_variables(
        self, mock_load_variable
    ):
        requested_variables = (
            list(HRRR_SURFACE.smrf_map.values()) +
            list(HRRR_HAG_2.smrf_map.values())
        )
        combined = xarray.Dataset(
            {
                variable: ('time', [1.0])
                for variable in requested_variables
            }
        )
        mock_load_variable.return_value = xarray.Dataset()

        with mock.patch(
            'xarray.combine_by_coords',
            return_value=combined,
        ), mock.patch.object(
            self.subject,
            'crop_to_bbox',
            return_value=combined,
        ):
            self.subject.load(
                file='first-hour.grib2',
                sixth_hour_file='sixth-hour.grib2',
                sixth_hour_variables=['precip_int'],
            )

        calls = mock_load_variable.call_args_list
        first_hour_calls = [
            call for call in calls
            if call.args[1]['stepRange'] == FIRST_HOUR['stepRange']
        ]
        sixth_hour_calls = [
            call for call in calls
            if call.args[1]['stepRange'] == SIXTH_HOUR['stepRange']
        ]

        self.assertEqual(2, len(first_hour_calls))
        self.assertEqual(1, len(sixth_hour_calls))
        self.assertTrue(
            all(call.args[0] == 'first-hour.grib2'
                for call in first_hour_calls)
        )
        self.assertTrue(
            all(call.args[0] == 'sixth-hour.grib2'
                for call in sixth_hour_calls)
        )
