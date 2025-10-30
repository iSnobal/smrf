import re
import unittest
from unittest import mock

import pandas as pd

from smrf.data.hrrr.file_loader import FileLoader
from smrf.data.hrrr.grib_file_gdal import GribFileGdal
from smrf.data.hrrr.grib_file_xarray import GribFileXarray

from smrf.tests.smrf_test_case_lakes import SMRFTestCaseLakes

FILE_DIR = "/path/to/files"
START_DT = pd.to_datetime("2018-07-22 01:00")

BBOX = mock.Mock(name="Bounding Box")
TOPO = mock.Mock(name="Topo NC")
UTM_NUMBER = 12
LOGGER = mock.Mock(name="Logger")

MOCK_DATA = {"var": mock.Mock(name="data")}


class TestFileLoader(unittest.TestCase):
    def setUp(self):
        self.subject = FileLoader(
            file_dir=FILE_DIR,
            forecast_hour=1,
            external_logger=LOGGER
        )

    def test_file_dir_property(self):
        assert self.subject.file_dir.endswith(FILE_DIR)

    def test_loads_grib2_suffix(self):
        self.assertEqual('grib2', self.subject.NAME_SUFFIX)

    def test_defaults_to_first_forecast_hour(self):
        self.assertEqual(self.subject._forecast_hour, 1)

    def test_defaults_to_no_sixth_hour_variables(self):
        self.assertEqual(self.subject._sixth_hour_variables, None)

    def test_can_set_sixth_hour_variables(self):
        variables = ['precip_int']
        self.subject = FileLoader(FILE_DIR, 1, variables)
        self.assertEqual(self.subject._sixth_hour_variables, variables)

    def test_change_file_dir(self):
        new_dir = 'somewhere/else'
        self.subject.file_dir = new_dir
        assert self.subject.file_dir.endswith(new_dir)

    def test_logger_to_file_loader(self):
        self.assertEqual(LOGGER, self.subject.log)

    def test_default_to_wind_load_false(self):
        self.assertFalse(self.subject._load_wind)

    @mock.patch.object(FileLoader, 'xarray', return_value=MOCK_DATA)
    def test_data_for_time_and_topo_no_gdal(self, xarray_mock):
        data = self.subject.data_for_time_and_topo(
            START_DT, BBOX, TOPO
        )
        xarray_mock.assert_called_once()
        self.assertEqual(['var'], list(data.keys()))

    @mock.patch.object(FileLoader, 'xarray', return_value=MOCK_DATA)
    @mock.patch.object(FileLoader, 'gdal', return_value={'var2': mock.Mock(name='GDAL')})
    def test_data_for_time_and_topo_with_gdal(self, gdal_mock, xarray_mock):
        subject = FileLoader(
            FILE_DIR, 1, ['precip_int'], load_gdal=['HRRR_VAR'], external_logger=LOGGER
        )
        data = subject.data_for_time_and_topo(
            START_DT, BBOX, TOPO
        )
        xarray_mock.assert_called_once()
        gdal_mock.assert_called_once()
        self.assertEqual(['var', 'var2'], list(data.keys()))

    @mock.patch("smrf.data.hrrr.file_loader.GribFileGdal")
    def test_gdal(self, mocK_grib_gdal):
        mock_instance = mocK_grib_gdal.return_value
        self.subject.gdal(START_DT, TOPO)

        mocK_grib_gdal.assert_called_once_with(TOPO, GribFileGdal.DEFAULT_ALGORITHM)
        mock_instance.load.assert_called_once_with(
            self.subject._load_gdal,
            "/path/to/files/hrrr.20180721/hrrr.t19z.wrfsfcf06.grib2"
        )

class TestFileLoaderXarray(unittest.TestCase,):
    METHOD_ARGS = [START_DT, BBOX]

    def setUp(self):
        self.subject = FileLoader(
            file_dir=FILE_DIR,
            forecast_hour=1,
            external_logger=LOGGER
        )

        self.file_loader_mock = mock.Mock(spec=GribFileXarray)
        self.file_loader_patch = mock.patch(
            "smrf.data.hrrr.file_loader.GribFileXarray",
            return_value=self.file_loader_mock,
        )
        self.file_loader_patch.start()

        self.xr_patcher = mock.patch("xarray.combine_by_coords")
        self.xr_patcher.start()

        self.convert_patch = mock.patch.object(
            FileLoader,
            'convert_to_dataframes',
            return_value=MOCK_DATA
        )
        self.convert_patch.start()

        LOGGER.debug = mock.Mock()
        LOGGER.error = mock.Mock()

    def tearDown(self):
        self.file_loader_patch.stop()
        self.xr_patcher.stop()
        self.convert_patch.stop()

    def test_sets_file_loader_bbox(self):
        with mock.patch('os.path.exists', return_value=True):
            self.subject.xarray(*self.METHOD_ARGS)

        self.assertEqual(BBOX, self.file_loader_mock.bbox)

    def test_load_attempts_per_timestamp(self):
        with mock.patch('os.path.exists', return_value=True):
            self.subject.xarray(*self.METHOD_ARGS)

        self.assertEqual(
            1,
            self.file_loader_mock.load.call_count,
            msg='More data was loaded than requested dates'
        )
        # Check file path of last loaded file
        self.assertRegex(
            self.file_loader_mock.load.call_args[1]['file'],
            r'.*/hrrr.20180722/hrrr.t00z.wrfsfcf01.grib2',
            msg='Path to file not passed to file loader'
        )
        self.assertEqual(
            self.subject._load_wind,
            self.file_loader_mock.load.call_args[1]['load_wind'],
            msg='Parameter: load_wind not passed to file loader'
        )

    def test_can_not_load_first_forecast_hour(self):
        message = "Data loading error"
        data_exception = Exception(message)
        self.file_loader_mock.load.side_effect = data_exception

        with mock.patch('os.path.exists', return_value=True):
            with self.assertRaisesRegex(
                type(data_exception), message
            ):
                self.subject.xarray(*self.METHOD_ARGS)

            self.assertEqual(
                1,
                self.file_loader_mock.load.call_count,
                msg='Tried to load more than one forecast hours for a '
                    'single time step'
            )

    def test_file_not_found(self):
        with mock.patch('os.path.exists', return_value=False):
            with self.assertRaises(FileNotFoundError):
                self.subject.xarray(*self.METHOD_ARGS)

        self.assertEqual(
            0,
            self.file_loader_mock.load.call_count,
            msg='Tried to load data from file although not present on disk'
        )

class TestFileLoaderMetadata(SMRFTestCaseLakes):
    DATE = pd.to_datetime("2019-10-01 17:00")

    def setUp(self):
        self.subject = FileLoader(
            file_dir=self.input_dir,
            forecast_hour=1,
            external_logger=LOGGER
        )

        LOGGER.debug = mock.Mock()

    def test_metadata_from_first_hour(self):
        data = self.subject.get_metadata(self.DATE, self.BBOX, 11)

        self.assertIsInstance(data, pd.DataFrame)

        # Required columns for interpolation
        self.assertListEqual(
            ['latitude', 'longitude', 'elevation', 'utm_x', 'utm_y'],
            data.columns.tolist()
        )
        # Required naming convention for index values
        self.assertTrue(data.index.values[0].startswith("grid_"))

    def test_metadata_raises_on_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            self.subject.get_metadata(
                self.DATE + pd.to_timedelta(3, "hour"), self.BBOX, 11
            )

class TestFileLoaderSixthHour(TestFileLoaderXarray):
    SIXTH_HOUR_VARIABLE = ['precip_int']

    def setUp(self):
        super().setUp()
        self.subject = FileLoader(
            file_dir=FILE_DIR,
            forecast_hour=1,
            sixth_hour_variables=self.SIXTH_HOUR_VARIABLE,
            external_logger=LOGGER
        )

    def test_load_attempts_per_timestamp(self):
        with mock.patch('os.path.exists', return_value=True):
            self.subject.xarray(*self.METHOD_ARGS)

        self.assertEqual(
            1,
            self.file_loader_mock.load.call_count,
            msg='More data was loaded than requested dates'
        )
        # Check arguments of last loaded file
        self.assertRegex(
            self.file_loader_mock.load.call_args[1]['file'],
            r'.*/hrrr.20180722/hrrr.t00z.wrfsfcf01.grib2',
            msg='Path to file not passed to file loader'
        )
        self.assertEqual(
            self.subject._load_wind,
            self.file_loader_mock.load.call_args[1]['load_wind'],
            msg='Var map not passed to file loader'
        )
        self.assertRegex(
            self.file_loader_mock.load.call_args[1]['sixth_hour_file'],
            r'.*/hrrr.20180721/hrrr.t19z.wrfsfcf06.grib2',
            msg='Path to file not passed to file loader'
        )
        self.assertEqual(
            self.file_loader_mock.load.call_args[1]['sixth_hour_variables'],
            self.SIXTH_HOUR_VARIABLE,
            msg='Path to file not passed to file loader'
        )

    def test_checks_sixth_hour_presence(self):
        first_hour = re.compile(r'.*hrrr.t00z\.wrfsfcf01\.grib2')
        sixth_hour = re.compile(r'.*hrrr.t19z\.wrfsfcf06\.grib2')
        with mock.patch('os.path.exists', return_value=True) as path_patch:
            self.subject.xarray(*self.METHOD_ARGS)

            assert any(
                first_hour.match(str(file)) for file in path_patch.call_args_list
            )
            assert any(
                sixth_hour.match(str(file)) for file in path_patch.call_args_list
            )

    def test_checks_sixth_hour_missing(self):
        with mock.patch('os.path.exists', return_value=True):
            with mock.patch.object(
                FileLoader, '_check_sixth_hour_presence', return_value=False
            ):
                with self.assertRaises(FileNotFoundError):
                    self.subject.xarray(*self.METHOD_ARGS)
