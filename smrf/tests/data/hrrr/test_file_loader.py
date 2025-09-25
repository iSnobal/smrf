import re
import unittest

from unittest import mock
import pandas as pd
import xarray

from smrf.data.hrrr.file_loader import FileLoader
from smrf.data.hrrr.grib_file_xarray import GribFileXarray


FILE_DIR = '/path/to/files'
START_DT = pd.to_datetime('2018-07-22 01:00')
END_DT = pd.to_datetime('2018-07-22 06:00')

LOGGER = mock.Mock(name='Logger')


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


def saved_data_return_values():
    metadata = mock.MagicMock()
    metadata.name = 'metadata'
    dataframe = mock.MagicMock(spec={})
    dataframe.name = 'dataframe'
    return metadata, dataframe


@mock.patch.object(FileLoader, 'get_data')
@mock.patch.object(
    FileLoader,
    'convert_to_dataframes',
    return_value=saved_data_return_values()
)
class TestFileLoaderGetSavedData(unittest.TestCase):
    BBOX = mock.Mock(name='Bounding Box')
    UTM_NUMBER = 12
    METHOD_ARGS = [START_DT, END_DT, BBOX, UTM_NUMBER]

    @classmethod
    def setUpClass(cls):
        LOGGER.info = mock.Mock()
        cls.subject = FileLoader(FILE_DIR, 1, external_logger=LOGGER)

    def test_sets_bbox(self, _get_data_patch, _df_patch):
        self.subject.get_saved_data(*self.METHOD_ARGS)

        self.assertEqual(self.BBOX, self.subject.file_loader.bbox)

    def test_call_get_data(self, _df_patch, get_data_patch):
        self.subject.get_saved_data(*self.METHOD_ARGS)

        get_data_patch.assert_called_once_with(
            START_DT, END_DT
        )

    def test_converts_df(self, df_patch, _get_data_patch):
        self.subject.get_saved_data(*self.METHOD_ARGS)

        df_patch.assert_called_once_with(self.UTM_NUMBER)

    def test_returns_metadata_and_df(self, _data_patch, _df_patch):
        metadata, dataframe = self.subject.get_saved_data(*self.METHOD_ARGS)

        self.assertEqual('metadata', metadata.name)
        self.assertEqual('dataframe', dataframe.name)


class TestFileLoaderGetData(TestFileLoader):
    def setUp(self):
        super().setUp()

        LOGGER.debug = mock.Mock()
        LOGGER.error = mock.Mock()

        file_loader = mock.MagicMock(spec=GribFileXarray)
        file_loader.name = "Mock GRIBXARRAY Loader"
        file_loader.SUFFIX = GribFileXarray.SUFFIX
        file_loader.load.return_value = [mock.Mock("HRRR data")]

        self.subject._file_loader = file_loader

    @mock.patch('xarray.combine_by_coords')
    def test_load_attempts_per_timestamp(self, _xarray_patch):
        with mock.patch('os.path.exists', return_value=True):
            self.subject.get_data(START_DT, END_DT)

        self.assertEqual(
            6,
            self.subject.file_loader.load.call_count,
            msg='More data was loaded than requested dates'
        )
        # Check file path of last loaded file
        self.assertRegex(
            self.subject.file_loader.load.call_args[1]['file'],
            r'.*/hrrr.20180722/hrrr.t05z.wrfsfcf01.grib2',
            msg='Path to file not passed to file loader'
        )
        self.assertEqual(
            self.subject._load_wind,
            self.subject.file_loader.load.call_args[1]['load_wind'],
            msg='Parameter: load_wind not passed to file loader'
        )

    @mock.patch('xarray.combine_by_coords')
    def test_sets_data_attribute(self, xarray_patch):
        xarray_patch.return_value = xarray.Dataset()

        with mock.patch('os.path.exists', return_value=True):
            self.subject.get_data(START_DT, END_DT)

        self.assertIsInstance(self.subject.data, xarray.Dataset)

    def test_can_not_load_first_forecast_hour(self):
        data_exception = Exception('Data loading error')
        self.subject.file_loader.load.side_effect = data_exception

        with mock.patch('os.path.exists', return_value=True):
            with self.assertRaisesRegex(
                type(data_exception), print(data_exception)
            ):
                self.subject.get_data(START_DT, END_DT)

            self.assertEqual(
                1,
                self.subject.file_loader.load.call_count,
                msg='Tried to load more than one forecast hours for a '
                    'single time step'
            )

    def test_file_not_found(self):
        with mock.patch('os.path.exists', return_value=False):
            with self.assertRaises(FileNotFoundError):
                self.subject.get_data(START_DT, END_DT)

        self.assertEqual(
            0,
            self.subject.file_loader.load.call_count,
            msg='Tried to load data from file although not present on disk'
        )

    @mock.patch('xarray.combine_by_coords')
    def test_failed_combine_coords(self, xarray_patch):
        combine_error = Exception('Combine failed')
        xarray_patch.side_effect = combine_error

        with mock.patch('os.path.exists', return_value=True):
            with self.assertRaisesRegex(
                type(combine_error), print(combine_error)
            ):
                self.subject.get_data(START_DT, END_DT)

                self.assertFalse(
                    hasattr(self.subject, 'data'),
                    msg='Data set although failed to combine'
                )


class TestFileLoaderSixthHour(unittest.TestCase):
    SIXTH_HOUR_VARIABLE = ['precip_int']

    def setUp(self):
        self.subject = FileLoader(
            file_dir=FILE_DIR,
            forecast_hour=1,
            sixth_hour_variables=self.SIXTH_HOUR_VARIABLE,
            external_logger=LOGGER
        )

        LOGGER.debug = mock.Mock()
        LOGGER.error = mock.Mock()

        file_loader = mock.MagicMock(spec=GribFileXarray)
        file_loader.name = "Mock GRIBXARRAY Loader"
        file_loader.SUFFIX = GribFileXarray.SUFFIX
        file_loader.load.return_value = [mock.Mock("HRRR data")]

        self.subject._file_loader = file_loader

    @mock.patch('xarray.combine_by_coords')
    def test_load_attempts_per_timestamp(self, _xarray_patch):
        with mock.patch('os.path.exists', return_value=True):
            self.subject.get_data(START_DT, END_DT)

        self.assertEqual(
            6,
            self.subject.file_loader.load.call_count,
            msg='More data was loaded than requested dates'
        )
        # Check arguments of last loaded file
        self.assertRegex(
            self.subject.file_loader.load.call_args[1]['file'],
            r'.*/hrrr.20180722/hrrr.t05z.wrfsfcf01.grib2',
            msg='Path to file not passed to file loader'
        )
        self.assertEqual(
            self.subject._load_wind,
            self.subject.file_loader.load.call_args[1]['load_wind'],
            msg='Var map not passed to file loader'
        )
        self.assertRegex(
            self.subject.file_loader.load.call_args[1]['sixth_hour_file'],
            r'.*/hrrr.20180722/hrrr.t00z.wrfsfcf06.grib2',
            msg='Path to file not passed to file loader'
        )
        self.assertEqual(
            self.subject.file_loader.load.call_args[1]['sixth_hour_variables'],
            self.SIXTH_HOUR_VARIABLE,
            msg='Path to file not passed to file loader'
        )

    @mock.patch('xarray.combine_by_coords')
    def test_checks_sixth_hour_presence(self, _xarray_patch):
        first_hour = re.compile(r'.*hrrr.t01z\.wrfsfcf01\.grib2')
        sixth_hour = re.compile(r'.*hrrr.t20z\.wrfsfcf06\.grib2')
        with mock.patch('os.path.exists', return_value=True) as path_patch:
            self.subject.get_data(START_DT, END_DT)

            assert (
                any(
                    first_hour.match(str(file))
                    for file in path_patch.call_args_list
                )
            )
            assert (
                any(
                    sixth_hour.match(str(file))
                    for file in path_patch.call_args_list
                )
            )

    @mock.patch('xarray.combine_by_coords')
    def test_checks_sixth_hour_missing(self, _xarray_patch):
        with mock.patch('os.path.exists', return_value=True):
            with mock.patch.object(
                FileLoader, '_check_sixth_hour_presence', return_value=False
            ):
                with self.assertRaises(FileNotFoundError):
                    self.subject.get_data(START_DT, END_DT)
