import logging
import unittest

import mock
import xarray
import pandas as pd

from smrf.data.hrrr.file_loader import FileLoader
from smrf.data.hrrr.grib_file import GribFile

FILE_DIR = 'path/to/files/'
START_DT = pd.to_datetime('2018-07-22 01:00')
END_DT = pd.to_datetime('2018-07-22 06:00')

LOGGER = mock.Mock(name='Logger')

class TestFileLoader(unittest.TestCase):
    def setUp(self):
        self.subject = FileLoader(FILE_DIR, external_logger=LOGGER)

    def test_file_dir_property(self):
        self.assertEqual(self.subject.file_dir, FILE_DIR)

    def test_defaults_to_grib2(self):
        self.assertIsInstance(self.subject.file_loader, GribFile)
        self.assertEqual(GribFile.SUFFIX, self.subject.file_type)

    def test_change_file_dir(self):
        NEW_DIR = 'somewhere/else'
        self.subject.file_dir = NEW_DIR
        self.assertEqual(NEW_DIR, self.subject.file_dir)

    def test_logger_to_file_loader(self):
        self.assertEqual(LOGGER, self.subject.log)

    def test_default_to_wind_load_false(self):
        self.assertTrue(GribFile.WIND_V not in self.subject._var_map)
        self.assertTrue(GribFile.WIND_U not in self.subject._var_map)

    def test_can_load_wind_data(self):
        self.subject = FileLoader(FILE_DIR, load_wind=True)
        self.assertTrue(GribFile.WIND_V in self.subject._var_map)
        self.assertTrue(GribFile.WIND_U in self.subject._var_map)


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
        cls.subject = FileLoader(FILE_DIR, external_logger=LOGGER)

    def test_sets_bbox(self, get_data_patch, _df_patch):
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


class TestFileLoaderGetData(unittest.TestCase):
    def setUp(self):
        super().setUp()

        LOGGER.debug = mock.Mock()
        LOGGER.error = mock.Mock()

        file_loader = mock.MagicMock(spec=GribFile)
        file_loader.name = 'Mock GRIB Loader'
        file_loader.SUFFIX = GribFile.SUFFIX

        self.subject = FileLoader(
            file_dir=FILE_DIR, external_logger=LOGGER
        )
        self.subject._file_loader = file_loader

    def test_load_attempts_per_timestamp(self):
        with mock.patch('os.path.exists', return_value=True):
            self.subject.get_data(START_DT, END_DT)

        self.assertEqual(
            6,
            self.subject.file_loader.load.call_count,
            msg='More data was loaded than requested forecast hours'
        )
        self.assertRegex(
            self.subject.file_loader.load.call_args.args[0],
            r'.*/hrrr.20180722/hrrr.t05z.wrfsfcf01.grib2',
            msg='Path to file not passed to file loader'
        )
        self.assertEqual(
            self.subject._var_map,
            self.subject.file_loader.load.call_args.args[1],
            msg='Var map not passed to file loader'
        )

    def test_tries_six_forecast_hours(self):
        self.subject.file_loader.load.side_effect = Exception('Data error')
        with mock.patch('os.path.exists', return_value=True):
            with self.assertRaisesRegex(IOError, 'Not able to find good file'):
                self.subject.get_data(START_DT, END_DT)

            self.assertEqual(
                6,
                self.subject.file_loader.load.call_count,
                msg='Tried to load more than six forecast hours for a '
                    'single time step'
            )

    def test_file_not_found(self):
        with mock.patch('os.path.exists', return_value=False):
            with self.assertRaises(IOError):
                self.subject.get_data(START_DT, END_DT)

        self.assertEqual(
            0,
            self.subject.file_loader.load.call_count,
            msg='Tried to load data from file although not present on disk'
        )

    def test_with_loading_error(self):
        self.subject.file_loader.load.side_effect = Exception('Data error')

        with mock.patch('os.path.exists', side_effect=[True]):
            with self.assertRaises(IOError):
                self.subject.get_data(START_DT, END_DT)

        # Can't load the file on disk and the other forecast hours are missing
        self.assertEqual(
            1,
            self.subject.file_loader.load.call_count,
            msg='Tried to find more files than present on disk'
        )

    def test_sets_data_attribute(self):
        self.subject.file_loader.load.return_value = []
        with mock.patch('os.path.exists', return_value=True):
            with mock.patch('xarray.combine_by_coords') as xr_patch:
                xr_patch.return_value = xarray.Dataset()
                self.subject.get_data(START_DT, END_DT)
        self.assertIsInstance(self.subject.data, xarray.Dataset)

    def test_failed_combine_coords(self):
        with mock.patch('os.path.exists', return_value=True):
            with mock.patch('xarray.combine_by_coords') as xr_patch:
                xr_patch.side_effect = Exception('Combine failed')
                self.subject.get_data(START_DT, END_DT)

                self.assertFalse(
                    hasattr(self.subject, 'data'),
                    msg='Data set although failed to combine'
                )
