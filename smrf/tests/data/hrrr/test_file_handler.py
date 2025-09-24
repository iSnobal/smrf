import unittest

import pandas as pd

from smrf.data.hrrr import FileHandler


class TestFileHandler(unittest.TestCase):
    def test_file_date(self):
        file_time = pd.to_datetime('2018-02-08 05:00')

        forecast_hour = 1
        day, file_hour = FileHandler.file_date(
            file_time, forecast_hour
        )
        self.assertEqual('2018-02-08', str(day))
        self.assertEqual(4, file_hour)

        forecast_hour = 3
        day, file_hour = FileHandler.file_date(
            file_time, forecast_hour
        )
        self.assertEqual('2018-02-08', str(day))
        self.assertEqual(2, file_hour)

        forecast_hour = 8
        day, file_hour = FileHandler.file_date(
            file_time, forecast_hour
        )
        self.assertEqual('2018-02-07', str(day))
        self.assertEqual(21, file_hour)

    def test_file_name_grib(self):
        self.assertEqual(
            'hrrr.t04z.wrfsfcf01.grib2',
            FileHandler.file_name(4, 1, 'grib2')
        )

    def test_folder_name(self):
        self.assertEqual(
            'hrrr.20180208',
            FileHandler.folder_name(pd.to_datetime('2018-02-08'))
        )

    def test_folder_and_file(self):
        folder_name, file_name = FileHandler.folder_and_file(
            pd.to_datetime('2018-02-08 05:00'), 1, 'grib2'
        )

        self.assertEqual('hrrr.20180208', folder_name)
        self.assertEqual('hrrr.t04z.wrfsfcf01.grib2', file_name)
