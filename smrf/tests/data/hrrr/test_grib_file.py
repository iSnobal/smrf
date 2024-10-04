import unittest
import mock

from smrf.data.hrrr.grib_file import GribFile

BBOX = mock.Mock(name='Bounding Box')
LOGGER = mock.Mock(name='Logger')


class TestGribFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.subject = GribFile(LOGGER)

    def test_log_name(self):
        self.assertEqual(
            LOGGER,
            self.subject.log,
        )

    def test_file_suffix(self):
        self.assertEqual('grib2', GribFile.SUFFIX)

    def test_variable_map(self):
        self.assertEqual(
            GribFile.VAR_MAP,
            self.subject.variable_map
        )

    def test_cell_size(self):
        self.assertEqual(3000, GribFile.CELL_SIZE)

    def test_variables(self):
        self.assertEqual(
            GribFile.VAR_MAP.keys(),
            GribFile.VARIABLES
        )
