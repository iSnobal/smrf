import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import netCDF4 as nc
import numpy as np
import pandas as pd
import utm

from smrf.data.input import InputWRF
from smrf.data.input.netcdf import metadata_name_from_index
from smrf.data.load_topo import Topo
from smrf.tests.smrf_test_case_lakes import SMRFTestCaseLakes

ZONE_NUMBER = 11
UTM_PLACES = 3
ELEVATION = 1500.0

LATS = np.array([43.25, 43.5, 43.75])
LONS = np.array([-109.75, -109.5, -109.25])
SPACING = LATS[1] - LATS[0]
XLAT, XLONG = np.meshgrid(LATS, LONS, indexing="ij")
HGT = np.full(XLAT.shape, ELEVATION)

CENTER = (1, 1)
CENTER_LAT = LATS[CENTER[0]]
CENTER_LON = LONS[CENTER[1]]
BBOX = [
    CENTER_LON - SPACING / 2,
    CENTER_LAT - SPACING / 2,
    CENTER_LON + SPACING / 2,
    CENTER_LAT + SPACING / 2,
]

START_DATE = pd.to_datetime("2015-03-03 00:00 UTC")
NAIVE_START_DATE = pd.to_datetime("2015-03-03 00:00")
END_DATE = pd.to_datetime("2015-03-03 04:00 UTC")
TIME_STAMPS = ["2015-03-03 00:00:00", "2015-03-03 01:00:00"]

LAKES_DIR = Path(__file__).resolve().parents[2] / "basins" / "Lakes" / "WUS-D3"


def write_wrf_file(
    path: str, time_stamps: list[str] = TIME_STAMPS, width: int | None = None
) -> None:
    """
    Write a small WRF-style netCDF file on the grid defined above.

    Args:
        path: location of the netCDF file to create
        time_stamps: times in "YYYY-MM-DD HH:MM:SS" form
        width: size of each Times entry in bytes. Each timestamp is padded
            with null bytes up to this size, as the RME file does. Defaults to
            the timestamp length, so no padding is added.
    """
    wrf_times = [s.replace(" ", "_") for s in time_stamps]
    width = width or max(len(s) for s in wrf_times)
    wrf_times = [s.ljust(width, "\x00") for s in wrf_times]
    with nc.Dataset(path, "w") as f:
        f.createDimension("Time", None)
        f.createDimension("DateStrLen", width)
        f.createDimension("south_north", XLAT.shape[0])
        f.createDimension("west_east", XLAT.shape[1])

        t = f.createVariable("Times", "S1", ("Time", "DateStrLen"))
        t[:] = np.array([list(s) for s in wrf_times], dtype="S1")

        for name, data in {"XLAT": XLAT, "XLONG": XLONG, "HGT": HGT}.items():
            v = f.createVariable(name, "f4", ("south_north", "west_east"))
            v[:] = data


def make_topo_mock() -> MagicMock:
    """
    Create a mocked Topo
    """
    topo = MagicMock(spec=Topo)
    topo.zone_number = ZONE_NUMBER
    return topo


def make_wrf_input(file_path: str, start_date: pd.Timestamp = START_DATE) -> InputWRF:
    """
    Create InputWRF test instance and open the netCDF dataset.

    The caller is responsible for closing `wrf.file`.

    Args:
        file_path: path to the WRF netCDF file
        start_date: start of the time range used by the input object

    Returns:
        InputWRF with `file` set, but no metadata or data loaded yet
    """
    config = {"gridded": {"wrf_file": file_path}}
    wrf = InputWRF(start_date, END_DATE, BBOX, make_topo_mock(), config)
    wrf.file = nc.Dataset(file_path)
    return wrf


class WRFTestCase(unittest.TestCase):
    """
    Base class that writes one WRF file per test class and opens it per test.

    Subclasses set `file_name`, `time_stamps` and `width` to choose the
    file's contents. The file is created in setUpClass and removed in
    tearDownClass; each test gets a fresh `self.wrf` that is closed afterward.
    """

    file_name = "wrf.nc"
    time_stamps = TIME_STAMPS
    width = None

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp()
        cls.file_path = f"{cls.tmp_dir}/{cls.file_name}"
        write_wrf_file(cls.file_path, cls.time_stamps, cls.width)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir)

    def setUp(self):
        self.wrf = make_wrf_input(self.file_path)

    def tearDown(self):
        self.wrf.file.close()


class TestInputWRF(WRFTestCase):
    """
    Tests for InputWRF using a standard WRF output file.
    """

    def test_get_file_times_char_array(self):
        times = self.wrf.get_file_times()

        expected = pd.to_datetime(TIME_STAMPS, utc=True)
        pd.testing.assert_index_equal(times, expected)

    def test_get_file_times_naive_when_not_utc(self):
        self.wrf.start_date = NAIVE_START_DATE

        times = self.wrf.get_file_times()

        self.assertIsNone(times.tz)

    def test_get_metadata_selects_cells_in_bbox(self):
        self.wrf.get_metadata()

        name = metadata_name_from_index(CENTER)
        np.testing.assert_array_equal(self.wrf.station_index, [CENTER])
        self.assertEqual(self.wrf.primary_id, [name])

        row = self.wrf.metadata.loc[name]
        self.assertAlmostEqual(row["latitude"], CENTER_LAT)
        self.assertAlmostEqual(row["longitude"], CENTER_LON)
        self.assertAlmostEqual(row["elevation"], ELEVATION)

        expected_x, expected_y = utm.from_latlon(
            CENTER_LAT, CENTER_LON, force_zone_number=ZONE_NUMBER
        )[:2]
        self.assertAlmostEqual(row["utm_x"], expected_x, places=UTM_PLACES)
        self.assertAlmostEqual(row["utm_y"], expected_y, places=UTM_PLACES)


class TestInputWRF_WUS_D3(WRFTestCase):
    """
    Tests for InputWRF using a WUS D3 output file.
    """

    file_name = "wus_d3.nc"
    width = 64

    def test_get_file_times_padded(self):
        times = self.wrf.get_file_times()

        expected = pd.to_datetime(TIME_STAMPS, utc=True)
        pd.testing.assert_index_equal(times, expected)

    def test_get_metadata_reads_static_file(self):
        static_path = LAKES_DIR / "WUS-D3_domain.nc"
        config = {
            "gridded": {
                "wrf_file": "not_opened_wrf_file.nc",
                "wrf_metadata": str(static_path),
            }
        }
        wrf = InputWRF(
            START_DATE, END_DATE, SMRFTestCaseLakes.BBOX, make_topo_mock(), config
        )

        wrf.get_metadata()

        with nc.Dataset(static_path) as f:
            xlat = f.variables["XLAT"][:]
            xlong = f.variables["XLONG"][:]
            hgt = f.variables["HGT"][:]
        mask = (
            (xlat >= SMRFTestCaseLakes.BBOX[1])
            & (xlat <= SMRFTestCaseLakes.BBOX[3])
            & (xlong >= SMRFTestCaseLakes.BBOX[0])
            & (xlong <= SMRFTestCaseLakes.BBOX[2])
        )
        np.testing.assert_array_equal(wrf.station_index, np.argwhere(mask))
        np.testing.assert_allclose(wrf.metadata["elevation"].values, hgt[mask])

    def test_get_metadata_raises_without_metadata_file(self):
        # Gracefully handle the case when a user does not set the wrf_metadata in
        # the config file although WUS-D3 is used
        hourly_path = LAKES_DIR / "auxhist_d01_2022-11-08_00:00:00.nc"
        config = {"gridded": {"wrf_file": str(hourly_path)}}
        wrf = InputWRF(
            START_DATE, END_DATE, SMRFTestCaseLakes.BBOX, make_topo_mock(), config
        )
        wrf.file = nc.Dataset(hourly_path)
        self.addCleanup(wrf.file.close)

        with self.assertRaisesRegex(ValueError, r"XLAT.*wrf_metadata .ini file option"):
            wrf.get_metadata()
