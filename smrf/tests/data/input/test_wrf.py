import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

import netCDF4 as nc
import numpy as np
import pandas as pd
import utm

from smrf.data.input import InputWRF
from smrf.data.input.netcdf import metadata_name_from_index
from smrf.data.load_topo import Topo

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


def write_wrf_file(
    path: str, time_stamps: list[str] = TIME_STAMPS, string_times: bool = False
) -> None:
    """
    Write a small WRF-style netCDF file on the grid defined above.

    Args:
        path: location of the netCDF file to create
        time_stamps: times in "YYYY-MM-DD HH:MM:SS" form
        string_times: store Times as a netCDF string instead of a char array
    """
    wrf_times = [s.replace(" ", "_") for s in time_stamps]
    with nc.Dataset(path, "w") as f:
        f.createDimension("Time", None)
        f.createDimension("DateStrLen", 19)
        f.createDimension("south_north", XLAT.shape[0])
        f.createDimension("west_east", XLAT.shape[1])

        if string_times:
            t = f.createVariable("Times", str, ("Time",))
            for i, s in enumerate(wrf_times):
                t[i] = s
        else:
            t = f.createVariable("Times", "S1", ("Time", "DateStrLen"))
            t[:] = np.array([list(s) for s in wrf_times], dtype="S1")

        for name, data in {"XLAT": XLAT, "XLONG": XLONG, "HGT": HGT}.items():
            v = f.createVariable(name, "f4", ("south_north", "west_east"))
            v[:] = data


class TestInputWRF(unittest.TestCase):

    def setUp(self):
        tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp_dir)
        self.file_path = f"{tmp_dir}/wrf.nc"
        write_wrf_file(self.file_path)

    def build_input(self, start_date=START_DATE):
        config = {"gridded": {"wrf_file": self.file_path}}
        topo = MagicMock(spec=Topo)
        topo.zone_number = ZONE_NUMBER
        wrf = InputWRF(start_date, END_DATE, BBOX, topo, config)
        wrf.file = nc.Dataset(self.file_path)
        self.addCleanup(wrf.file.close)
        return wrf

    def test_get_file_times_char_array(self):
        wrf = self.build_input()

        times = wrf.get_file_times()

        expected = pd.to_datetime(TIME_STAMPS, utc=True)
        pd.testing.assert_index_equal(times, expected)

    def test_get_file_times_string_type(self):
        write_wrf_file(self.file_path, string_times=True)
        wrf = self.build_input()

        times = wrf.get_file_times()

        expected = pd.to_datetime(TIME_STAMPS, utc=True)
        pd.testing.assert_index_equal(times, expected)

    def test_get_file_times_naive_when_not_utc(self):
        wrf = self.build_input(start_date=NAIVE_START_DATE)

        times = wrf.get_file_times()

        self.assertIsNone(times.tz)

    def test_get_metadata_selects_cells_in_bbox(self):
        wrf = self.build_input()

        wrf.get_metadata()

        name = metadata_name_from_index(CENTER)
        np.testing.assert_array_equal(wrf.station_index, [CENTER])
        self.assertEqual(wrf.primary_id, [name])

        row = wrf.metadata.loc[name]
        self.assertAlmostEqual(row["latitude"], CENTER_LAT)
        self.assertAlmostEqual(row["longitude"], CENTER_LON)
        self.assertAlmostEqual(row["elevation"], ELEVATION)

        expected_x, expected_y = utm.from_latlon(
            CENTER_LAT, CENTER_LON, force_zone_number=ZONE_NUMBER
        )[:2]
        self.assertAlmostEqual(row["utm_x"], expected_x, places=UTM_PLACES)
        self.assertAlmostEqual(row["utm_y"], expected_y, places=UTM_PLACES)
