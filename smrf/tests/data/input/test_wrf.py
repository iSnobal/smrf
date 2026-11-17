import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import netCDF4 as nc
import numpy as np
import pandas as pd
import utm

from smrf.data.helper import grid_station_names
from smrf.data.input import InputWRF
from smrf.data.load_topo import Topo
from smrf.tests.smrf_test_case_lakes import SMRFTestCaseLakes

# Synthetic grid: a 3 x 3 grid whose center cell is inside BBOX
ZONE_NUMBER = 11
UTM_PLACES = 3
ELEVATION = 1500.0
HEIGHT_LEVEL = 2

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

# Synthetic times
FILE_PREFIX = "wrf"
START_DATE = pd.to_datetime("2015-03-03 00:00 UTC")
HOUR_01 = pd.to_datetime("2015-03-03 01:00 UTC")
HOUR_02 = pd.to_datetime("2015-03-03 02:00 UTC")
END_DATE = pd.to_datetime("2015-03-03 04:00 UTC")
NAIVE_START_DATE = pd.to_datetime("2015-03-03 00:00")
TIME_STAMPS = ["2015-03-03 00:00:00", "2015-03-03 01:00:00"]

# Lakes sample data (WUS-D3)
LAKES_DIR = SMRFTestCaseLakes.basin_dir / "WUS-D3"
LAKES_PREFIX = "auxhist_d01"
LAKES_DOMAIN = LAKES_DIR / "WUS-D3_domain.nc"
LAKES_BBOX = SMRFTestCaseLakes.BBOX
LAKES_START = pd.to_datetime("2022-11-08 00:00 UTC")
LAKES_01 = pd.to_datetime("2022-11-08 01:00 UTC")
LAKES_END = pd.to_datetime("2022-11-08 02:00 UTC")
LAKES_FUTURE = pd.to_datetime("2030-01-01 00:00 UTC")


TOPO = MagicMock(spec=Topo)
TOPO.zone_number = ZONE_NUMBER


def lakes_file(hour: str) -> Path:
    """
    Get the path of a Lakes hourly file for 2022-11-08.

    Parameters
    ----------
    hour : str
        Two-digit hour, such as "00"

    Returns
    -------
    Path
        Path of the file in the repo
    """
    return LAKES_DIR / f"{LAKES_PREFIX}_2022-11-08_{hour}:00:00.nc"


def make_config(path, prefix=None, metadata=None) -> dict:
    """
    Build a gridded config for InputWRF.

    Parameters
    ----------
    path : str or Path
        Value of wrf_path
    prefix : str, optional
        Value of wrf_prefix
    metadata : str or Path, optional
        Value of wrf_metadata

    Returns
    -------
    dict
        Config with a gridded section
    """
    gridded = {InputWRF.ConfigKey.PATH: str(path)}
    if prefix is not None:
        gridded[InputWRF.ConfigKey.PREFIX] = prefix
    if metadata is not None:
        gridded[InputWRF.ConfigKey.METADATA] = str(metadata)
    return {"gridded": gridded}


def write_wrf_file(
    path: str, time_stamps: list[str] = TIME_STAMPS, width: int | None = None
) -> None:
    """
    Write a small WRF-style netCDF file on the synthetic grid.

    Parameters
    ----------
    path : str
        Location of the netCDF file to create
    time_stamps : list of str
        Times in "YYYY-MM-DD HH:MM:SS" form
    width : int, optional
        Size of each Times entry in bytes. Each timestamp is padded with null
        bytes up to this size, as the RME file does. Defaults to the timestamp
        length.
    """
    wrf_times = [s.replace(" ", "_") for s in time_stamps]
    width = width or max(len(s) for s in wrf_times)
    wrf_times = [s.ljust(width, "\x00") for s in wrf_times]
    with nc.Dataset(path, "w") as f:
        f.createDimension("Time", None)
        f.createDimension("DateStrLen", width)
        f.createDimension("south_north", XLAT.shape[0])
        f.createDimension("west_east", XLAT.shape[1])
        f.createDimension("bottom_top", HEIGHT_LEVEL)

        t = f.createVariable("Times", "S1", ("Time", "DateStrLen"))
        t[:] = np.array([list(s) for s in wrf_times], dtype="S1")

        for name, data in {"XLAT": XLAT, "XLONG": XLONG, "HGT": HGT}.items():
            v = f.createVariable(name, "f4", ("south_north", "west_east"))
            v[:] = data

        n_times = len(wrf_times)
        forcing = {"T2": 275.15, "DWPT": 270.0, "UGRD": 0.0, "VGRD": -1.0}
        for name, value in forcing.items():
            v = f.createVariable(name, "f4", ("Time", "south_north", "west_east"))
            v[:] = np.full((n_times, *XLAT.shape), value)

        rain = f.createVariable("RAINNC", "f4", ("Time", "south_north", "west_east"))
        rain[:] = np.broadcast_to(
            np.arange(n_times)[:, None, None], (n_times, *XLAT.shape)
        )

        cloud = f.createVariable(
            "CLDFRA", "f4", ("Time", "bottom_top", "south_north", "west_east")
        )
        cloud[:] = np.full((n_times, HEIGHT_LEVEL, *XLAT.shape), 0.5)


def make_lakes_copy(hour: str, dst_dir: str | Path) -> Path:
    """
    Copy a Lakes hourly file and add the fields the loader needs.

    The repo's WUS-D3 hourly files lack DWPT, UGRD, VGRD and CLDFRA, which the
    loader reads for every file. The copies add these fields artificially, so
    they test the loader's logic, not the physics: UGRD and VGRD copy U10 and
    V10, DWPT is derived from Q2 and PSFC, and CLDFRA is a constant 0.5. The
    repo data is not changed.

    Parameters
    ----------
    hour : str
        Two-digit hour of the source file, such as "00"
    dst_dir : str or Path
        Folder to write the copy into, with the same file name

    Returns
    -------
    Path
        Path of the copy
    """
    src = lakes_file(hour)
    dst = Path(dst_dir, src.name)
    shutil.copy(src, dst)
    with nc.Dataset(dst, "r+") as f:
        q2 = f.variables["Q2"][:]
        p_hpa = f.variables["PSFC"][:] / 100.0
        vapor_hpa = q2 * p_hpa / (0.622 + q2)
        ln_e = np.log(vapor_hpa / 6.112)
        dwpt = 243.5 * ln_e / (17.67 - ln_e)
        dims = ("Time", "south_north", "west_east")
        f.createVariable("DWPT", "f4", dims)[:] = dwpt
        f.createVariable("UGRD", "f4", dims)[:] = f.variables["U10"][:]
        f.createVariable("VGRD", "f4", dims)[:] = f.variables["V10"][:]
        f.createDimension("bottom_top", 1)
        cloud_dims = ("Time", "bottom_top", "south_north", "west_east")
        f.createVariable("CLDFRA", "f4", cloud_dims)[:] = 0.5
    return dst


class WRFTestCase(unittest.TestCase):
    """
    Base class that writes one synthetic WRF file per test class.

    Subclasses set `file_name`, `time_stamps` and `width` to choose the file's
    contents. Each test gets a fresh `self.wrf`, which is closed afterward.
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
        self.wrf = InputWRF(
            START_DATE, END_DATE, BBOX, TOPO, make_config(self.file_path)
        )
        self.wrf.file = nc.Dataset(self.file_path)

    def tearDown(self):
        self.wrf.file.close()

    def new_folder(self) -> str:
        """
        Create an empty temporary folder, removed after the test.

        Returns
        -------
        str
            Path of the folder
        """
        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder)
        return folder

    def test_load(self):
        wrf = InputWRF(START_DATE, HOUR_01, BBOX, TOPO, make_config(self.file_path))

        wrf.load()

        name = grid_station_names([CENTER[0]], [CENTER[1]])[0]
        self.assertEqual(wrf.air_temp.columns.tolist(), [name])
        np.testing.assert_allclose(wrf.air_temp[name], 2.0, atol=1e-3)
        np.testing.assert_allclose(wrf.precip[name], [0.0, 1.0], atol=1e-3)
        np.testing.assert_allclose(wrf.cloud_factor[name], 0.5, atol=1e-3)
        np.testing.assert_allclose(wrf.wind_speed[name], 1.0, atol=1e-3)
        np.testing.assert_allclose(wrf.wind_direction[name], 270.0, atol=1e-3)


class TestInputWRF(WRFTestCase):
    """
    Tests for InputWRF using a synthetic WRF file and folder.
    """

    def make_hourly_folder(self, hours: list[str]) -> str:
        """
        Write one synthetic hourly WRF file per hour into a new folder.

        Parameters
        ----------
        hours : list of str
            Two-digit hours to write, such as ["00", "02"]

        Returns
        -------
        str
            Path of the folder
        """
        folder = self.new_folder()
        for hour in hours:
            path = Path(folder, f"{FILE_PREFIX}_2015-03-03_{hour}:00:00.nc")
            write_wrf_file(str(path), [f"2015-03-03 {hour}:00:00"])
        return folder

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

        name = grid_station_names([CENTER[0]], [CENTER[1]])[0]
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

    def test_load_file_single_hour(self):
        path = make_lakes_copy("00", self.new_folder())
        config = make_config(path, metadata=LAKES_DOMAIN)
        wrf = InputWRF(LAKES_START, LAKES_START, LAKES_BBOX, TOPO, config)

        wrf.load()

        self.assertEqual(wrf.air_temp.index.tolist(), [LAKES_START])
        np.testing.assert_allclose(wrf.precip.values, 0.0)
        self.assertIsNone(wrf.file)
        self.assertIsNone(wrf.hours)

    def test_load_directory_raises_when_hour_missing(self):
        folder = self.make_hourly_folder(["00", "02"])
        wrf = InputWRF(
            START_DATE, HOUR_02, BBOX, TOPO, make_config(folder, FILE_PREFIX)
        )

        with self.assertRaisesRegex(ValueError, "no time step"):
            wrf.load()

    def test_load_directory_logs_missing_hours(self):
        folder = self.make_hourly_folder(["00"])
        wrf = InputWRF(
            START_DATE, HOUR_02, BBOX, TOPO, make_config(folder, FILE_PREFIX)
        )

        with self.assertLogs(wrf._logger, level="WARNING") as logs:
            with self.assertRaisesRegex(ValueError, "no time step"):
                wrf.load()

        self.assertEqual(len(logs.output), 2)
        self.assertIn("No WRF file for 2015-03-03 01:00:00", logs.output[0])
        self.assertIn("No WRF file for 2015-03-03 02:00:00", logs.output[1])

    def test_load_directory_raises_when_file_times_do_not_match(self):
        folder = self.new_folder()
        write_wrf_file(
            str(Path(folder, f"{FILE_PREFIX}_2015-03-03_01:00:00.nc")),
            ["2015-03-03 00:00:00"],
        )
        wrf = InputWRF(HOUR_01, HOUR_01, BBOX, TOPO, make_config(folder, FILE_PREFIX))

        with self.assertRaisesRegex(ValueError, "no time step"):
            wrf.load()


    def test_load_file_raises_when_hour_missing(self):
        wrf = InputWRF(
            START_DATE, HOUR_02, BBOX, TOPO, make_config(self.file_path)
        )

        with self.assertRaisesRegex(ValueError, "no time step"):
            wrf.load()


class TestInputWRF_WUS_D3(WRFTestCase):
    """
    Tests for InputWRF using the WUS-D3 Lakes data.
    """

    file_name = "wus_d3.nc"
    width = 64

    def test_get_file_times_padded(self):
        times = self.wrf.get_file_times()

        expected = pd.to_datetime(TIME_STAMPS, utc=True)
        pd.testing.assert_index_equal(times, expected)

    def test_get_metadata_reads_static_file(self):
        config = make_config("not_opened_wrf_file.nc", metadata=LAKES_DOMAIN)
        wrf = InputWRF(START_DATE, END_DATE, LAKES_BBOX, TOPO, config)

        wrf.get_metadata()

        with nc.Dataset(LAKES_DOMAIN) as f:
            xlat = f.variables["XLAT"][:]
            xlong = f.variables["XLONG"][:]
            hgt = f.variables["HGT"][:]
        mask = (
            (xlat >= LAKES_BBOX[1])
            & (xlat <= LAKES_BBOX[3])
            & (xlong >= LAKES_BBOX[0])
            & (xlong <= LAKES_BBOX[2])
        )
        np.testing.assert_array_equal(wrf.station_index, np.argwhere(mask))
        np.testing.assert_allclose(wrf.metadata["elevation"].values, hgt[mask])

    def test_get_metadata_raises_without_metadata_file(self):
        # The hourly file has no grid fields, so the loader needs wrf_metadata
        wrf = InputWRF(
            START_DATE, END_DATE, LAKES_BBOX, TOPO, make_config(lakes_file("00"))
        )
        wrf.file = nc.Dataset(lakes_file("00"))
        self.addCleanup(wrf.file.close)

        with self.assertRaisesRegex(ValueError, r"XLAT.*wrf_metadata .ini file option"):
            wrf.get_metadata()

    def test_load_directory(self):
        folder = self.new_folder()
        for hour in ["00", "01", "02"]:
            make_lakes_copy(hour, folder)
        config = make_config(folder, LAKES_PREFIX, LAKES_DOMAIN)
        wrf = InputWRF(LAKES_START, LAKES_END, LAKES_BBOX, TOPO, config)

        wrf.load()

        hours = pd.date_range(LAKES_START, LAKES_END, freq="h")
        self.assertEqual(wrf.air_temp.index.tolist(), hours.tolist())
        rows, cols = wrf.station_index.T
        rainnc = []
        for hour in hours:
            with nc.Dataset(lakes_file(f"{hour:%H}")) as f:
                t2 = f.variables["T2"][0]
                rainnc.append(f.variables["RAINNC"][0][rows, cols])
            np.testing.assert_allclose(
                wrf.air_temp.loc[hour].values, t2[rows, cols] - 273.15, atol=1e-3
            )

        rainnc = np.array(rainnc)
        expected_precip = np.zeros_like(rainnc)
        expected_precip[1:] = np.diff(rainnc, axis=0)
        np.testing.assert_allclose(wrf.precip.values, expected_precip, atol=1e-3)
        self.assertIsNone(wrf.file)
        self.assertIsNone(wrf.hours)

    def test_file_for_time_returns_existing_file(self):
        wrf = InputWRF(
            LAKES_START, LAKES_START, BBOX, TOPO, make_config(LAKES_DIR, LAKES_PREFIX)
        )

        path = wrf._file_for_time(LAKES_01)

        self.assertEqual(path, str(lakes_file("01")))

    def test_file_for_time_returns_none_when_missing(self):
        wrf = InputWRF(
            LAKES_START, LAKES_START, BBOX, TOPO, make_config(LAKES_DIR, LAKES_PREFIX)
        )

        self.assertIsNone(wrf._file_for_time(LAKES_FUTURE))

    def test_load_directory_raises_without_metadata_file(self):
        config = make_config(LAKES_DIR, LAKES_PREFIX)
        wrf = InputWRF(LAKES_START, LAKES_01, LAKES_BBOX, TOPO, config)

        with self.assertRaisesRegex(ValueError, r"XLAT.*wrf_metadata .ini file option"):
            wrf.load()

    def test_load_directory_requires_prefix(self):
        wrf = InputWRF(
            LAKES_START, LAKES_START, LAKES_BBOX, TOPO, make_config(LAKES_DIR)
        )

        with self.assertRaisesRegex(ValueError, "wrf_prefix must be set"):
            wrf.load()

    def test_load_directory_raises_when_folder_empty(self):
        config = make_config(self.new_folder(), LAKES_PREFIX)
        wrf = InputWRF(LAKES_START, LAKES_END, LAKES_BBOX, TOPO, config)

        with self.assertRaisesRegex(ValueError, "No WRF files in"):
            wrf.load()

    def test_load_directory_raises_when_no_file_in_period(self):
        config = make_config(LAKES_DIR, LAKES_PREFIX)
        wrf = InputWRF(LAKES_FUTURE, LAKES_FUTURE, LAKES_BBOX, TOPO, config)

        with self.assertRaisesRegex(ValueError, "No WRF files in"):
            wrf.load()
