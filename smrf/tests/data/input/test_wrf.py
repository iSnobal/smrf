import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import netCDF4 as nc
import numpy as np
import pandas as pd
import utm

from smrf.data.helper import grid_station_names
from smrf.data.input import InputWRF, wrf_schema
from smrf.data.load_topo import Topo
from smrf.envphys.constants import EPS
from smrf.tests.smrf_test_case_lakes import SMRFTestCaseLakes
from smrf.tests.tolerances import UTM_PLACES, WRF_ATOL

# Synthetic grid: a 3 x 3 grid whose center cell is inside BBOX
ZONE_NUMBER = 11
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
CENTER_NAME = grid_station_names([CENTER[0]], [CENTER[1]])[0]
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
FUTURE_DATE = pd.to_datetime("2030-01-01 00:00 UTC")
NAIVE_START_DATE = pd.to_datetime("2015-03-03 00:00")
TIME_STAMPS = ["2015-03-03 00:00:00", "2015-03-03 01:00:00"]

# Lakes sample data (WUS-D3)
LAKES_DIR = SMRFTestCaseLakes.basin_dir / "WUS-D3"
LAKES_PREFIX = "auxhist_d01"
LAKES_DOMAIN = LAKES_DIR / "WUS-D3_domain.nc"
LAKES_BBOX = SMRFTestCaseLakes.BBOX
LAKES_START = pd.to_datetime("2022-11-08 00:00 UTC")
LAKES_END = pd.to_datetime("2022-11-08 02:00 UTC")


TOPO = MagicMock(spec=Topo)
TOPO.zone_number = ZONE_NUMBER

# Error raised when grid fields are missing from the data file and no wrf_metadata
MISSING_GRID_ERROR = r"XLAT.*wrf_metadata .ini file option"

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
    path: str,
    time_stamps: list[str] = TIME_STAMPS,
    width: int | None = None,
    grid: bool = True,
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
    grid : bool, optional
        Whether to write the grid fields XLAT, XLONG and HGT. Defaults to True.
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

        if grid:
            for name, data in {"XLAT": XLAT, "XLONG": XLONG, "HGT": HGT}.items():
                v = f.createVariable(name, "f4", ("south_north", "west_east"))
                v[:] = data

        n_times = len(wrf_times)
        forcing = {"T2": 275.15, "DWPT": 270.0, "UGRD": 0.0, "VGRD": -1.0}
        for name, value in forcing.items():
            v = f.createVariable(name, "f4", ("Time", "south_north", "west_east"))
            v[:] = np.full((n_times, *XLAT.shape), value)

        rain = f.createVariable("RAINNC", "f4", ("Time", "south_north", "west_east"))
        # RAINNC is the hour of each time step, so hourly precip is 1 per hour
        hours = np.array([pd.Timestamp(s).hour for s in time_stamps])
        rain[:] = np.broadcast_to(hours[:, None, None], (n_times, *XLAT.shape))

        cloud = f.createVariable(
            "CLDFRA", "f4", ("Time", "bottom_top", "south_north", "west_east")
        )
        cloud[:] = np.full((n_times, HEIGHT_LEVEL, *XLAT.shape), 0.5)


class TestInputWRFFile(unittest.TestCase):
    """
    Tests for InputWRF loading one synthetic WRF file.
    """

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp()
        cls.file_path = f"{cls.tmp_dir}/wrf.nc"
        write_wrf_file(cls.file_path)

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

    def test_get_file_times_char_array(self):
        times = self.wrf.get_file_times()

        expected = pd.to_datetime(TIME_STAMPS, utc=True)
        pd.testing.assert_index_equal(times, expected)

    def test_get_file_times_padded(self):
        folder = self.enterContext(tempfile.TemporaryDirectory())
        path = f"{folder}/padded.nc"
        write_wrf_file(path, TIME_STAMPS, width=64)
        wrf = InputWRF(START_DATE, END_DATE, BBOX, TOPO, make_config(path))
        wrf.file = nc.Dataset(path)
        self.addCleanup(wrf.file.close)

        times = wrf.get_file_times()

        expected = pd.to_datetime(TIME_STAMPS, utc=True)
        pd.testing.assert_index_equal(times, expected)

    def test_get_file_times_naive_when_not_utc(self):
        self.wrf.start_date = NAIVE_START_DATE

        times = self.wrf.get_file_times()

        self.assertIsNone(times.tz)

    def test_get_metadata_selects_cells_in_bbox(self):
        self.wrf.get_metadata()

        np.testing.assert_array_equal(self.wrf.station_index, [CENTER])
        self.assertEqual(self.wrf.primary_id, [CENTER_NAME])

        row = self.wrf.metadata.loc[CENTER_NAME]
        self.assertAlmostEqual(row["latitude"], CENTER_LAT)
        self.assertAlmostEqual(row["longitude"], CENTER_LON)
        self.assertAlmostEqual(row["elevation"], ELEVATION)

        expected_x, expected_y = utm.from_latlon(
            CENTER_LAT, CENTER_LON, force_zone_number=ZONE_NUMBER
        )[:2]
        self.assertAlmostEqual(row["utm_x"], expected_x, places=UTM_PLACES)
        self.assertAlmostEqual(row["utm_y"], expected_y, places=UTM_PLACES)

    def test_get_metadata_raises_without_grid_fields(self):
        folder = self.enterContext(tempfile.TemporaryDirectory())
        path = f"{folder}/no_grid.nc"
        write_wrf_file(path, grid=False)
        wrf = InputWRF(START_DATE, END_DATE, BBOX, TOPO, make_config(path))
        wrf.file = nc.Dataset(path)
        self.addCleanup(wrf.file.close)

        with self.assertRaisesRegex(ValueError, MISSING_GRID_ERROR):
            wrf.get_metadata()

    def test_load(self):
        wrf = InputWRF(START_DATE, HOUR_01, BBOX, TOPO, make_config(self.file_path))

        wrf.load()

        self.assertEqual(wrf.air_temp.columns.tolist(), [CENTER_NAME])
        np.testing.assert_allclose(wrf.air_temp[CENTER_NAME], 2.0, atol=WRF_ATOL)
        np.testing.assert_allclose(wrf.cloud_factor[CENTER_NAME], 0.5, atol=WRF_ATOL)
        np.testing.assert_allclose(wrf.wind_speed[CENTER_NAME], 1.0, atol=WRF_ATOL)
        np.testing.assert_allclose(wrf.wind_direction[CENTER_NAME], 270.0, atol=WRF_ATOL)

    def test_load_file_warns_when_previous_hour_missing(self):
        # The file starts at 01:00, so the 00:00 precip is missing and set to 0
        folder = self.enterContext(tempfile.TemporaryDirectory())
        path = f"{folder}/late.nc"
        write_wrf_file(path, ["2015-03-03 01:00:00", "2015-03-03 02:00:00"])
        wrf = InputWRF(HOUR_01, HOUR_02, BBOX, TOPO, make_config(path))

        with self.assertLogs(wrf._logger, level="WARNING") as logs:
            wrf.load()

        np.testing.assert_allclose(wrf.precip[CENTER_NAME], [0.0, 1.0], atol=WRF_ATOL)
        self.assertIn("Previous hour", logs.output[0])

    def test_load_file_raises_when_hour_missing(self):
        wrf = InputWRF(START_DATE, HOUR_02, BBOX, TOPO, make_config(self.file_path))

        with self.assertRaisesRegex(ValueError, "no time step"):
            wrf.load()

    def test_load_file_raises_without_grid_fields(self):
        folder = self.enterContext(tempfile.TemporaryDirectory())
        path = f"{folder}/no_grid.nc"
        write_wrf_file(path, grid=False)
        wrf = InputWRF(START_DATE, HOUR_01, BBOX, TOPO, make_config(path))

        with self.assertRaisesRegex(ValueError, MISSING_GRID_ERROR):
            wrf.load()

    def test_load_standard_wrf_ignores_rainc(self):
        folder = self.enterContext(tempfile.TemporaryDirectory())
        path = f"{folder}/wrf.nc"
        write_wrf_file(path)
        with nc.Dataset(path, "a") as f:
            conv = f.createVariable("RAINC", "f4", ("Time", "south_north", "west_east"))
            conv[:] = np.full((len(TIME_STAMPS), *XLAT.shape), 3.0)
        wrf = InputWRF(START_DATE, HOUR_01, BBOX, TOPO, make_config(path))

        wrf.load()

        np.testing.assert_allclose(wrf.precip[CENTER_NAME], [0.0, 1.0], atol=WRF_ATOL)


class TestInputWRFFolder(unittest.TestCase):
    """
    Tests for InputWRF loading a folder of synthetic hourly WRF files.
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
            Path of the folder. RAINNC in each file equals its hour, so hourly
            precip is 1 for each hour after the first.
        """
        folder = self.enterContext(tempfile.TemporaryDirectory())
        for hour in hours:
            path = Path(folder, f"{FILE_PREFIX}_2015-03-03_{hour}:00:00.nc")
            write_wrf_file(str(path), [f"2015-03-03 {hour}:00:00"])
        return folder

    def test_load_directory_checks_input_type_once(self):
        folder = self.make_hourly_folder(["00", "01", "02"])
        config = make_config(folder, FILE_PREFIX)
        wrf = InputWRF(START_DATE, HOUR_02, BBOX, TOPO, config)
        detect = wrf_schema.detect_input_type

        with patch("smrf.data.input.wrf_schema.detect_input_type", wraps=detect) as spy:
            wrf.load()

        self.assertEqual(spy.call_count, 1)
        self.assertIsNone(wrf.input_type)

    def test_load_directory_raises_and_logs_missing_hours(self):
        folder = self.make_hourly_folder(["00"])
        wrf = InputWRF(
            START_DATE, HOUR_02, BBOX, TOPO, make_config(folder, FILE_PREFIX)
        )

        with self.assertLogs(wrf._logger, level="WARNING") as logs:
            with self.assertRaisesRegex(ValueError, "no time step"):
                wrf.load()

        missing = [line for line in logs.output if "No WRF file" in line]
        self.assertTrue(any("2015-03-03 01:00:00" in line for line in missing))
        self.assertTrue(any("2015-03-03 02:00:00" in line for line in missing))

    def test_load_directory_raises_when_file_times_do_not_match(self):
        folder = self.enterContext(tempfile.TemporaryDirectory())
        write_wrf_file(
            str(Path(folder, f"{FILE_PREFIX}_2015-03-03_01:00:00.nc")),
            ["2015-03-03 00:00:00"],
        )
        wrf = InputWRF(HOUR_01, HOUR_01, BBOX, TOPO, make_config(folder, FILE_PREFIX))

        with self.assertRaisesRegex(ValueError, "no time step"):
            wrf.load()

    def test_load_directory_requires_prefix(self):
        folder = self.make_hourly_folder(["00"])
        wrf = InputWRF(START_DATE, HOUR_02, BBOX, TOPO, make_config(folder))

        with self.assertRaisesRegex(ValueError, "wrf_prefix must be set"):
            wrf.load()

    def test_load_directory_raises_when_folder_empty(self):
        folder = self.enterContext(tempfile.TemporaryDirectory())
        config = make_config(folder, FILE_PREFIX)
        wrf = InputWRF(START_DATE, HOUR_02, BBOX, TOPO, config)

        with self.assertRaisesRegex(ValueError, "No WRF files in"):
            wrf.load()

    def test_load_directory_raises_when_no_file_in_period(self):
        folder = self.make_hourly_folder(["00"])
        config = make_config(folder, FILE_PREFIX)
        wrf = InputWRF(FUTURE_DATE, FUTURE_DATE, BBOX, TOPO, config)

        with self.assertRaisesRegex(ValueError, "No WRF files in"):
            wrf.load()

    def test_load_directory_raises_without_grid_fields(self):
        folder = self.enterContext(tempfile.TemporaryDirectory())
        path = Path(folder, f"{FILE_PREFIX}_2015-03-03_00:00:00.nc")
        write_wrf_file(str(path), ["2015-03-03 00:00:00"], grid=False)
        config = make_config(folder, FILE_PREFIX)
        wrf = InputWRF(START_DATE, START_DATE, BBOX, TOPO, config)

        with self.assertRaisesRegex(ValueError, MISSING_GRID_ERROR):
            wrf.load()

    def test_load_directory_warns_when_previous_hour_missing(self):
        folder = self.make_hourly_folder(["00", "01", "02"])
        config = make_config(folder, FILE_PREFIX)
        wrf = InputWRF(START_DATE, HOUR_02, BBOX, TOPO, config)

        with self.assertLogs(wrf._logger, level="WARNING") as logs:
            wrf.load()

        self.assertTrue(any("Previous hour" in line for line in logs.output))
        # The first precip step is 0, then the file values for each later hour
        np.testing.assert_allclose(wrf.precip[CENTER_NAME], [0.0, 1.0, 1.0], atol=WRF_ATOL)

    def test_file_for_time_returns_existing_file(self):
        folder = self.make_hourly_folder(["01"])
        config = make_config(folder, FILE_PREFIX)
        wrf = InputWRF(START_DATE, HOUR_02, BBOX, TOPO, config)

        path = wrf._file_for_time(HOUR_01)

        expected = Path(folder, f"{FILE_PREFIX}_2015-03-03_01:00:00.nc")
        self.assertEqual(path, str(expected))

    def test_file_for_time_returns_none_when_missing(self):
        folder = self.make_hourly_folder(["01"])
        config = make_config(folder, FILE_PREFIX)
        wrf = InputWRF(START_DATE, HOUR_02, BBOX, TOPO, config)

        self.assertIsNone(wrf._file_for_time(HOUR_02))


class TestProcessWRF(unittest.TestCase):
    """
    Tests for the standard WRF outputs of InputWRF.process.
    """

    @classmethod
    def setUpClass(cls):
        raw = {
            "air_temp": np.array([[275.15]]),
            "precip": np.array([[1.0], [3.0]]),
            "u10": np.array([[3.0]]),
            "v10": np.array([[4.0]]),
            "dew_point": np.array([[0.0]]),
            "cloud_factor": np.array([[[0.2], [0.6]]]),
        }
        cls.wrf = InputWRF(START_DATE, START_DATE, BBOX, TOPO, make_config("wrf.nc"))
        cls.wrf.input_type = wrf_schema.WRF
        cls.wrf.hours = pd.DatetimeIndex([START_DATE])
        cls.wrf.primary_id = [CENTER_NAME]
        cls.wrf.process(raw, cls.wrf.hours)

    def test_vapor_pressure(self):
        # From the vapor pressure math at 0 C
        np.testing.assert_allclose(self.wrf.vapor_pressure[CENTER_NAME], 610.207270)

    def test_air_temp(self):
        np.testing.assert_allclose(self.wrf.air_temp[CENTER_NAME], 2.0)

    def test_wind_speed(self):
        np.testing.assert_allclose(self.wrf.wind_speed[CENTER_NAME], 5.0)

    def test_wind_direction(self):
        np.testing.assert_allclose(
            self.wrf.wind_direction[CENTER_NAME], np.degrees(np.arctan2(4.0, 3.0))
        )

    def test_precip(self):
        np.testing.assert_allclose(self.wrf.precip[CENTER_NAME], 2.0)

    def test_cloud_factor(self):
        # Cloud fraction is the mean of its two levels, 0.4, so the factor is 0.6
        np.testing.assert_allclose(self.wrf.cloud_factor[CENTER_NAME], 0.6)


class TestInputWRF_WUS_D3(unittest.TestCase):
    """
    Tests for InputWRF loading the WUS-D3 Lakes data.
    """

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

    def test_load_file_single_hour(self):
        config = make_config(lakes_file("00"), metadata=LAKES_DOMAIN)
        wrf = InputWRF(LAKES_START, LAKES_START, LAKES_BBOX, TOPO, config)

        with patch.object(InputWRF, "process") as process:
            wrf.load()

        _, time = process.call_args.args
        self.assertEqual(time.tolist(), [LAKES_START])
        self.assertIsNone(wrf.file)
        self.assertIsNone(wrf.hours)

    def test_load_directory(self):
        hours = ["00", "01", "02"]
        folder = self.lakes_folder(hours)
        previous = f"{LAKES_PREFIX}_2022-11-07_23:00:00.nc"
        shutil.copy(LAKES_DIR / previous, folder)
        config = make_config(folder, LAKES_PREFIX, LAKES_DOMAIN)
        wrf = InputWRF(LAKES_START, LAKES_END, LAKES_BBOX, TOPO, config)

        with patch.object(InputWRF, "process") as process:
            wrf.load()

        raw, time = process.call_args.args
        rows, cols = wrf.station_index.T
        self.assertEqual(
            set(raw),
            {
                "air_temp",
                "humidity",
                "surface_pressure",
                "precip",
                "u10",
                "v10",
                "shortwave",
                "clear_shortwave",
                "longwave",
            },
        )
        self.assertEqual(
            time.tolist(), pd.date_range(LAKES_START, LAKES_END, freq="h").tolist()
        )
        np.testing.assert_allclose(
            raw["air_temp"],
            [self.read_lakes(lakes_file(h), "T2", rows, cols) for h in hours],
        )
        files = [LAKES_DIR / previous] + [lakes_file(h) for h in hours]
        np.testing.assert_allclose(
            raw["precip"],
            [
                self.read_lakes(f, "RAINNC", rows, cols)
                + self.read_lakes(f, "RAINC", rows, cols)
                for f in files
            ],
        )
        self.assertIsNone(wrf.file)
        self.assertIsNone(wrf.hours)

    def test_read_precip_without_rainc(self):
        wrf = InputWRF(START_DATE, END_DATE, BBOX, TOPO, make_config("unused.nc"))
        wrf.input_type = wrf_schema.WUS_D3
        wrf.file = MagicMock(variables={"RAINNC": None})

        with patch.object(InputWRF, "_read_field", return_value=np.array([[1.0]])):
            precip = wrf._read_precip(np.array([True]))

        np.testing.assert_allclose(precip, [[1.0]])

    @staticmethod
    def read_lakes(path: Path, name: str, rows, cols) -> np.ndarray:
        """
        Read one Lakes field at its first time step for the given grid cells.

        Parameters
        ----------
        path : Path
            Lakes hourly file
        name : str
            Name of the WRF variable
        rows, cols : ndarray
            Row and column of each grid cell in the full WRF grid

        Returns
        -------
        ndarray
            Field values for each grid cell
        """
        with nc.Dataset(path) as f:
            return f.variables[name][0][rows, cols]

    def lakes_folder(self, hours: list[str]) -> str:
        """
        Copy the Lakes hourly files for the given hours into a new folder.

        Parameters
        ----------
        hours : list of str
            Two-digit hours to copy, such as ["00", "01"]

        Returns
        -------
        str
            Path of the folder
        """
        folder = self.enterContext(tempfile.TemporaryDirectory())
        for hour in hours:
            shutil.copy(lakes_file(hour), folder)
        return folder


class TestProcessWUS_D3(unittest.TestCase):
    """
    Tests for the WUS-D3 outputs of InputWRF.process.
    """

    @classmethod
    def setUpClass(cls):
        raw = {
            "air_temp": np.array([[275.15]]),
            "precip": np.array([[1.0], [3.0]]),
            "u10": np.array([[3.0]]),
            "v10": np.array([[4.0]]),
            "humidity": np.array([[0.01]]),
            "surface_pressure": np.array([[90000.0]]),
            "shortwave": np.array([[300.0]]),
            "clear_shortwave": np.array([[400.0]]),
            "longwave": np.array([[350.0]]),
        }
        cls.wrf = InputWRF(START_DATE, START_DATE, BBOX, TOPO, make_config("wrf.nc"))
        cls.wrf.input_type = wrf_schema.WUS_D3
        cls.wrf.hours = pd.DatetimeIndex([START_DATE])
        cls.wrf.primary_id = [CENTER_NAME]
        cls.wrf.process(raw, cls.wrf.hours)

    def test_vapor_pressure(self):
        # Dalton's law from the mixing ratio, as in vapor_pressure_from_mixing_ratio
        np.testing.assert_allclose(
            self.wrf.vapor_pressure[CENTER_NAME], 0.01 * 90000.0 / (EPS + 0.01)
        )

    def test_air_temp(self):
        np.testing.assert_allclose(self.wrf.air_temp[CENTER_NAME], 2.0)

    def test_wind_speed(self):
        np.testing.assert_allclose(self.wrf.wind_speed[CENTER_NAME], 5.0)

    def test_wind_direction(self):
        np.testing.assert_allclose(
            self.wrf.wind_direction[CENTER_NAME], np.degrees(np.arctan2(4.0, 3.0))
        )

    def test_precip(self):
        np.testing.assert_allclose(self.wrf.precip[CENTER_NAME], 2.0)

    def test_cloud_factor(self):
        np.testing.assert_allclose(self.wrf.cloud_factor[CENTER_NAME], 0.75)

    def test_thermal(self) -> None:
        np.testing.assert_allclose(self.wrf.thermal[CENTER_NAME], 350.0)
