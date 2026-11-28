import os
import tempfile
import unittest

import netCDF4 as nc
import numpy as np
import numpy.testing as npt
import pandas as pd

from smrf.output.buffered_netcdf_writer import BufferedNetcdfWriter


class TestBufferedNetcdfWriter(unittest.TestCase):
    SHAPE = (2, 3)
    START_DATE = "2025-10-01 00:00"
    VARIABLES = ("air_temp", "thermal")

    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.out_dir = temp_dir.name

        self.file_paths = {}
        for variable in self.VARIABLES:
            self.file_paths[variable] = os.path.join(self.out_dir, variable + ".nc")
            self.create_file(self.file_paths[variable], [variable])

        self.writer = BufferedNetcdfWriter(
            {file_path: [variable] for variable, file_path in self.file_paths.items()}
        )

    def create_file(self, file_path: str, variables: list[str]) -> None:
        """
        Create an empty output file with hourly time units since START_DATE.
        """
        with nc.Dataset(file_path, "w") as dataset:
            dataset.createDimension("time", None)
            dataset.createDimension("y", self.SHAPE[0])
            dataset.createDimension("x", self.SHAPE[1])

            time_variable = dataset.createVariable("time", "f4", ("time",))
            time_variable.units = f"hours since {self.START_DATE}"
            time_variable.calendar = "standard"

            for variable in variables:
                dataset.createVariable(variable, "f4", ("time", "y", "x"))

    @staticmethod
    def time_steps(start: str, count: int, hours: int = 1) -> pd.DatetimeIndex:
        """
        Time steps in UTC, as they are passed.
        """
        return pd.date_range(start, periods=count, freq=f"{hours}h", tz="UTC")

    def add_steps(self, time_steps: pd.DatetimeIndex, offset: float = 0) -> None:
        """
        Add each time step with all cells set to its position plus offset.
        """
        for position, time_step in enumerate(time_steps):
            self.writer.add(
                time_step,
                {
                    variable: np.full(self.SHAPE, position + offset)
                    for variable in self.VARIABLES
                },
            )

    def read(self, variable: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Times and values of a variable in its file.
        """
        with nc.Dataset(self.file_paths[variable]) as dataset:
            return dataset.variables["time"][:], dataset.variables[variable][:]

    def assert_written(self, hours: list[float], values: list[float]) -> None:
        """
        Assert every variable file holds the hours, with all grid cells set to each value.
        """
        expected = np.asarray(values, dtype=float)[:, None, None] * np.ones(self.SHAPE)
        for variable in self.VARIABLES:
            file_times, data = self.read(variable)
            npt.assert_equal(file_times, hours)
            npt.assert_equal(data, expected)

    def test_add_buffers(self):
        self.add_steps(self.time_steps(self.START_DATE, 3))

        self.assert_written([], [])

    def test_add_flushes_at_flush_hour(self):
        self.add_steps(self.time_steps(self.START_DATE, 24))

        self.assert_written(list(range(24)), list(range(24)))

    def test_add_flushes_on_new_date(self):
        self.add_steps(self.time_steps("2025-10-01 15:00", 4, hours=3))

        # Adding 00:00 on the next date writes 15:00 to 21:00, and 00:00 stays in the buffer
        self.assert_written([15, 18, 21], [0, 1, 2])

    def test_flush_part_day(self):
        self.add_steps(self.time_steps(self.START_DATE, 3))
        self.writer.flush()

        self.assert_written([0, 1, 2], [0, 1, 2])

    def test_flush_empty(self):
        self.writer.flush()

        self.assert_written([], [])

    def test_flush_empties_buffer(self):
        self.add_steps(self.time_steps(self.START_DATE, 3))
        self.writer.flush()
        self.writer.flush()

        self.assert_written([0, 1, 2], [0, 1, 2])

    def test_flush_failed_file_empties_buffer(self):
        self.add_steps(self.time_steps(self.START_DATE, 1))
        # Replace the second file with one that does not have its variable
        self.create_file(self.file_paths[self.VARIABLES[1]], [])

        with self.assertRaises(KeyError):
            self.writer.flush()
        # The buffer is empty, so the second flush does not write again
        self.writer.flush()

        file_times, data = self.read(self.VARIABLES[0])
        npt.assert_equal(file_times, [0])
        npt.assert_equal(data, np.zeros((1, *self.SHAPE)))

    def test_overwrite(self):
        time_steps = self.time_steps(self.START_DATE, 3)
        self.add_steps(time_steps)
        self.writer.flush()
        self.add_steps(time_steps, offset=10)
        self.writer.flush()

        self.assert_written([0, 1, 2], [10, 11, 12])

    def test_overwrite_and_append(self):
        self.add_steps(self.time_steps(self.START_DATE, 3))
        self.writer.flush()
        self.add_steps(self.time_steps("2025-10-01 01:00", 4), offset=10)
        self.writer.flush()

        # 01:00 and 02:00 are overwritten, 03:00 and 04:00 are appended
        self.assert_written([0, 1, 2, 3, 4], [0, 10, 11, 12, 13])

    def test_overwrite_non_contiguous(self):
        self.add_steps(self.time_steps(self.START_DATE, 3))
        self.writer.flush()
        self.add_steps(self.time_steps("2025-10-01 01:00", 2, hours=4), offset=10)
        self.writer.flush()

        # 01:00 is overwritten and 05:00 is appended after 02:00
        self.assert_written([0, 1, 2, 5], [0, 10, 2, 11])

    def test_add_copies(self):
        data = {variable: np.zeros(self.SHAPE) for variable in self.VARIABLES}
        self.writer.add(self.time_steps(self.START_DATE, 1)[0], data)
        for array in data.values():
            array += 1
        self.writer.flush()

        self.assert_written([0], [0])

    def test_missing_variable(self):
        with self.assertRaises(KeyError):
            self.writer.add(
                self.time_steps(self.START_DATE, 1)[0],
                {self.VARIABLES[0]: np.zeros(self.SHAPE)},
            )

    def test_shared_file(self):
        file_path = os.path.join(self.out_dir, "shared.nc")
        self.create_file(file_path, list(self.VARIABLES))
        self.writer = BufferedNetcdfWriter({file_path: list(self.VARIABLES)})
        # Read all variables from the shared file
        self.file_paths = {variable: file_path for variable in self.VARIABLES}

        self.add_steps(self.time_steps(self.START_DATE, 3))
        self.writer.flush()

        self.assert_written([0, 1, 2], [0, 1, 2])
