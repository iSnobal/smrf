import logging
from datetime import datetime

import netCDF4 as nc
import numpy as np


class BufferedNetcdfWriter:
    """
    File writer that buffers time steps of (y, x) arrays and writes them to
    NetCDF files in blocks.

    The buffer is written when a time step on a new date is added, after the
    time step at ``FLUSH_HOUR`` is added, and on :meth:`flush`. Each file is
    opened once per write and can hold several variables. Times that are
    already in a file are overwritten.
    """

    FLUSH_HOUR = 23

    def __init__(self, file_to_variables: dict[str, list[str]]) -> None:
        """
        Parameters
        ----------
        file_to_variables : dict[str, list[str]]
            Path of an existing NetCDF file to the names of the variables it holds
        """
        self._logger = logging.getLogger(__name__)
        self.file_to_variables = file_to_variables
        self._times = []
        self._arrays = {
            variable: []
            for variables in file_to_variables.values()
            for variable in variables
        }

    def add(self, date_time: datetime, data: dict[str, np.ndarray]) -> None:
        """
        Buffer one time step.

        Parameters
        ----------
        date_time : datetime
            Time of the step
        data : dict[str, np.ndarray]
            Variable name to its (y, x) array, for every registered variable

        Raises
        ------
        KeyError
            When a registered variable is missing from ``data``
        """
        # Flush if the date has changed since the last time step
        if self._times and date_time.date() != self._times[-1].date():
            self.flush()

        self._times.append(date_time)
        for variable, arrays in self._arrays.items():
            # Keep a copy of the data since later calls might change the array
            arrays.append(data[variable].copy())

        # Flush if the time step is at the flush hour (end of day)
        if date_time.hour == self.FLUSH_HOUR:
            self.flush()

    def flush(self) -> None:
        """
        Write all buffered time steps and empty the buffer.
        """
        # Nothing to write when the last time step was at the flush hour
        # or no time step was added
        if not self._times:
            return

        try:
            for file_path, variables in self.file_to_variables.items():
                self._logger.debug(
                    f"Writing {len(self._times)} time steps of {variables} to {file_path}"
                )
                with nc.Dataset(file_path, mode="a") as dataset:
                    self.write_file(dataset, variables)
        finally:
            # Empty the buffer also when a write fails, so it is not written twice
            self._times.clear()
            for arrays in self._arrays.values():
                arrays.clear()

    def write_file(self, dataset: nc.Dataset, variables: list[str]) -> None:
        """
        Write the buffered times and variables to one open file.

        Parameters
        ----------
        dataset : nc.Dataset
            File opened in append mode
        variables : list[str]
            Variables in this file
        """
        time_variable = dataset.variables["time"]
        time_values = nc.date2num(
            [date_time.replace(tzinfo=None) for date_time in self._times],
            time_variable.units,
            time_variable.calendar,
        )
        indices = self.time_indices(time_variable[:], time_values)

        time_variable[indices] = time_values
        for variable in variables:
            dataset.variables[variable][indices, :, :] = np.stack(self._arrays[variable])

    @staticmethod
    def time_indices(file_times: np.ndarray, time_values: np.ndarray) -> np.ndarray:
        """
        Index in the file for each time value.

        Parameters
        ----------
        file_times : np.ndarray
            Time values already in the file
        time_values : np.ndarray
            Time values to write

        Returns
        -------
        np.ndarray
            Index of each time value, either its existing index or a new one
            after the end of the file
        """
        indices = np.empty(len(time_values), dtype=int)
        # New time values are appended after the last time in the file
        next_index = len(file_times)
        for position, value in enumerate(time_values):
            match = np.flatnonzero(file_times == value)
            if match.size > 0:
                # Time is already in the file, overwrite its data
                indices[position] = match[0]
            else:
                # Time is not in the file, append it
                indices[position] = next_index
                next_index += 1
        return indices
