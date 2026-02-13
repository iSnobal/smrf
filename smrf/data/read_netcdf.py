import logging
from datetime import datetime, tzinfo
from pathlib import Path

import netCDF4
import numpy.typing as npt
from cftime import num2date


class ReadNetCDF:
    """
    General purpose class to load external NetCDF data for calculations in SMRF.

    The data are usually forcing data that were calculated outside the main source
    (i.e. HRRR). Example is using a remote sensing source for albedo over the time-decay
    function.

    This class can also be used in general to load a variable from a NetCDF file such as
    the snow.nc file from pysnobal.

    Design of this class:
    * Upon initialization: Open the file, read all available variables and timesteps
    * To get a value for a single time step, use the :py:meth:`load`
    * The opened file is automatically closed upon garbage collection
    """

    def __init__(self, file: Path, time_zone: tzinfo):
        self.file = netCDF4.Dataset(file, "r")
        self.time_zone = time_zone
        self._logger = logging.getLogger(self.__class__.__module__)

        self.dates = None
        self._load_timesteps()
        self.variables = list(self.file.variables.keys())

        self._logger.info(f"Opening file: {self.file.name} for reading")

    def _load_timesteps(self) -> None:
        """
        Load and parse timesteps from the NetCDF file's time variable.

        Converts the "time" variable from the NetCDF file into a list of timestamps
        using the file's time units and calendar. The timestamps are stored in
        as Unix timestamps in the configured timezone from :py:meth:`__init__`.

        Sets:
        :py:attr:`dates`
        """
        date_times = self.file["time"]
        dates = num2date(
            date_times[:],
            units=date_times.units,
            calendar=date_times.calendar,
            only_use_cftime_datetimes=False,
        )
        self.dates = [date.replace(tzinfo=self.time_zone).timestamp() for date in dates]
        self._logger.debug(f"Found {len(self.dates)} timesteps in file: {self.file.name}")

    def load(self, variable_name: str, timestep: datetime) -> npt.NDArray:
        """
        Load given variable at a given timestep from the NetCDF file.

        Args:
            variable_name: The name of the variable to load from the NetCDF file.
            timestep: Datetime object of requested timestep.

        Returns:
            Values for variable at the timestep

        Raises:
            ValueError: If the timestep is not found in the file's dates.
        """
        self._logger.debug(
            f"Reading variable {variable_name} at time {str(datetime)} from file: {self.file.name}"
        )
        return self.file[variable_name][self.dates.index(timestep.timestamp())]

    def close(self):
        """
        Closes the file handle
        """
        if hasattr(self, "file") and self.file.isopen():
            self.file.close()
