import os
from types import MappingProxyType

import netCDF4 as nc
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytz

from smrf.data.helper import add_utm_columns, grid_station_names
from smrf.data.input.gridded_input import GriddedInput
from smrf.envphys.vapor_pressure import svp_for_celsius

WRF_TIME_FORMAT = "%Y-%m-%d_%H:%M:%S"


class InputWRF(GriddedInput):
    """
    Load data from any WRF-based netcdf files.
    Examples include WRF-Hydro, CONUS-404, or WUS-D3.
    """

    DATA_TYPE = "wrf"

    class ConfigKey:
        PATH = "wrf_path"
        PREFIX = "wrf_prefix"
        METADATA = "wrf_metadata"

    WRF_VARIABLES = MappingProxyType(
        {
            "air_temp": "T2",
            "dew_point": "DWPT",
            "u10": "UGRD",
            "v10": "VGRD",
            "cloud_factor": "CLDFRA",
            "precip": "RAINNC",
        }
    )

    GRID_VARIABLES = ("XLAT", "XLONG", "HGT")

    def __init__(self, start_date, end_date, bbox, topo, config):
        super().__init__(start_date, end_date, bbox, topo, config)

        # Use the user file when given in the .ini to read grid variables
        self.metadata_file = self.config.get(self.ConfigKey.METADATA)

    def get_file_times(self) -> pd.DatetimeIndex:
        """
        Read the times from the file and interpret them as WRF timestamps.

        Raises:
            ValueError: a time does not match the WRF format

        Returns:
            list: list of date times
        """

        t = self.file.variables["Times"]
        t.set_auto_maskandscale(False)
        times = nc.chartostring(t[:])

        in_utc = str(self.time_zone).lower() == str(pytz.UTC).lower()
        return pd.to_datetime(times, format=WRF_TIME_FORMAT, utc=in_utc)

    @staticmethod
    def _read_grid_fields(
        file_with_grid_info: nc.Dataset,
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Read the grid fields from an open netCDF dataset.

        Args:
            file_with_grid_info: netCDF4.Dataset holding every variable listed
                in :attr:`InputWRF.GRID_VARIABLES`

        Raises:
            ValueError: a variable from :attr:`InputWRF.GRID_VARIABLES` is
                missing from the dataset

        Returns:
            tuple: XLAT, XLONG and HGT as float64 arrays
        """
        missing = [
            v for v in InputWRF.GRID_VARIABLES if v not in file_with_grid_info.variables
        ]
        if missing:
            raise ValueError(
                f"{missing} not found in {file_with_grid_info.filepath()}. "
                "The grid fields can be "
                "supplied with the wrf_metadata .ini file option"
            )

        xlat = file_with_grid_info.variables["XLAT"][:].astype("float64")
        xlong = file_with_grid_info.variables["XLONG"][:].astype("float64")
        hgt = file_with_grid_info.variables["HGT"][:].astype("float64")
        return xlat, xlong, hgt

    def get_metadata(self) -> None:
        # Use the metadata file for the grid fields when one is set
        if self.metadata_file:
            with nc.Dataset(self.metadata_file) as grid:
                xlat, xlong, hgt = self._read_grid_fields(grid)
        else:
            xlat, xlong, hgt = self._read_grid_fields(self.file)

        # get the values that are in the modeling domain
        model_domain_mask = (
            (xlat >= self.bbox[1])
            & (xlat <= self.bbox[3])
            & (xlong >= self.bbox[0])
            & (xlong <= self.bbox[2])
        )

        # create some fake station names based on the index
        self.station_index = np.argwhere(model_domain_mask)
        self.primary_id = grid_station_names(
            self.station_index[:, 0], self.station_index[:, 1]
        )
        self._logger.debug(f"{len(self.station_index)} grid cells within model domain")

        # create a metadata dataframe to store all the grid info
        metadata = pd.DataFrame(
            {
                "latitude": xlat[model_domain_mask],
                "longitude": xlong[model_domain_mask],
                "elevation": hgt[model_domain_mask],
            },
            index=self.primary_id,
        )
        metadata = add_utm_columns(metadata, self.topo.zone_number)

        self.metadata = metadata

    def load(self) -> None:
        """
        Load the data from a WRF output file set in the wrf_path option.

        The input file must contain these WRF variables:
        - Times
        - T2
        - DWPT
        - RAINNC
        - CLDFRA
        - UGRD
        - VGRD

        The grid fields XLAT, XLONG and HGT are read by get_metadata, from the
        wrf_metadata file when it is set and from wrf_path otherwise.

        If wrf_path is a directory, the file for each hour from start_date through
        end_date is read, named with wrf_prefix. A missing file raises a ValueError
        when the data is processed.

        Only the grid cells that are within the topo domain are read. Each cell is
        identified by grid_y{y}_x{x}, using its index in the full WRF grid.
        """

        self.hours = pd.date_range(self.start_date, self.end_date, freq="h")
        wrf_path = self.config[self.ConfigKey.PATH]
        self._logger.info(f"Reading WRF data from: {wrf_path}")
        if os.path.isdir(wrf_path):
            if not self.config.get(self.ConfigKey.PREFIX):
                raise ValueError("wrf_prefix must be set when wrf_path is a directory")
            self._load_directory()
        else:
            self._load_file(wrf_path)
        self.file = None
        self.hours = None

    def _load_file(self, path: str) -> None:
        """
        Load the time steps of one WRF file that fall in the requested period.

        Parameters
        ----------
        path : str
            Path to the WRF file.
        """
        with nc.Dataset(path) as self.file:
            self.get_metadata()
            # Times in the hourly files can be a few seconds off the hour, for
            # example 01:00:49 in the 01:00 file, so round them before matching
            time = self.get_file_times().round("h")
            time_index = time.isin(self.hours)
            raw = self.read_raw(time_index)
        self.process(raw, time[time_index])

    def _load_directory(self) -> None:
        """
        Load one hourly WRF file per hour in the requested period.

        A missing file is logged as a warning. The load then raises a ValueError
        in process, because the hour is missing from the data.
        """
        time_parts, raw_parts = [], []
        for hour in self.hours:
            path = self._file_for_time(hour)
            if path is None:
                continue
            with nc.Dataset(path) as self.file:
                if not time_parts:
                    self.get_metadata()
                time = self.get_file_times().round("h")
                time_index = time.isin(self.hours)
                raw_parts.append(self.read_raw(time_index))
                time_parts.append(time[time_index])

        if not raw_parts:
            raise ValueError(
                f"No WRF files in {self.config[self.ConfigKey.PATH]} for the period "
                f"{self.start_date} to {self.end_date}"
            )
        raw = {k: np.concatenate([p[k] for p in raw_parts]) for k in raw_parts[0]}
        self.process(raw, time_parts[0].append(time_parts[1:]))

    def _file_for_time(self, time: pd.Timestamp) -> str | None:
        """
        Get the path of the hourly WRF file for a time.

        Parameters
        ----------
        time : pd.Timestamp
            Hour of the file to find.

        Returns
        -------
        str or None
            Path of the file, or None if it is missing.
        """
        name = (
            f"{self.config[self.ConfigKey.PREFIX]}_{time.strftime(WRF_TIME_FORMAT)}.nc"
        )
        path = os.path.join(self.config[self.ConfigKey.PATH], name)
        if os.path.exists(path):
            return path
        self._logger.warning(f"No WRF file for {time}")
        return None

    def _topo_domain(self) -> tuple[slice, slice]:
        """
        Get the row and column slices of the topo domain in the full WRF grid.

        Returns:
            tuple: row slice and column slice in the full WRF grid
        """
        rows = self.station_index[:, 0]
        cols = self.station_index[:, 1]
        return (
            slice(rows.min(), rows.max() + 1),
            slice(cols.min(), cols.max() + 1),
        )

    def read_raw(self, time_ind: npt.NDArray) -> dict[str, npt.NDArray]:
        """
        Read the WRF fields for the selected times over the topo domain.

        Args:
            time_ind: boolean mask of the times to read

        Returns:
            dict: station arrays keyed by variable name
        """
        topo_rows, topo_cols = self._topo_domain()
        rows = self.station_index[:, 0] - topo_rows.start
        cols = self.station_index[:, 1] - topo_cols.start
        raw = {}
        for variable, wrf_variable in self.WRF_VARIABLES.items():
            self._logger.debug(
                f"Loading variable {variable} from WRF field {wrf_variable}"
            )
            field = self.file.variables[wrf_variable]
            if variable == "cloud_factor":
                # CLDFRA has a vertical dimension, which is kept in full
                topo_values = field[time_ind, :, topo_rows, topo_cols]
                raw[variable] = topo_values[:, :, rows, cols]
            else:
                topo_values = field[time_ind, topo_rows, topo_cols]
                raw[variable] = topo_values[:, rows, cols]
        return raw

    def process(self, raw_data: dict[str, npt.NDArray], time: pd.DatetimeIndex) -> None:
        """
        Calculate variables from loaded WRF data.
        Converts all instance variables to pandas DataFrames.

        Args:
            raw_data: Read WRF values from read_raw, keyed by variable name
            time: timestamps of the selected times

        Raises:
            ValueError: a requested hour is missing from time
        """
        missing = self.hours.difference(time)
        if len(missing) > 0:
            raise ValueError(f"WRF data has no time step for {missing}")

        # RAINNC is accumulated, so the first step has no previous value to
        # difference against. Its precip is set to zero.
        precip = np.zeros_like(raw_data["precip"])
        precip[1:] = np.diff(raw_data["precip"], axis=0)

        # arctan2 returns -180 to 180, so wrap the angles into 0 to 360
        direction = np.mod(
            np.degrees(np.arctan2(raw_data["v10"], raw_data["u10"])), 360
        )

        frames = {
            "air_temp": raw_data["air_temp"] - 273.15,
            "vapor_pressure": svp_for_celsius(raw_data["dew_point"]),
            "cloud_factor": 1 - np.mean(raw_data["cloud_factor"], axis=1),
            "wind_speed": np.sqrt(raw_data["u10"] ** 2 + raw_data["v10"] ** 2),
            "wind_direction": direction,
            "precip": precip,
        }
        for name, values in frames.items():
            frame = pd.DataFrame(values, index=time, columns=self.primary_id)
            setattr(self, name, frame)
