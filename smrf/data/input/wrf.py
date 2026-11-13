from types import MappingProxyType

import netCDF4 as nc
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytz

from smrf.data.helper import add_utm_columns, grid_station_names
from smrf.data.input.gridded_input import GriddedInput
from smrf.envphys.vapor_pressure import svp_for_celsius


class InputWRF(GriddedInput):
    """
    Load data from any WRF-based netcdf files.
    Examples include WRF-Hydro, CONUS-404, or WUS-D3.
    """

    DATA_TYPE = "wrf"

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
        self.metadata_file = self.config.get("wrf_metadata")

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
        return pd.to_datetime(times, format="%Y-%m-%d_%H:%M:%S", utc=in_utc)

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
        Load the data from a WRF output file set in the wrf_file option.

        The input file must contain these WRF variables:
        - Times
        - T2
        - DWPT
        - RAINNC
        - CLDFRA
        - UGRD
        - VGRD

        The grid fields XLAT, XLONG and HGT are read by get_metadata, from the
        wrf_metadata file when it is set and from wrf_file otherwise.

        Only the grid cells that are within the topo domain are read. Each cell is
        identified by grid_y{y}_x{x}, using its index in the full WRF grid.
        """

        self._logger.info("Reading WRF data from: {}".format(self.config["wrf_file"]))
        with nc.Dataset(self.config["wrf_file"]) as self.file:
            self.get_metadata()

            time = self.get_file_times()
            time_index = (time >= self.start_date) & (time <= self.end_date)
            raw = self.read_raw(time_index)

        self.process(raw, time[time_index])

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
        """
        self._logger.debug("Converting air temperature from K to C")
        air_temp = raw_data["air_temp"] - 273.15
        self.air_temp = pd.DataFrame(air_temp, index=time, columns=self.primary_id)

        self._logger.debug("Calculating vapor pressure from dew point")
        vapor_pressure = svp_for_celsius(raw_data["dew_point"])
        self.vapor_pressure = pd.DataFrame(
            vapor_pressure, index=time, columns=self.primary_id
        )

        self._logger.debug("Calculating cloud factor")
        cloud_factor = 1 - np.mean(raw_data["cloud_factor"], axis=1)
        self.cloud_factor = pd.DataFrame(
            cloud_factor, index=time, columns=self.primary_id
        )

        self._logger.debug("Calculating wind speed and wind direction")
        speed = np.sqrt(raw_data["u10"] ** 2 + raw_data["v10"] ** 2)
        self.wind_speed = pd.DataFrame(speed, index=time, columns=self.primary_id)
        # arctan2 returns -180 to 180, so wrap the angles into 0 to 360
        direction = np.mod(
            np.degrees(np.arctan2(raw_data["v10"], raw_data["u10"])), 360
        )
        self.wind_direction = pd.DataFrame(
            direction, index=time, columns=self.primary_id
        )

        self._logger.debug("Loading precipitation")
        self.precip = pd.DataFrame(
            raw_data["precip"], index=time, columns=self.primary_id
        )
        self.precip = self.precip.diff(axis=0)
        # RAINNC is accumulated, so the first step has no previous value to
        # difference against. Its precip is set to zero.
        self.precip.iloc[0, :] = 0
