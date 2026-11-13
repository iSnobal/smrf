from types import MappingProxyType

import netCDF4 as nc
import numpy as np
import pandas as pd
import pytz

from smrf.data.input.gridded_input import GriddedInput
from smrf.envphys.vapor_pressure import svp_for_celsius
from smrf.data.helper import add_utm_columns, grid_station_names


class InputWRF(GriddedInput):
    """
    Load data from any WRF-based netcdf files.
    Examples include WRF-Hydro, CONUS-404, or WUS-D3.
    """

    DATA_TYPE = "wrf"

    BASE_VARIABLES = MappingProxyType(
        {"air_temp": "T2", "dew_point": "DWPT", "precip": "RAINNC"}
    )

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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    def load(self):
        """
        Load the data from a netcdf file. This was setup to work with a WRF
        output file, i.e. wrf_out so it's going to look for the following
        variables:
        - Times
        - XLAT
        - XLONG
        - HGT
        - T2
        - DWPT
        - GLW
        - RAINNC
        - CLDFRA
        - UGRD
        - VGRD

        Each cell will be identified by grid_IX_IY
        """

        self._logger.info(
            "Reading data coming from WRF output: {}".format(self.config["wrf_file"])
        )
        self.file = nc.Dataset(self.config["wrf_file"])

        self.get_metadata()

        time = self.get_file_times()
        time_ind = (time >= self.start_date) & (time <= self.end_date)
        time = time[time_ind]

        for variable, wrf_variable in self.BASE_VARIABLES.items():
            self._logger.debug(
                f"Loading variable {variable} from WRF field {wrf_variable}"
            )

            df = pd.DataFrame(index=time, columns=self.primary_id)
            for g, i in zip(self.primary_id, self.station_index):
                df[g] = self.file.variables[wrf_variable][time_ind, i[0], i[1]]

            setattr(self, variable, df)

        # Post processing of the data
        self.air_temp = self.air_temp - 273.15

        self._logger.debug("Calculating vapor_pressure")
        vp = svp_for_celsius(self.dew_point.values)
        self.vapor_pressure = pd.DataFrame(vp, index=time, columns=self.primary_id)

        self._logger.debug("Loading cloud_factor")
        self.cloud_factor = pd.DataFrame(index=time, columns=self.primary_id)
        cf = 1 - np.mean(self.file.variables["CLDFRA"][time_ind, :], axis=1)
        for g, i in zip(self.primary_id, self.station_index):
            v = cf[:, i[0], i[1]]
            self.cloud_factor[g] = v

        self._logger.debug("Loading wind_speed and wind_direction")
        self.wind_speed = pd.DataFrame(index=time, columns=self.primary_id)
        self.wind_direction = pd.DataFrame(index=time, columns=self.primary_id)

        u10 = self.file.variables["UGRD"][time_ind, :]
        v10 = self.file.variables["VGRD"][time_ind, :]

        # Calculate the wind speed
        s = np.sqrt(u10**2 + v10**2)

        # Calculate the wind direction
        d = np.degrees(np.arctan2(v10, u10))
        ind = d < 0
        d[ind] = d[ind] + 360

        for g, i in zip(self.primary_id, self.station_index):
            self.wind_speed[g] = s[:, i[0], i[1]]
            self.wind_direction[g] = d[:, i[0], i[1]]

        self._logger.debug("Loading precip")
        self.precip = self.precip.diff(axis=0)
        self.precip.iloc[0, :] = 0
