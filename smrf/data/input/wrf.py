from types import MappingProxyType

import netCDF4 as nc
import numpy as np
import pandas as pd
import pytz

from smrf.data.input.gridded_input import GriddedInput
from smrf.data.input.netcdf import metadata_name_from_index
from smrf.envphys.vapor_pressure import svp_for_celsius
from smrf.utils.utils import apply_utm


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

    def get_metadata(self):

        # get the values that are in the modeling domain
        ind = (
            (self.file.variables["XLAT"] >= self.bbox[1])
            & (self.file.variables["XLAT"] <= self.bbox[3])
            & (self.file.variables["XLONG"] >= self.bbox[0])
            & (self.file.variables["XLONG"] <= self.bbox[2])
        )

        mlat = self.file.variables["XLAT"][:][ind]
        mlon = self.file.variables["XLONG"][:][ind]
        mhgt = self.file.variables["HGT"][:][ind]

        # create some fake station names based on the index
        self.station_index = np.argwhere(ind)
        self.primary_id = [metadata_name_from_index(i) for i in self.station_index]
        self._logger.debug(f"{len(self.station_index)} grid cells within model domain")

        # create a metadata dataframe to store all the grid info
        metadata = pd.DataFrame(
            index=self.primary_id,
            columns=("utm_x", "utm_y", "latitude", "longitude", "elevation"),
        )

        metadata["latitude"] = mlat.flatten()
        metadata["longitude"] = mlon.flatten()
        metadata["elevation"] = mhgt.flatten()
        metadata = metadata.apply(apply_utm, args=(self.topo.zone_number,), axis=1)

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
            for i in self.station_index:
                g = metadata_name_from_index(i)
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
        for i in self.station_index:
            g = metadata_name_from_index(i)
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

        for i in self.station_index:
            g = metadata_name_from_index(i)
            self.wind_speed[g] = s[:, i[0], i[1]]
            self.wind_direction[g] = d[:, i[0], i[1]]

        self._logger.debug("Loading precip")
        self.precip = self.precip.diff(axis=0)
        self.precip.iloc[0, :] = 0
