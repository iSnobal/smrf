import numpy as np
import pandas as pd
from smrf.data.hrrr.file_loader import FileLoader
from smrf.data.hrrr.grib_file_gdal import GribFileGdal
from smrf.distribute.wind.wind_ninja import WindNinjaModel
from smrf.envphys.solar.cloud import get_hrrr_cloud
from smrf.envphys.vapor_pressure import rh2vp

from .gridded_input import GriddedInput
from smrf.distribute import SolarHRRR, ThermalHRRR


class InputGribHRRR(GriddedInput):
    """
    Load the data from the High Resolution Rapid Refresh (HRRR) model.
    The possible variables returned are dependent on the configuration
    and returned as Pandas dataframes.
    """

    DATA_TYPE = "hrrr_grib"
    GDAL_VARIABLE_KEY = "hrrr_gdal_variables"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cloud_factor_memory = None

        # Skip loading wind when WindNinja is used
        wind_model = kwargs["config"].get("wind", {}).get("wind_model", None)
        self._load_wind = wind_model != WindNinjaModel.MODEL_TYPE

        self._calculate_cloud_factor = (
            "hrrr_cloud" not in kwargs["config"]["output"]["variables"]
        )

        self._load_gdal = self.config.get(self.GDAL_VARIABLE_KEY) or []
        # Matches the default value from CoreConfig.ini
        self._gdal_algorithm = (
            self.config.get("hrrr_gdal_algorithm") or GribFileGdal.DEFAULT_ALGORITHM
        )

    def load(self):
        """
        The function will take the keys and load them into the appropriate
        objects within the `grid` class.
        """
        self._logger.debug(
            "Reading data from from HRRR directory: {}".format(
                self.config["hrrr_directory"]
            )
        )

        data = FileLoader(
            external_logger=self._logger,
            file_dir=self.config["hrrr_directory"],
            forecast_hour=self.config["hrrr_forecast_hour"],
            load_gdal=self._load_gdal,
            gdal_algorithm=self._gdal_algorithm,
            load_wind=self._load_wind,
            sixth_hour_variables=self.config["hrrr_sixth_hour_variables"],
        ).data_for_time_and_topo(
            start_date=self.start_date,
            bbox=self.bbox,
            topo=self.topo,
        )

        self.parse_data(data)

    def load_timestep(self, date_time):
        """Load a single time step for HRRR

        Args:
            date_time (datetime): date time to load
        """

        self.start_date = date_time
        self.load()

    def parse_data(self, data):
        """
        Parse the data from HRRR into Pandas dataframes for SMRF.
        Set variables on the class to retrieve them at distribution time.

        # Attributes set:
        * :py:attr: metadata
        * :py:attr: air_temp
        * :py:attr: vapor_pressure
        * :py:attr: precip
        ## If configured to load cloud factor:
            * :py:attr: cloud_factor
        ## If configured to load wind:
            * :py:attr: wind_speed
            * :py:attr: wind_direction

        Args:
            data (dict): dictionary of DataFrames from HRRR
        """
        # Enforce configured timezone
        for key in data.keys():
            # Skip conversion when loaded via GDAL as there is no use of dataframes
            if key in self._load_gdal:
                setattr(self, key, data[key])
            else:
                data[key] = data[key].apply(pd.to_numeric)
                data[key] = data[key].tz_localize(tz=self.time_zone)

        idx = data["air_temp"].index
        cols = data["air_temp"].columns

        self._logger.debug("Loading air_temp")
        self.air_temp = data["air_temp"]

        # calculate vapor pressure
        self._logger.debug("Calculating vapor_pressure")
        vp = rh2vp(data["air_temp"].values, data["relative_humidity"].values)
        self.vapor_pressure = pd.DataFrame(vp, index=idx, columns=cols)

        self.calculate_wind(data)

        # precip
        self._logger.debug("Loading precip")
        self.precip = pd.DataFrame(data["precip_int"], index=idx, columns=cols)

        # cloud factor
        if self._calculate_cloud_factor:
            self._logger.debug("Loading solar")
            solar = pd.DataFrame(data["short_wave"], index=idx, columns=cols)
            self._logger.debug("Calculating cloud factor")
            self.cloud_factor = get_hrrr_cloud(
                solar, self.metadata, self.topo.basin_lat, self.topo.basin_long
            )
            self.check_cloud_factor()

        # DLWRF from HRRR set the "thermal"
        if data.get(ThermalHRRR.GRIB_NAME, None) is not None:
            setattr(self, ThermalHRRR.DISTRIBUTION_KEY, data[ThermalHRRR.GRIB_NAME])

        # DSWRF, VBDSF, and VDDSF for shortwave
        for variable in SolarHRRR.GRIB_VARIABLES:
            if data.get(variable, None) is not None:
                if not hasattr(self, SolarHRRR.DISTRIBUTION_KEY):
                    setattr(self, SolarHRRR.DISTRIBUTION_KEY, {})

                getattr(self, SolarHRRR.DISTRIBUTION_KEY)[variable] = data[variable]

    def calculate_wind(self, data):
        """
        Calculate the wind speed and wind direction.

        Args:
            data: Loaded data from weather_forecast_retrieval
        """
        dataframe_options = dict(
            index=data["air_temp"].index, columns=data["air_temp"].columns
        )
        wind_speed = np.empty_like(data["air_temp"].values)
        wind_speed[:] = np.nan
        wind_direction = np.empty_like(data["air_temp"].values)
        wind_direction[:] = np.nan

        if self._load_wind:
            self._logger.debug("Loading wind_speed and wind_direction")

            wind_speed = np.sqrt(data["wind_u"] ** 2 + data["wind_v"] ** 2)

            wind_direction = np.degrees(np.arctan2(data["wind_v"], data["wind_u"]))
            ind = wind_direction < 0
            wind_direction[ind] = wind_direction[ind] + 360

        self.wind_speed = pd.DataFrame(wind_speed, **dataframe_options)
        self.wind_direction = pd.DataFrame(wind_direction, **dataframe_options)

    def check_cloud_factor(self):
        """
        Check the cloud factor when in the timestep mode.
        This will fill NaN values as they happen by linearly
        interpolating from the last hour (i.e. copy it). This
        is similar to how `get_hrrr_cloud` with the difference
        being that it won't interpolate over the entire night.
        """

        if self.cloud_factor_memory is None:
            self.cloud_factor_memory = self.cloud_factor

            # if initial cloud factor is at night, default to clear sky
            if self.cloud_factor_memory.isnull().values.any():
                self.cloud_factor_memory[:] = 1

        else:
            self.cloud_factor_memory = pd.concat(
                [self.cloud_factor_memory.tail(1), self.cloud_factor]
            )

        self.cloud_factor_memory = self.cloud_factor_memory.interpolate(
            method="linear"
        ).ffill()

        if self.cloud_factor_memory.isnull().values.any():
            self._logger.error("There are NaN values in the cloud factor")

        self.cloud_factor = self.cloud_factor_memory.tail(1)

    def get_metadata(self):
        """
        Load the metadata from HRRR grib files and set the attribute on the class.
        This value is later retrieved when loading the interpolation algorithm
        """
        self.metadata = FileLoader(
            external_logger=self._logger,
            file_dir=self.config["hrrr_directory"],
            forecast_hour=self.config["hrrr_forecast_hour"],
        ).get_metadata(
            date=self.start_date,
            bbox=self.bbox,
            utm_zone_number=self.topo.zone_number,
        )
