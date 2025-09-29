import numpy as np
import pandas as pd

from .gridded_input import GriddedInput
from smrf.data.hrrr.file_loader import FileLoader
from smrf.distribute.wind import Wind
from smrf.distribute.wind.wind_ninja import WindNinjaModel
from smrf.envphys.solar.cloud import get_hrrr_cloud
from smrf.envphys.vapor_pressure import rh2vp


class InputGribHRRR(GriddedInput):
    """
    Load the data from the High Resolution Rapid Refresh (HRRR) model.
    The possible variables returned are dependent on the configuration
    and returned as Pandas dataframes.
    """

    DATA_TYPE = 'hrrr_grib'

    VARIABLES = [
        'air_temp',
        'vapor_pressure',
        'precip',
        'cloud_factor'
    ]
    WIND_VARIABLES = [
        'wind_speed',
        'wind_direction',
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cloud_factor_memory = None

        self._load_wind = not Wind.config_model_type(
            kwargs['config'], WindNinjaModel.MODEL_TYPE
        )

        self._calculate_cloud_factor = (
            "hrrr_cloud" not in kwargs["config"]["output"]["variables"]
        )

    @property
    def variables(self):
        if self._load_wind:
            return self.data_variables
        else:
            return self.VARIABLES

    @property
    def data_variables(self):
        return np.union1d(self.VARIABLES, self.WIND_VARIABLES)

    def load(self):
        """
        The function will take the keys and load them into the appropriate
        objects within the `grid` class.
        """
        self._logger.info(
            "Reading data from from HRRR directory: {}".format(
                self.config["hrrr_directory"]
            )
        )

        metadata, data = FileLoader(
            external_logger=self._logger,
            file_dir=self.config['hrrr_directory'],
            forecast_hour=self.config['hrrr_forecast_hour'],
            load_wind = self._load_wind,
            sixth_hour_variables=self.config['hrrr_sixth_hour_variables'],
        ).data_for_time_and_topo(
            start_date=self.start_date,
            bbox=self.bbox,
            utm_zone_number=self.topo.zone_number,
        )

        self.parse_data(metadata, data)

    def load_timestep(self, date_time):
        """Load a single time step for HRRR

        Args:
            date_time (datetime): date time to load
        """

        self.start_date = date_time
        self.load()

    def load_timestep_thread(self, date_times, data_queue):
        """Load HRRR within a thread and add the data to the data
        queue.

        Args:
            date_times (list): list of the simulation date_times
            data_queue (dict): dict of the data queues
        """

        for date_time in date_times[1:]:
            self.load_timestep(date_time)

            for variable in self.data_variables:
                data_queue[variable].put(
                    [date_time, getattr(self, variable).iloc[0]])

        self._logger.debug('Finished loading data')

    def parse_data(self, metadata, data):
        """
        Parse the data from HRRR into dataframes for SMRF

        Args:
            metadata (DataFrame): metadata DataFrame
            data (dict): dictionary of DataFrames from HRRR
        """
        # Enforce configured timezone
        for key in data.keys():
            data[key] = data[key].apply(pd.to_numeric)
            data[key] = data[key].tz_localize(tz=self.time_zone)

        self.metadata = metadata

        idx = data['air_temp'].index
        cols = data['air_temp'].columns

        self._logger.debug('Loading air_temp')
        self.air_temp = data['air_temp']

        # calculate vapor pressure
        self._logger.debug('Calculating vapor_pressure')
        vp = rh2vp(data["air_temp"].values, data["relative_humidity"].values)
        self.vapor_pressure = pd.DataFrame(vp, index=idx, columns=cols)

        self.calculate_wind(data)

        # precip
        self._logger.debug('Loading precip')
        self.precip = pd.DataFrame(data['precip_int'], index=idx, columns=cols)

        # cloud factor
        if self._calculate_cloud_factor:
            self._logger.debug('Loading solar')
            solar = pd.DataFrame(data['short_wave'], index=idx, columns=cols)
            self._logger.debug('Calculating cloud factor')
            self.cloud_factor = get_hrrr_cloud(
                solar, self.metadata,
                self.topo.basin_lat, self.topo.basin_long
            )
            self.check_cloud_factor()

    def calculate_wind(self, data):
        """
        Calculate the wind speed and wind direction.

        Args:
            data: Loaded data from weather_forecast_retrieval
        """
        dataframe_options = dict(
            index=data['air_temp'].index,
            columns=data['air_temp'].columns
        )
        wind_speed = np.empty_like(data['air_temp'].values)
        wind_speed[:] = np.nan
        wind_direction = np.empty_like(data['air_temp'].values)
        wind_direction[:] = np.nan

        if self._load_wind:
            self._logger.debug('Loading wind_speed and wind_direction')

            wind_speed = np.sqrt(data['wind_u']**2 + data['wind_v']**2)

            wind_direction = np.degrees(
                np.arctan2(data['wind_v'], data['wind_u'])
            )
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
            self._logger.error('There are NaN values in the cloud factor')

        self.cloud_factor = self.cloud_factor_memory.tail(1)
