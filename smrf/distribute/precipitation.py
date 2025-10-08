from datetime import timedelta

import netCDF4 as nc
import numpy as np
from dateutil.parser import parse
from smrf.envphys import Snow, precip, storms
from smrf.utils import utils

from .variable_base import VariableBase


class Precipitation(VariableBase):
    """
    The instantaneous precipitation typically has a positive trend with
    elevation due to orographic effects. However, the precipitation
    distribution can be further complicated for storms that have isolated
    impact at only a few measurement locations, for example thunderstorms
    or small precipitation events. Some distribution methods may be better
    suited than others for capturing the trend of these small events with
    multiple stations that record no precipitation may have a negative impact
    on the distribution.

    The precipitation phase, or the amount of precipitation falling as rain or
    snow, can significantly alter the energy and mass balance of the snowpack,
    either leading to snow accumulation or inducing melt :cite:`Marks&al:1998`
    :cite:`Kormos&al:2014`. The precipitation phase and initial snow density
    estimated using a variety of models that can be set in the configuration
    file.

    For more information on the available models, checkout
    :mod:`~smrf.envphys.snow`.

    After the precipitation phase is calculated, the storm information can be
    determined. The spatial resolution for which storm definitions are applied
    is based on the snow model that's selected.

    The time since last storm is based on an accumulated precipitation mass
    threshold, the time elapsed since it last snowed, and the precipitation
    phase.  These factors determine the start and end time of a storm that
    has produced enough precipitation as snow to change the surface albedo.

    Args:
        config: The [precip] section of the configuration file
        time_step: The time step in minutes of the data, defaults to 60
    """

    # TODO: https://github.com/iSnobal/smrf/issues/32
    DISTRIBUTION_KEY = "precip"

    # these are variables that can be output
    OUTPUT_VARIABLES = {
        DISTRIBUTION_KEY: {
            "units": "mm",
            "standard_name": "precipitation_mass",
            "long_name": "Precipitation mass",
        },
        "percent_snow": {
            "units": "%",
            "standard_name": "percent_snow",
            "long_name": "Percent of precipitation as snow",
        },
        "snow_density": {
            "units": "kg/m3",
            "standard_name": "snow_density",
            "long_name": "Precipitation snow density",
        },
        "storm_days": {
            "units": "day",
            "standard_name": "days_since_last_storm",
            "long_name": "Days since the last storm",
        },
        "storm_total": {
            "units": "mm",
            "standard_name": "precipitation_mass_storm",
            "long_name": "Precipitation mass for the storm period",
        },
    }

    def __init__(self, config, topo, start_date, time_step=60):
        super().__init__(config=config, topo=topo)

        self.time_step = float(time_step)
        self.start_date = start_date

        # Possible output variables
        self.precip = None
        self.percent_snow = None
        self.snow_density = None
        self.storm_days = None
        self.storm_total = None

    def initialize(self, data):
        """
        See :mod:`smrf.distribute.ImageData.initialize` for documentation on the base
        initialization.

        Precipitation is the only class that needs both the data and metadata to
        do the below modifications.
        """
        super().initialize(data.metadata)

        self.precip = np.zeros((self.topo.ny, self.topo.nx))
        self.percent_snow = np.zeros((self.topo.ny, self.topo.nx))
        self.snow_density = np.zeros((self.topo.ny, self.topo.nx))
        self.storm_days = np.zeros((self.topo.ny, self.topo.nx))
        self.storm_total = np.zeros((self.topo.ny, self.topo.nx))

        # Assign storm_days array if given
        if self.config.get("storm_days_restart", None):
            self._logger.debug(
                "Reading {} from {}".format(
                    "storm_days", self.config["storm_days_restart"]
                )
            )
            f = nc.Dataset(self.config["storm_days_restart"])
            f.set_always_mask(False)

            if "storm_days" in f.variables:
                t = f.variables["time"]
                t_max = t[:].max()
                time = nc.num2date(
                    t_max,
                    t.getncattr("units"),
                    t.getncattr("calendar"),
                    only_use_cftime_datetimes=False,
                    only_use_python_datetimes=True,
                )
                # Check whether the last storm day entry and the start
                # of this run is an hour apart (3600 seconds)
                max_time = time.replace(tzinfo=self.start_date.tzinfo)
                delta_seconds = self.start_date.to_pydatetime() - parse(str(max_time))

                # Python timedelta are handled in seconds and days
                if delta_seconds > timedelta(seconds=(self.time_step * 60)):
                    self._logger.warning("Invalid storm_days input! Setting to 0.0")
                    self.storm_days = np.zeros((self.topo.ny, self.topo.nx))

                else:
                    # start at index of storm_days - 1
                    self.storm_days = f.variables["storm_days"][t_max]
            else:
                self._logger.error(
                    "Variable storm_days not in {}".format(
                        self.config["storm_days_restart"]
                    )
                )

            f.close()

        self.ppt_threshold = self.config["storm_mass_threshold"]
        self.nasde_model = self.config["new_snow_density_model"]

        self._logger.info(
            "Using {0} for the new accumulated snow density model: ".format(
                self.nasde_model
            )
        )

        if self.nasde_model == "marks2017":
            # Time steps needed to end a storm definition
            self.time_to_end_storm = self.config["marks2017_timesteps_to_end_storms"]

            self.storm_total = np.zeros((self.topo.ny, self.topo.nx))

            self.storms = []
            self.time_steps_since_precip = 0
            self.storming = False

            # Clip and adjust the precip data so that there is only precip
            # during the storm and ad back in the missing data to conserve mass
            if self.stations is not None:
                data.precip = data.precip[self.stations]

            self.storms, storm_count = storms.tracking_by_station(
                data.precip,
                mass_thresh=self.ppt_threshold,
                steps_thresh=self.time_to_end_storm,
            )
            self.corrected_precip = storms.clip_and_correct(
                data.precip, self.storms, stations=self.stations
            )

            if storm_count != 0:
                self._logger.info("Identified Storms:\n{0}".format(self.storms))
                self.storm_id = 0
                self._logger.info("Estimated number of storms: {0}".format(storm_count))

            else:
                if (data.precip.sum() > 0).any():
                    self.storm_id = np.nan
                    self._logger.warning(
                        "Zero events triggered a storm "
                        " definition, None of the precip will"
                        " be used in this run."
                    )

        # if redistributing due to wind
        if self.config["precip_rescaling_model"] == "winstral":
            self._tbreak_file = nc.Dataset(self.config["winstral_tbreak_netcdf"], "r")
            self.tbreak = self._tbreak_file.variables["tbreak"][:]
            self.tbreak_direction = self._tbreak_file.variables["direction"][:]
            self._tbreak_file.close()
            self._logger.debug(
                "Read data from {}".format(self.config["winstral_tbreak_netcdf"])
            )

            # get the veg values
            matching = [s for s in self.config.keys() if "winstral_veg_" in s]
            v = {}
            for m in matching:
                if m != "winstral_veg_default":
                    ms = m.split("_")
                    # v[ms[1]] = float(self.config[m])
                    if type(self.config[m]) == list:
                        v[ms[1]] = float(self.config[m][0])
                    else:
                        v[ms[1]] = float(self.config[m])
            self.veg = v

    def distribute(
        self,
        data,
        dpt,
        precip_temp,
        ta,
        time,
        wind,
        temp,
        wind_direction=None,
        dir_round_cell=None,
        wind_speed=None,
        cell_maxus=None,
    ):
        """
        Distribute given a Panda's dataframe for a single time step. Calls
        :mod:`smrf.distribute.ImageData._distribute`.

        The following steps are taken when distributing precip, if there is
        precipitation measured:

        1. Distribute the instantaneous precipitation from the measurement data
        2. Determine the distributed precipitation phase based on the
            precipitation temperature
        3. Calculate the storms based on the accumulated mass, time since last
            storm, and precipitation phase threshold

        Args:
            data:           Pandas dataframe for a single time step from precip
            dpt:            dew point numpy array that will be used for
            precip_temp:    numpy array of the precipitation temperature
            ta:             air temp numpy array
            time:           pass in the time were are currently on
            wind:           station wind speed at time step
            temp:           station air temperature at time step
            wind_direction: numpy array for simulated wind direction
            dir_round_cell: numpy array for wind direction in discreet
                            increments for referencing maxus at a specific
                            direction
            wind_speed:     numpy array of wind speed
            cell_maxus:     numpy array for maxus at correct wind directions
        """

        self._logger.debug("%s Distributing all precip" % data.name)
        if self.stations is not None:
            data = data[self.stations]

        if self.config["distribution"] != "grid":
            if self.nasde_model == "marks2017":
                # Adjust the precip for undercatch
                if self.config["station_adjust_for_undercatch"]:
                    self._logger.debug(
                        "%s Adjusting precip for undercatch..." % data.name
                    )
                    self.corrected_precip.loc[time] = precip.adjust_for_undercatch(
                        self.corrected_precip.loc[time],
                        wind,
                        temp,
                        self.config,
                        self.metadata,
                    )

                # Use the clipped and corrected precip
                self.distribute_for_marks2017(
                    self.corrected_precip.loc[time], precip_temp, ta, time
                )

            else:
                # Adjust the precip for undercatch
                if self.config["station_adjust_for_undercatch"]:
                    self._logger.debug(
                        "%s Adjusting precip for undercatch..." % data.name
                    )
                    data = precip.adjust_for_undercatch(
                        data, wind, temp, self.config, self.metadata
                    )

                self.distribute_for_susong1999(data, precip_temp, time)
        else:
            self.distribute_for_susong1999(data, precip_temp, time)

        # redistribute due to wind to account for drifting
        if self.config["precip_rescaling_model"] == "winstral":
            self._logger.debug("%s Redistributing due to wind" % data.name)
            if np.any(dpt < 0.5):
                self.precip = precip.dist_precip_wind(
                    self.precip,
                    dpt,
                    wind_direction,
                    dir_round_cell,
                    wind_speed,
                    cell_maxus,
                    self.tbreak,
                    self.tbreak_direction,
                    self.veg_type,
                    self.veg,
                    self.config,
                )

        # Mask the precip temperature to where we have any precipitation amounts.
        # This reduces the amount of data saved on disk.
        precip_temp[self.precip == 0] = np.nan

    def distribute_for_marks2017(self, data, precip_temp, ta, time):
        """
        Specialized distribute function for working with the new accumulated
        snow density model Marks2017 requires storm total and a corrected
        precipitation as to avoid precip between storms.
        """
        if data.sum() > 0.0:
            # Check for time in every storm
            for i, s in self.storms.iterrows():
                storm_start = s["start"]
                storm_end = s["end"]

                if time >= storm_start and time <= storm_end:
                    # establish storm info
                    self.storm_id = i
                    storm = self.storms.iloc[self.storm_id]

                    # Subset
                    if self.stations is not None:
                        storm = storm[self.stations]
                    else:
                        storm = storm.drop("start").drop("end")

                    self.storming = True
                    break
                else:
                    self.storming = False

            self._logger.debug("Storming? {0}".format(self.storming))
            self._logger.debug("Current Storm ID = {0}".format(self.storm_id))

            # distribute data and set the min/max
            self._distribute(data, zeros=None)
            self.precip = utils.set_min_max(self.precip, self.min, self.max)

            if time == storm_start:
                # Entered into a new storm period distribute the storm total
                self._logger.debug(
                    "{0} Entering storm #{1}".format(data.name, self.storm_id + 1)
                )
                if precip_temp.min() < 2.0:
                    self._logger.debug(
                        """Distributing Total Precip
                                        for Storm #{0}""".format(self.storm_id + 1)
                    )
                    self._distribute(storm.astype(float), other_attribute="storm_total")
                    self.storm_total = utils.set_min_max(
                        self.storm_total, self.min, self.max
                    )

            if self.storming and precip_temp.min() < 2.0:
                self._logger.debug(
                    """Calculating new snow density for
                                    storm #{0}""".format(self.storm_id + 1)
                )
                # determine the precip phase and den
                snow_den, perc_snow = Snow.phase_and_density(
                    precip_temp, self.precip, nasde_model=self.nasde_model
                )

            else:
                snow_den = np.zeros(self.precip.shape)
                perc_snow = np.zeros(self.precip.shape)

            # calculate decimal days since last storm
            self.storm_days = storms.time_since_storm_pixel(
                self.precip,
                precip_temp,
                perc_snow,
                storming=self.storming,
                time_step=self.time_step / 60.0 / 24.0,
                stormDays=self.storm_days,
                mass=self.ppt_threshold,
            )

        else:
            self.storm_days += self.time_step / 60.0 / 24.0
            self.precip = np.zeros(self.storm_days.shape)
            perc_snow = np.zeros(self.storm_days.shape)
            snow_den = np.zeros(self.storm_days.shape)

        # save the model state
        self.percent_snow = perc_snow
        self.snow_density = snow_den

    def distribute_for_susong1999(self, data, ppt_temp, _time):
        """
        Susong 1999 estimates percent snow and snow density based on
        Susong et al, (1999) :cite:`Susong&al:1999`.

        Args:
            data (pd.DataFrame): Precipitation mass data
            ppt_temp (pd.DataFrame): Precipitation temperature data
            _time : Unused
        """

        if data.sum() > 0:
            self._distribute(data)
            self.precip = utils.set_min_max(self.precip, self.min, self.max)

            # determine the precip phase and den
            snow_den, perc_snow = Snow.phase_and_density(
                ppt_temp, self.precip, nasde_model=self.nasde_model
            )

            # determine the time since last storm
            storm_days, storm_precip = storms.time_since_storm(
                self.precip,
                perc_snow,
                storm_days=self.storm_days,
                storm_precip=self.storm_total,
                time_step=self.time_step / 60 / 24,
                mass_threshold=self.ppt_threshold,
            )

            # save the model state
            self.percent_snow = perc_snow
            self.snow_density = snow_den
            self.storm_days = storm_days
            self.storm_total = storm_precip

        else:
            self.storm_days += self.time_step / 60 / 24

            # make everything else zeros
            self.precip = np.zeros(self.storm_days.shape)
            self.percent_snow = np.zeros(self.storm_days.shape)
            self.snow_density = np.zeros(self.storm_days.shape)
