"""
The module :mod:`~smrf.framework.model_framework` acts as a wrapper to execute the
configured forcing data distribution.

Example:
    The following examples shows the most generic method of running SMRF. These
    commands will generate all the forcing data required to run iSnobal.  A
    complete example can be found in run_smrf.py

    >>> import smrf
    >>> configFile = '/path/to/smrf.ini'
    >>> s = smrf.framework.SMRF(configFile) # initialize SMRF
    >>> s.load_topo() # load topo data
    >>> s.create_distribution() # initialize the distribution
    >>> s.initialize_output() # initialize the outputs if desired
    >>> s.load_data() # load weather data  and station metadata
    >>> s.distribute_data() # distribute

"""

import logging
import os
import sys
from datetime import datetime
from os.path import abspath, join

import netCDF4
import numpy as np
import pandas as pd
import pytz
from inicheck.config import UserConfig
from inicheck.output import generate_config, print_config_report
from inicheck.tools import check_config, get_user_config
from smrf.distribute import (
    AirTemperature, VaporPressure, Wind, Precipitation, Solar, Albedo, CloudFactor,
    Thermal, ThermalHRRR, SoilTemperature
)
from smrf.data import GriddedInput, InputData, InputGribHRRR, Topo
from smrf.envphys import sunang
from smrf.framework import art, logger
from smrf.output import output_netcdf
from smrf.utils.utils import backup_input, date_range, getqotw
from topocalc.shade import shade


class SMRF:
    """
    SMRF - Spatial Modeling for Resources Framework
    """

    def __init__(self, config, external_logger=None):
        # read the config file and store
        if isinstance(config, str):
            if not os.path.isfile(config):
                raise Exception(
                    "Configuration file does not exist --> {}".format(config)
                )
            self.configFile = config

            # Read in the original users config
            ucfg = get_user_config(config, modules="smrf")

        elif isinstance(config, UserConfig):
            ucfg = config
            self.configFile = config.filename

        else:
            raise Exception(
                "Config passed to SMRF is neither file name nor  UserConfig instance"
            )
        # start logging
        if external_logger is None:
            self.smrf_logger = logger.SMRFLogger(ucfg.cfg["system"])
            self._logger = logging.getLogger(__name__)
        else:
            self._logger = external_logger

        # add the title
        self.title(2)

        # Make the output directory if it do not exist
        out = ucfg.cfg["output"]["out_location"]
        os.makedirs(out, exist_ok=True)

        # Check the user config file for errors and report issues if any
        self._logger.info("Checking config file for issues...")
        warnings, errors = check_config(ucfg)
        print_config_report(warnings, errors, logger=self._logger)
        self.ucfg = ucfg
        self.config = self.ucfg.cfg

        # Exit SMRF if config file has errors
        if len(errors) > 0:
            self._logger.error(
                "Errors in the config file. See configuration status report above."
            )
            sys.exit()

        # Write the config file to the output dir
        full_config_out = abspath(join(out, "config.ini"))

        self._logger.info("Writing config file with full options.")
        generate_config(self.ucfg, full_config_out)

        # Process the system variables
        for k, v in self.config["system"].items():
            setattr(self, k, v)

        self._setup_date_and_time()

        # need to align date time
        if "date_method_start_decay" in self.config[Albedo.DISTRIBUTION_KEY].keys():
            self.config[Albedo.DISTRIBUTION_KEY]["date_method_start_decay"] = self.config[
                Albedo.DISTRIBUTION_KEY
            ]["date_method_start_decay"].replace(tzinfo=self.time_zone)
            self.config[Albedo.DISTRIBUTION_KEY]["date_method_end_decay"] = self.config[
                Albedo.DISTRIBUTION_KEY
            ]["date_method_end_decay"].replace(tzinfo=self.time_zone)

        # Add thread configuration to all distribute sections. Used by DK method.
        for section in [
            AirTemperature.DISTRIBUTION_KEY,
            VaporPressure.DISTRIBUTION_KEY,
            Precipitation.DISTRIBUTION_KEY,
            CloudFactor.DISTRIBUTION_KEY,
            Thermal.DISTRIBUTION_KEY,
            Wind.DISTRIBUTION_KEY,
        ]:
            self.config[section]["threads"] = self.config["system"]["threads"]

        # if a gridded dataset will be used
        self.forecast_flag = False
        self.gridded = True if GriddedInput.TYPE in self.config else False
        self.load_hrrr = False
        if self.gridded:
            self.load_hrrr = self.config[GriddedInput.TYPE]["data_type"] in [
                InputGribHRRR.DATA_TYPE
            ]

        now = datetime.now().astimezone(self.time_zone)
        if (self.start_date > now and not self.gridded) or (
            self.end_date > now and not self.gridded
        ):
            raise ValueError(
                "A date set in the future can only be used with WRF generated data!"
            )

        self.output_variables = set(self.config["output"]["variables"])
        self._logger.info(
            "Configured output variables: \n {}".format(", ".join(self.output_variables))
        )
        self.output_writer = None

        self.distribute = {}

        if self.config["system"]["qotw"]:
            self._logger.info(getqotw())

        # Initialize the distribute dict
        self._logger.info("Started SMRF --> %s" % now)
        self._logger.info("Model start --> %s" % self.start_date)
        self._logger.info("Model end --> %s" % self.end_date)
        self._logger.info("Number of time steps --> %i" % self.time_steps)

    def _setup_date_and_time(self):
        self.time_zone = pytz.timezone(self.config["time"]["time_zone"])
        is_utz = self.time_zone == pytz.UTC

        # Get the time section utils
        self.start_date = pd.to_datetime(self.config["time"]["start_date"], utc=is_utz)
        self.end_date = pd.to_datetime(self.config["time"]["end_date"], utc=is_utz)

        if not is_utz:
            self.start_date = self.start_date.tz_localize(self.time_zone)
            self.end_date = self.end_date.tz_localize(self.time_zone)

        # Get the time steps correctly in the time zone
        self.date_time = date_range(
            self.start_date,
            self.end_date,
            self.config["time"]["time_step"],
            self.time_zone,
        )
        self.time_steps = len(self.date_time)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Provide some logging info about when SMRF was closed
        """

        self._logger.info("SMRF closed --> %s" % datetime.now())
        logging.shutdown()

    def load_topo(self):
        """
        Load the information from the configFile in the ['topo'] section. See
        :func:`smrf.data.loadTopo.Topo` for full description.
        """

        self.topo = Topo(self.config["topo"])

    def create_distribution(self):
        """
        This initializes the distribution classes based on the configFile
        sections for each variable.
        :func:`~smrf.framework.model_framework.SMRF.create_distribution`
        will initialize the variables within the :func:`smrf.distribute`
        package and insert into a dictionary 'distribute' with variable names
        as the keys.

        Variables that are initialized are:
            * :func:`Air temperature <smrf.distribute.ta>`
            * :func:`Vapor pressure <smrf.distribute.vp>`
            * :func:`Wind speed and direction <smrf.distribute.Wind>`
            * :func:`Precipitation <smrf.distribute.ppt>`
            * :func:`Albedo <smrf.distribute.Albedo>`
            * :func:`Solar radiation <smrf.distribute.Solar>`
            * :func:`Thermal radiation <smrf.distribute.Thermal>`
            * :func:`Soil Temperature <smrf.distribute.ts>`
        """
        # Air temperature and vapor pressure
        # Always process air temperature and vapor pressure together since
        # they depend on each other
        if (
            AirTemperature.is_requested(self.output_variables) or
            VaporPressure.is_requested(self.output_variables)
        ):
            self.distribute[AirTemperature.DISTRIBUTION_KEY] = AirTemperature(self.config)
            self.distribute[VaporPressure.DISTRIBUTION_KEY] = VaporPressure(self.config)

        # Wind
        if Wind.is_requested(self.output_variables):
            self.distribute[Wind.DISTRIBUTION_KEY] = Wind(self.config)

        # Precipitation
        if Precipitation.is_requested(self.output_variables):
            # Need air temp and vapor pressure for precip phase
            self.distribute[AirTemperature.DISTRIBUTION_KEY] = AirTemperature(self.config)
            self.distribute[VaporPressure.DISTRIBUTION_KEY] = VaporPressure(self.config)

            if self.config[Precipitation.DISTRIBUTION_KEY]["precip_rescaling_model"] == "winstral":
                self.distribute[Wind.DISTRIBUTION_KEY] = Wind(self.config)

            self.distribute[Precipitation.DISTRIBUTION_KEY] = Precipitation(
                self.config,
                self.start_date,
                self.config["time"]["time_step"],
            )

        # Cloud_factor
        if CloudFactor.is_requested(self.output_variables):
            self.distribute[CloudFactor.DISTRIBUTION_KEY] = CloudFactor(self.config)

        # Solar radiation; requires albedo and clouds
        if (
            Solar.is_requested(self.output_variables) or
            Albedo.is_requested(self.output_variables)
        ):
            # Need precip for albedo:
            self.distribute[Precipitation.DISTRIBUTION_KEY] = Precipitation(
                self.config,
                self.start_date,
                self.config["time"]["time_step"],
            )
            # Need clouds for solar, either use external one or add to distributed list
            if "hrrr_cloud" not in self.output_variables:
                self.distribute[CloudFactor.DISTRIBUTION_KEY] = CloudFactor(self.config)

            self.distribute[Albedo.DISTRIBUTION_KEY] = Albedo(self.config)
            self.distribute[Solar.DISTRIBUTION_KEY] = Solar(self.config)
        else:
            self._logger.info("Using HRRR solar in iSnobal")

        # Thermal radiation
        if Thermal.is_requested(self.output_variables):
            # Need air temperature and vapor pressure
            self.distribute[AirTemperature.DISTRIBUTION_KEY] = AirTemperature(self.config)
            self.distribute[VaporPressure.DISTRIBUTION_KEY] = VaporPressure(self.config)

            # Need clouds for solar, either use external one or add to distributed list
            if "hrrr_cloud" not in self.output_variables:
                self.distribute[CloudFactor.DISTRIBUTION_KEY] = CloudFactor(self.config)
            else:
                self._logger.info("Using HRRR cloud file for thermal.")

            self.distribute[Thermal.DISTRIBUTION_KEY] = Thermal(self.config)
        elif ThermalHRRR.INI_VARIABLE in self.output_variables:
            self.distribute[AirTemperature.DISTRIBUTION_KEY] = AirTemperature(self.config)

            # Trigger loading of longwave from HRRR
            self.config["gridded"].setdefault(
                InputGribHRRR.GDAL_VARIABLE_KEY, []
            ).append(ThermalHRRR.GRIB_NAME)

            self.distribute[ThermalHRRR.DISTRIBUTION_KEY] = ThermalHRRR()

            # Also swap out the ini variable to treat running HRRR as the standard
            # 'thermal' variable
            self.output_variables.remove(ThermalHRRR.INI_VARIABLE)
            self.output_variables.add(Thermal.DISTRIBUTION_KEY)

        # Soil temperature
        self.distribute[SoilTemperature.DISTRIBUTION_KEY] = SoilTemperature(self.config)

    def load_data(self):
        """
        Load the measurement point data for distributing to the DEM,
        must be called after the distributions are initialized. Currently, data
        can be loaded from two different sources:

            * :func:`CSV files <smrf.data.loadData.wxdata>`
            * :func:`Gridded data source (WRF) <smrf.data.loadGrid.grid>`

        After loading, :func:`~smrf.framework.mode_framework.SMRF.loadData`
        will call :func:`smrf.framework.model_framework.find_pixel_location`
        to determine the pixel locations of the point measurements and filter
        the data to the desired stations if CSV files are used.
        """

        self.data = InputData(self.config, self.start_date, self.end_date, self.topo)

        # Pre-filter the data to the desired stations in each [variable] section
        self._logger.debug("Filter data to those specified in each variable section")
        for module in self.distribute.values():
            # Skip variable classes that don't work with stations such as HRRRThermal
            if not hasattr(module, "stations"):
                continue

            for variable in module.LOADED_DATA:
                # Check to find the matching stations
                data = getattr(self.data, variable, pd.DataFrame())
                if module.stations is not None:
                    match = data.columns.isin(module.stations)
                    sta_match = data.columns[match]

                    # Update the dataframe and the distribute stations
                    module.stations = sta_match.tolist()
                    setattr(self.data, variable, data[sta_match])

                else:
                    module.stations = data.columns.tolist()

        # Does the user want to create a CSV copy of the station data used.
        if self.config["output"]["input_backup"]:
            self._logger.info("Backing up input data...")
            backup_input(self.data, self.ucfg)

    def distribute_data(self):
        """
        Distribute the measurement point data for all variables in serial. Each
        variable is initialized first using the :func:`smrf.data.loadTopo.Topo`
        instance and the metadata loaded from
        :func:`~smrf.framework.model_framework.SMRF.loadData`.
        The function distributes over each time step, all the variables below.

        Steps performed:
            1. Sun angle for the time step
            2. Illumination angle
            3. Air temperature
            4. Vapor pressure
            5. Wind direction and speed
            6. Precipitation
            7. Cloud Factor
            8. Solar radiation
            9. Thermal radiation
            10. Soil temperature
            11. Output time step if needed
        """
        # Initialize method for each distribute module
        for v in self.distribute:
            # TODO: Marks 2017 requires the raw parsed data to be passed instead of
            #       just the metadata.
            if v == Precipitation.DISTRIBUTION_KEY:
                self.distribute[v].initialize(self.topo, self.data)
            else:
                self.distribute[v].initialize(self.topo, self.data.metadata)

        # -------------------------------------
        # Distribute the data
        for output_count, t in enumerate(self.date_time):
            startTime = datetime.now()

            self.distribute_single_timestep(t)
            self.output(t)

            telapsed = datetime.now() - startTime
            self._logger.debug(
                "{0:.2f} seconds for time step".format(telapsed.total_seconds())
            )

        self.forcing_data = 1

    def distribute_single_timestep(self, t):
        self._logger.info("Distributing time step {}".format(t))

        if self.load_hrrr:
            self.data.load_class.load_timestep(t)
            self.data.set_variables()

        # Air temperature
        if AirTemperature.DISTRIBUTION_KEY in self.distribute:
            self.distribute[AirTemperature.DISTRIBUTION_KEY].distribute(
                self.data.air_temp.loc[t]
            )

        # Vapor pressure
        if VaporPressure.DISTRIBUTION_KEY in self.distribute:
            self.distribute[VaporPressure.DISTRIBUTION_KEY].distribute(
                self.data.vapor_pressure.loc[t],
                self.distribute[AirTemperature.DISTRIBUTION_KEY].air_temp,
            )

        # Wind_speed and wind_direction
        if Wind.DISTRIBUTION_KEY in self.distribute:
            self.distribute[Wind.DISTRIBUTION_KEY].distribute(
                self.data.wind_speed.loc[t], self.data.wind_direction.loc[t], t
            )

        # Precipitation
        if Precipitation.DISTRIBUTION_KEY in self.distribute:
            # Get arguments for wind when 'winstral' rescaling is requested
            if self.config["precip"]["precip_rescaling_model"] == "winstral":
                try:
                    wind_args = dict(
                        wind_direction=self.distribute[Wind.DISTRIBUTION_KEY].wind_direction,
                        dir_round_cell=self.distribute[
                            Wind.DISTRIBUTION_KEY
                        ].wind_model.dir_round_cell,
                        wind_speed=self.distribute[Wind.DISTRIBUTION_KEY].wind_speed,
                        cell_maxus=self.distribute[Wind.DISTRIBUTION_KEY].wind_model.cellmaxus,
                    )
                except AttributeError:
                    self._logger.error(
                        "Required wind argument for precipitation interpolation"
                        " not found. Please add 'wind' as output variable in the"
                        " .ini file"
                    )
            else:
                wind_args = dict()

            self.distribute[Precipitation.DISTRIBUTION_KEY].distribute(
                self.data.precip.loc[t],
                self.distribute[VaporPressure.DISTRIBUTION_KEY].dew_point,
                self.distribute[VaporPressure.DISTRIBUTION_KEY].precip_temp,
                self.distribute[AirTemperature.DISTRIBUTION_KEY].air_temp,
                t,
                self.data.wind_speed.loc[t],
                self.data.air_temp.loc[t],
                **wind_args,
            )

        # Cloud_factor
        if CloudFactor.DISTRIBUTION_KEY in self.distribute:
            self.distribute[CloudFactor.DISTRIBUTION_KEY].distribute(
                self.data.cloud_factor.loc[t]
            )
            cloud_factor = self.distribute[CloudFactor.DISTRIBUTION_KEY].cloud_factor
        elif "hrrr_cloud" in self.output_variables:
            try:
                with netCDF4.Dataset(
                    self.config["output"]["out_location"] + "/cloud_factor.nc"
                ) as cloud_data:
                    from cftime import num2date

                    cloud_date_times = cloud_data["time"]
                    cloud_dates = num2date(
                        cloud_date_times[:],
                        units=cloud_date_times.units,
                        calendar=cloud_date_times.calendar,
                        only_use_cftime_datetimes=False,
                    )
                    cloud_dates = [
                        date.replace(tzinfo=self.time_zone).timestamp()
                        for date in cloud_dates
                    ]
                    cloud_factor = cloud_data["TCDC"][cloud_dates.index(t.timestamp())]
            except FileNotFoundError:
                self._logger.error(
                    "Thermal or Solar were requested as output, but either"
                    " the 'cloud_factor' needs to be in the .ini file or a"
                    " cloud_factor.nc file has to be supplied in the output"
                    " folder location."
                )
                sys.exit()

        # Solar
        if Solar.DISTRIBUTION_KEY in self.distribute:
            # Sun angle for time step
            cosz, azimuth, rad_vec = sunang.sunang(
                t.astimezone(pytz.utc), self.topo.basin_lat, self.topo.basin_long
            )

            # Illumination angle
            illum_ang = None
            if cosz > 0:
                illum_ang = shade(self.topo.sin_slope, self.topo.aspect, azimuth, cosz)

            # Albedo
            self.distribute[Albedo.DISTRIBUTION_KEY].distribute(
                t, illum_ang, self.distribute[Precipitation.DISTRIBUTION_KEY].storm_days
            )

            # Net Solar
            self.distribute[Solar.DISTRIBUTION_KEY].distribute(
                t,
                cloud_factor,
                illum_ang,
                cosz,
                azimuth,
                self.distribute[Albedo.DISTRIBUTION_KEY].albedo_vis,
                self.distribute[Albedo.DISTRIBUTION_KEY].albedo_ir,
            )

        # Thermal radiation
        if (
            Thermal.DISTRIBUTION_KEY in self.distribute and
            isinstance(self.distribute[Thermal.DISTRIBUTION_KEY], Thermal)
        ):
            self.distribute[Thermal.DISTRIBUTION_KEY].distribute(
                t,
                self.distribute[AirTemperature.DISTRIBUTION_KEY].air_temp,
                self.distribute[VaporPressure.DISTRIBUTION_KEY].vapor_pressure,
                self.distribute[VaporPressure.DISTRIBUTION_KEY].dew_point,
                cloud_factor,
            )
        elif (
            Thermal.DISTRIBUTION_KEY in self.distribute and
            isinstance(self.distribute[ThermalHRRR.DISTRIBUTION_KEY], ThermalHRRR)
        ):
            self.distribute[ThermalHRRR.DISTRIBUTION_KEY].distribute(
                t,
                self.data.thermal,
                self.distribute[AirTemperature.DISTRIBUTION_KEY].air_temp,
            )

        # Soil temperature
        self.distribute[SoilTemperature.DISTRIBUTION_KEY].distribute()

    def initialize_output(self):
        """
        Initialize the output files based on the configFile section ['output'].
        Currently only :func:`NetCDF files
        <smrf.output.output_netcdf.OutputNetcdf>` are supported.
        """
        out_location = self.config["output"]["out_location"]
        variable_dict = {}

        for module_variable, module in self.distribute.items():
            requested_variables = self.output_variables & module.OUTPUT_OPTIONS

            for variable in requested_variables:
                variable_dict[variable] = {
                    "variable": variable,
                    "module": module,
                    "out_location": join(out_location, variable),
                    "nc_attributes": module.OUTPUT_VARIABLES[variable],
                }

        # determine what type of file to output
        if self.config["output"]["file_type"].lower() == "netcdf":
            self.output_writer = output_netcdf.OutputNetcdf(
                variable_dict, self.topo, self.config["time"], self.config["output"]
            )
        else:
            raise Exception("Could not determine type of file for output")

    def output(self, current_time_step: datetime) -> None:
        """
        Output the forcing data or model outputs for the current_time_step.

        Args:
            current_time_step (date_time): The current time step
        """
        output_count = self.date_time.index(current_time_step)

        # Only output according to the user specified value,
        # or if it is the end.
        if (output_count % self.config["output"]["frequency"] == 0) or (
            output_count == len(self.date_time)
        ):
            # Get the output variables then pass to the function
            for variable in self.output_writer.variables_info.values():
                 # Get the data from the distribution class
                data = getattr(self.distribute[variable["module"].DISTRIBUTION_KEY], variable["variable"])

                if data is None:
                    data = np.zeros((self.topo.ny, self.topo.nx))

                # output the time step
                self._logger.debug("Outputting {0}".format(variable["module"]))
                self.output_writer.output(variable["variable"], data, current_time_step)

    def title(self, option):
        """
        A little title to go at the top of the logger file
        """

        if option == 1:
            title = art.title1

        elif option == 2:
            title = art.title2

        for line in title:
            self._logger.info(line)


def run_smrf(config, external_logger=None):
    """
    Function that runs smrf how it should be operate for full runs.

    Args:
        config: string path to the config file or inicheck UserConfig instance
        external_logger: Logging instance
    """
    start = datetime.now()
    # initialize
    with SMRF(config, external_logger) as s:
        # load topo data
        s.load_topo()

        # initialize the distribution
        s.create_distribution()

        # initialize the outputs if desired
        s.initialize_output()

        # load weather data and station metadata
        s.load_data()

        # distribute
        s.distribute_data()

        s._logger.info(datetime.now() - start)

    return s
