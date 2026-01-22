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
import pandas as pd
import pytz
from inicheck.config import UserConfig
from inicheck.output import generate_config, print_config_report
from inicheck.tools import check_config, get_user_config
from smrf.data import GriddedInput, InputData, InputGribHRRR, Topo
from smrf.distribute import (
    AirTemperature,
    Albedo,
    CloudFactor,
    Precipitation,
    SoilTemperature,
    Solar,
    SolarHRRR,
    Thermal,
    ThermalHRRR,
    VaporPressure,
    Wind,
)
from smrf.envphys import sunang
from smrf.framework import ascii_art, logger
from smrf.output import output_netcdf
from smrf.utils.utils import backup_input, date_range, getqotw
from topocalc.illumination_angle import illumination_angle


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

        self._logger.info(ascii_art.TITLE_SM)

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

        # Gridded dataset
        self.forecast_flag = False
        self.gridded = True if GriddedInput.TYPE in self.config else False
        self.load_hrrr = False
        if self.gridded:
            self.load_hrrr = self.config[GriddedInput.TYPE]["data_type"] in [
                InputGribHRRR.DATA_TYPE
            ]
            # List of HRRR variables loaded with GDAL
            self.config[GriddedInput.TYPE].setdefault(
                InputGribHRRR.GDAL_VARIABLE_KEY, []
            )

        self.output_variables = set(self.config["output"]["variables"])
        self._logger.info(
            "Configured output variables: \n {}".format(", ".join(self.output_variables))
        )
        self.output_writer = None

        # Initialize the distribute dict
        self.distribute = {}

        # Attribute to class that holds the loaded forcing data
        self.data = None

        if self.config["system"]["qotw"]:
            self._logger.info(getqotw())

        self._logger.info(
            "Started SMRF --> %s" % datetime.now().astimezone(self.time_zone)
        )
        self._logger.info("Number of time steps: %i" % self.time_steps)

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

    def distribute_precip(self) -> None:
        """
        Helper method to streamline enqueuing precip. This variables has a lot of
        dependencies and is required for by others such as albedo.
        """
        init_args = dict(config=self.config, topo=self.topo)

        # Need air temp and vapor pressure for precip phase
        self.distribute[AirTemperature.DISTRIBUTION_KEY] = AirTemperature(**init_args)
        self.distribute[VaporPressure.DISTRIBUTION_KEY] = VaporPressure(**init_args)

        if (
            self.config[Precipitation.DISTRIBUTION_KEY]["precip_rescaling_model"]
            == "winstral"
        ):
            self.distribute[Wind.DISTRIBUTION_KEY] = Wind(**init_args)

        self.distribute[Precipitation.DISTRIBUTION_KEY] = Precipitation(
            **init_args,
            start_date=self.start_date,
            time_step=self.config["time"]["time_step"],
        )

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
        init_args=dict(config=self.config, topo=self.topo)

        if (
            AirTemperature.is_requested(self.output_variables) or
            VaporPressure.is_requested(self.output_variables)
        ):
            self.distribute[AirTemperature.DISTRIBUTION_KEY] = AirTemperature(**init_args)
            self.distribute[VaporPressure.DISTRIBUTION_KEY] = VaporPressure(**init_args)

        # Wind
        if Wind.is_requested(self.output_variables):
            self.distribute[Wind.DISTRIBUTION_KEY] = Wind(**init_args)

        # Precipitation
        if Precipitation.is_requested(self.output_variables):
            self.distribute_precip()

        # Cloud Factor
        if CloudFactor.is_requested(self.output_variables):
            self.distribute[CloudFactor.DISTRIBUTION_KEY] = CloudFactor(**init_args)

        # Albedo
        if Albedo.is_requested(self.output_variables):
            self.distribute[Albedo.DISTRIBUTION_KEY] = Albedo(**init_args)

        # Solar radiation; requires albedo and clouds
        if Solar.is_requested(self.output_variables):
            # Need clouds for solar, either use external one or add to distributed list
            if "hrrr_cloud" not in self.output_variables:
                self.distribute[CloudFactor.DISTRIBUTION_KEY] = CloudFactor(**init_args)

            # Need precipitation for albedo (days since last storm)
            self.distribute_precip()
            self.distribute[Albedo.DISTRIBUTION_KEY] = Albedo(**init_args)

            self.distribute[Solar.DISTRIBUTION_KEY] = Solar(**init_args)
        elif SolarHRRR.is_requested(self.output_variables):
            # Need precipitation for albedo (days since last storm)
            self.distribute_precip()
            self.distribute[Albedo.DISTRIBUTION_KEY] = Albedo(**init_args)

            # Trigger loading all shortwave variables from HRRR
            self.config[GriddedInput.TYPE][InputGribHRRR.GDAL_VARIABLE_KEY] += (
                SolarHRRR.GRIB_VARIABLES
            )

            self.distribute[SolarHRRR.DISTRIBUTION_KEY] = SolarHRRR(**init_args)

            # Add the required 'net_solar' output when requesting HRRR solar
            self.output_variables.add(SolarHRRR.DEFAULT_OUTPUT)

        # Thermal radiation
        if Thermal.is_requested(self.output_variables):
            # Need air temperature and vapor pressure
            self.distribute[AirTemperature.DISTRIBUTION_KEY] = AirTemperature(**init_args)
            self.distribute[VaporPressure.DISTRIBUTION_KEY] = VaporPressure(**init_args)

            # Need clouds for solar, either use external one or add to distributed list
            if "hrrr_cloud" not in self.output_variables:
                self.distribute[CloudFactor.DISTRIBUTION_KEY] = CloudFactor(**init_args)
            else:
                self._logger.info("Using HRRR cloud file for thermal.")

            self.distribute[Thermal.DISTRIBUTION_KEY] = Thermal(**init_args)
        elif ThermalHRRR.INI_VARIABLE in self.output_variables:
            self.distribute[AirTemperature.DISTRIBUTION_KEY] = AirTemperature(**init_args)

            # Trigger loading of longwave from HRRR
            self.config[GriddedInput.TYPE][InputGribHRRR.GDAL_VARIABLE_KEY].append(
                ThermalHRRR.GRIB_NAME
            )

            self.distribute[ThermalHRRR.DISTRIBUTION_KEY] = ThermalHRRR(topo=self.topo)

            # Also swap out the ini variable to treat running HRRR as the standard
            # 'thermal' variable
            self.output_variables.remove(ThermalHRRR.INI_VARIABLE)
            self.output_variables.add(Thermal.DISTRIBUTION_KEY)

        # Soil temperature
        self.distribute[SoilTemperature.DISTRIBUTION_KEY] = SoilTemperature(**init_args)

    def load_data(self):
        """
        Load the forcing data configured in the .ini file.
        See the :py:class:`~smrf.data.loadData.InputData` class for full list of
        supported sources
        """
        self.data = InputData(
            self.config, self.start_date, self.end_date, self.topo
        ).loader

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
                self.distribute[v].initialize(self.data)
            else:
                self.distribute[v].initialize(self.data.metadata)

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

    def distribute_single_timestep(self, timestep: datetime) -> None:
        """
        Perform the distribution of the data for a single time step.

        :param timestep: Time step to process
        """
        self._logger.info("Distributing time step {}".format(timestep))

        if self.data.DATA_TYPE == InputGribHRRR.DATA_TYPE:
            self.data.load_timestep(timestep)

        # Air temperature
        if AirTemperature.DISTRIBUTION_KEY in self.distribute:
            self.distribute[AirTemperature.DISTRIBUTION_KEY].distribute(
                self.data.air_temp.loc[timestep]
            )

        # Vapor pressure
        if VaporPressure.DISTRIBUTION_KEY in self.distribute:
            self.distribute[VaporPressure.DISTRIBUTION_KEY].distribute(
                self.data.vapor_pressure.loc[timestep],
                self.distribute[AirTemperature.DISTRIBUTION_KEY].air_temp,
            )

        # Wind_speed and wind_direction
        if Wind.DISTRIBUTION_KEY in self.distribute:
            self.distribute[Wind.DISTRIBUTION_KEY].distribute(
                self.data.wind_speed.loc[timestep], self.data.wind_direction.loc[timestep], timestep
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
                self.data.precip.loc[timestep],
                self.distribute[VaporPressure.DISTRIBUTION_KEY].dew_point,
                self.distribute[VaporPressure.DISTRIBUTION_KEY].precip_temp,
                self.distribute[AirTemperature.DISTRIBUTION_KEY].air_temp,
                timestep,
                self.data.wind_speed.loc[timestep],
                self.data.air_temp.loc[timestep],
                **wind_args,
            )

        # Cloud_factor
        if CloudFactor.DISTRIBUTION_KEY in self.distribute:
            self.distribute[CloudFactor.DISTRIBUTION_KEY].distribute(
                self.data.cloud_factor.loc[timestep]
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
                    cloud_factor = cloud_data["TCDC"][cloud_dates.index(timestep.timestamp())]
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
            cos_z, azimuth, _rad_vec = sunang.sunang(
                timestep.astimezone(pytz.utc), self.topo.basin_lat, self.topo.basin_long
            )

            illumination_angles = None
            if cos_z > 0:
                illumination_angles = illumination_angle(
                    self.topo.sin_slope, self.topo.aspect, azimuth, cos_z
                )

            # Albedo
            self.distribute[Albedo.DISTRIBUTION_KEY].distribute(
                timestep,
                illumination_angles,
                self.distribute[Precipitation.DISTRIBUTION_KEY].storm_days,
            )

            if isinstance(self.distribute[Solar.DISTRIBUTION_KEY], Solar):
                # Net Solar
                self.distribute[Solar.DISTRIBUTION_KEY].distribute(
                    timestep,
                    cloud_factor,
                    illumination_angles,
                    cos_z,
                    azimuth,
                    self.distribute[Albedo.DISTRIBUTION_KEY],
                )
            elif isinstance(self.distribute[SolarHRRR.DISTRIBUTION_KEY], SolarHRRR):
                self.distribute[SolarHRRR.DISTRIBUTION_KEY].distribute(
                    timestep,
                    self.data.solar,
                    cos_z,
                    azimuth,
                    illumination_angles,
                    self.distribute[Albedo.DISTRIBUTION_KEY],
                )

        # Thermal radiation
        if Thermal.DISTRIBUTION_KEY in self.distribute:
            if isinstance(self.distribute[Thermal.DISTRIBUTION_KEY], Thermal):
                self.distribute[Thermal.DISTRIBUTION_KEY].distribute(
                    timestep,
                    self.distribute[AirTemperature.DISTRIBUTION_KEY].air_temp,
                    self.distribute[VaporPressure.DISTRIBUTION_KEY].vapor_pressure,
                    self.distribute[VaporPressure.DISTRIBUTION_KEY].dew_point,
                    cloud_factor,
                )
            elif isinstance(self.distribute[ThermalHRRR.DISTRIBUTION_KEY], ThermalHRRR):
                self.distribute[ThermalHRRR.DISTRIBUTION_KEY].distribute(
                    timestep,
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
        variable_dict = {}

        # Collect all requested variables and the module that provides them
        for module in self.distribute.values():
            requested_variables = self.output_variables & module.OUTPUT_OPTIONS

            for variable in requested_variables:
                variable_dict[variable] = module

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
            self.output_writer.output(current_time_step)


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
