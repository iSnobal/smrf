import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

from smrf.data.load_topo import Topo
from smrf.data.read_netcdf import ReadNetCDF
from smrf.spatial import (
    DetrendedKriging,
    Grid,
    InverseDistanceWeighted,
    Kriging,
)


class VariableBase:
    """
    Base class that will ensure all variables are distributed in the same manner.

    A child class needs to define the following constants:
      * DISTRIBUTION_KEY A unique key for distributing the class.
      * OUTPUT_VARIABLES: A dictionary defining the key as the output file name and
                         the value as a dictionary of the NetCDF variable attributes
                         that are used upon writing the file. The variable will be named
                         the same as the key (filename).
      * LOADED_DATA: A list of the input data that is loaded from the input source
                     By default, this is matches the DISTRIBUTION_KEY

    The following constants are set automatically based on the above:
      * OUTPUT_OPTIONS: A set of the variables set from the OUTPUT_VARIABLES keys

    Instance attributes:
        config:   Parsed dictionary from the configuration file for the variable
        topo:     Access to the variables stored in the Topo
        metadata: The metadata Pandas dataframe containing the station
                  information from :mod:`smrf.data.loadData` or :mod:`smrf.data.loadGrid`
        stations: The stations to be used for the variable, if set, in
                  alphabetical order
        gridded:  Boolean indicator of whether the variable is gridded or not
        max:      Maximum allowed value for the variable
        min:      Minimum allowed value for the variable
    """

    DISTRIBUTION_KEY = ""
    OUTPUT_VARIABLES = {}

    # Configuration key that is available with each section, when an external file
    # should be read.
    SOURCE_FILES = "source_files"

    def __init__(self, config: dict = None, topo: Topo = None):
        """
        Initialize the class and parse out the relevant section from the configuration
        when given. The section name of the configuration needs to match the VARIABLE
        for this class to recognize it.

        Example:
        * .ini file
           ```
           [air_temp]
           max: 100
           min: -100
           ```
        * The corresponding class
           ```
           class AirTemp(ImageData):
               VARIABLE = 'air_temp'
           ```

        :param config: Parsed configuration file
        """
        if self.DISTRIBUTION_KEY != "":
            setattr(self, self.DISTRIBUTION_KEY, None)

        self.config = None
        self.topo = topo
        self.metadata = None
        # Externally loaded forcing files if present in the configuration
        self.source_files = None

        # System wide configurations
        if config is not None:
            self.threads = config.get("system", {}).get("threads", 1)
            self.start_date = pd.to_datetime(config["time"]["start_date"]).strftime(
                "%Y%m%d"
            )
            self.time_zone = pytz.timezone(config["time"]["time_zone"])

        # Variable specific configurations
        if config and config.get(self.DISTRIBUTION_KEY, False):
            self.config = config[self.DISTRIBUTION_KEY]

            # Check of gridded interpolation
            self.gridded = self.config.get("distribution", None) == "grid"

            # List of stations that have this variable observation
            self.stations = None
            if self.config.get("stations", None) is not None:
                self.stations = sorted(self.config["stations"])

            self.min = self.config.get("min", -np.Inf)
            self.max = self.config.get("max", np.Inf)

        self._logger = logging.getLogger(self.__class__.__module__)
        self._logger.debug(
            "Created distribute.%s", self.__class__.__module__.replace("smrf.", "")
        )

    def __str__(self) -> str:
        """
        Name of the file this class is defined in.
        This used when writing output to NetCDF files.

        :return: str - Name of this file
        """
        return self.__module__.split(".")[-1]

    # Add some accessor methods to topo information
    @property
    def dem(self):
        return self.topo.dem

    @property
    def sky_view_factor(self):
        return self.topo.sky_view_factor

    @property
    def veg_type(self):
        return self.topo.veg_type

    @property
    def veg_height(self):
        return self.topo.veg_height

    @property
    def veg_tau(self):
        return self.topo.veg_tau

    @property
    def veg_k(self):
        return self.topo.veg_k

    @property
    def veg_type(self):
        return self.topo.veg_type

    # END - Topo accessor methods

    # START - config accessors
    @property
    def distribution_method(self):
        return self.config["distribution"]

    # END - config accessors

    # START - Class methods

    # This sets constants for:
    # * OUTPUT_OPTIONS dictionary defined in the child class
    # * List of LOADED_DATA from the input when not already defined on the child class
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        setattr(cls, "OUTPUT_OPTIONS", set(cls.OUTPUT_VARIABLES.keys()))
        if not hasattr(cls, "LOADED_DATA"):
            setattr(cls, "LOADED_DATA", [cls.DISTRIBUTION_KEY])

    @classmethod
    def is_requested(cls, config_variables: set) -> bool:
        """
        Test if any of the available output variables are in the requested via the
        [output] section `variables`

        :param config_variables: Set - List of requested variables

        :return: Boolean - True if one of the output variables is in the requested
        """
        return len(cls.OUTPUT_OPTIONS & config_variables) > 0

    # END - Class methods

    def initialize(self, metadata: pd.DataFrame):
        """
        Second initialzie step to load stations (if configured) and open the configured
        `source_files` from the `start_date` in the `.ini` file.

        Args:
            metadata: metadata Pandas dataframe containing the station metadata
                      from :mod:`smrf.data.loadData` or :mod:`smrf.data.loadGrid

        Attributes set:
            * :py:attr:`date_time`
            * :py:attr:`metadata`
            * :py:attr:`topo`
        """
        self._logger.debug("Initializing")

        # pull out the metadata subset
        if self.stations is not None:
            self.metadata = metadata.loc[self.stations]
        else:
            self.metadata = metadata

        if self.config.get(self.SOURCE_FILES, None) is not None:
            self._logger.info(
                f"Using {self.DISTRIBUTION_KEY} files from configured path"
            )
            self._open_source_files(self.config[self.SOURCE_FILES])

        self._initialize()

    def _initialize(self):
        """
        Initialize the distribution based on the parameters in :py:attr:`config`.

        Raises:
            Exception: If the distribution method could not be determined
        """

        if "distribution" in self.config.keys():
            if self.distribution_method == InverseDistanceWeighted.CONFIG_KEY:
                self.idw = InverseDistanceWeighted(
                    self.metadata.utm_x.values,
                    self.metadata.utm_y.values,
                    self.topo.X,
                    self.topo.Y,
                    mz=self.metadata.elevation.values,
                    GridZ=self.topo.dem,
                    power=self.config["idw_power"],
                )

            elif self.distribution_method == DetrendedKriging.CONFIG_KEY:
                self.dk = DetrendedKriging(
                    self.metadata.utm_x.values,
                    self.metadata.utm_y.values,
                    self.metadata.elevation.values,
                    self.topo.X,
                    self.topo.Y,
                    self.topo.dem,
                    self.config,
                    self.threads,
                )

            elif self.distribution_method == Grid.CONFIG_KEY:
                # linear interpolation between points
                self.grid = Grid(
                    self.config,
                    self.metadata.utm_x.values,
                    self.metadata.utm_y.values,
                    self.topo.X,
                    self.topo.Y,
                    mz=self.metadata.elevation.values,
                    grid_z=self.topo.dem,
                    mask=self.topo.mask,
                    metadata=self.metadata,
                )

            elif self.distribution_method == Kriging.CONFIG_KEY:
                self.kriging = Kriging(
                    self.metadata.utm_x.values,
                    self.metadata.utm_y.values,
                    self.metadata.elevation.values,
                    self.topo.X,
                    self.topo.Y,
                    self.topo.dem,
                    self.config,
                )

            else:
                raise Exception(
                    "Could not determine the distribution method for {}".format(
                        self.DISTRIBUTION_KEY
                    )
                )

    def _distribute(self, data, other_attribute=None, zeros=None):
        """
        Distribute the data using the defined distribution method in
        :py:attr:`config`

        Args:
            data: Pandas dataframe for a single time step
            other_attribute (str): By default, the distributed data matrix goes
                into self.variable but this specifies another attribute in self
            zeros: data values that should be treated as zeros (not used)

        Raises:
            Exception: If all input data is NaN
        """
        # Subset if necessary
        if self.stations is not None:
            data = data[self.stations]

        if np.sum(data.isnull()) == data.shape[0]:
            raise Exception("{}: All data values are NaN".format(self.DISTRIBUTION_KEY))

        if self.distribution_method == InverseDistanceWeighted.CONFIG_KEY:
            if self.config["detrend"]:
                v = self.idw.detrendedIDW(
                    data.values, self.config["detrend_slope"], zeros=zeros
                )
            else:
                v = self.idw.calculateIDW(data.values)

        elif self.distribution_method == DetrendedKriging.CONFIG_KEY:
            v = self.dk.calculate(data.values)

        elif self.distribution_method == Grid.CONFIG_KEY:
            if self.config["detrend"]:
                v = self.grid.detrended_interpolation(
                    data, self.config["detrend_slope"], self.config["grid_method"]
                )
            else:
                v = self.grid.calculate_interpolation(
                    data.values, self.config["grid_method"]
                )

        elif self.distribution_method == Kriging.CONFIG_KEY:
            v, ss = self.kriging.calculate(data.values)
            setattr(self, "{}_variance".format(self.DISTRIBUTION_KEY), ss)

        if other_attribute is not None:
            setattr(self, other_attribute, v)
        else:
            setattr(self, self.DISTRIBUTION_KEY, v)

    def _open_source_files(self, base_path: str) -> None:
        """
        Construct a file path to read files configured via the `source_files' key in
        a section. The data is expected to be with the following structure:
        ```
        /path/to/forcing/
            YYYYMMDD/
                variable.nc
        ```
        Where the `variable` should match the [section] in the .ini file

        Example config:
        ```
        [my_variable]
        source_files = /path/to/forcing/
        ```

        The above example would attempt to load the following file on 2025-10-01:
        `/path/to/forcing/20251001/my_variable.nc`

        Sets:
        * :py:attr:`source_files`

        Args:
            base_path: Configured path from the variable section

        :return:
        """
        file = Path(base_path) / self.start_date / (self.DISTRIBUTION_KEY + ".nc")
        self.source_files = ReadNetCDF(file, self.time_zone)
