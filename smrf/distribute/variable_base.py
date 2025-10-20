import logging

import numpy as np
import pandas as pd

from smrf.data.load_topo import Topo
from smrf.spatial import grid, idw, kriging
from smrf.spatial.dk import dk


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
    DISTRIBUTION_KEY = ''
    OUTPUT_VARIABLES = {}

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
        if self.DISTRIBUTION_KEY != '':
            setattr(self, self.DISTRIBUTION_KEY, None)

        self.config = None
        self.topo = topo
        self.metadata = None

        if config and config.get(self.DISTRIBUTION_KEY, False):
            self.config = config[self.DISTRIBUTION_KEY]

            # Check of gridded interpolation
            self.gridded = self.config.get("distribution", None) == "grid"

            self.stations = None
            if self.config.get("stations", None) is not None:
                stations = self.config['stations']
                stations.sort()
                self.stations = stations

            self.min = self.config.get('min', -np.Inf)
            self.max = self.config.get('max', np.Inf)

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
    def veg_height(self):
        return self.topo.veg_height

    @property
    def veg_tau(self):
        return self.topo.veg_tau

    @property
    def veg_k(self):
        return self.topo.veg_k

    # END - Topo accessor methods

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
        Args:
            metadata: metadata Pandas dataframe containing the station metadata
                  from :mod:`smrf.data.loadData` or :mod:`smrf.data.loadGrid

        Attributes set:
            * :py:attr:`date_time`
            * :py:attr:`metadata`
            * :py:attr:`topo`
        """
        self._logger.debug("Initializing")
        self.metadata = metadata

        # pull out the metadata subset
        if self.stations is not None:
            self.metadata = metadata.loc[self.stations]
        else:
            self.stations = metadata.index.values

        self._initialize()

    def _initialize(self):
        """
        Initialize the distribution based on the parameters in :py:attr:`config`.

        Raises:
            Exception: If the distribution method could not be determined
        """

        if "distribution" in self.config.keys():
            if self.config["distribution"] == "idw":
                # inverse distance weighting
                self.idw = idw.IDW(
                    self.metadata.utm_x.values,
                    self.metadata.utm_y.values,
                    self.topo.X,
                    self.topo.Y,
                    mz=self.metadata.elevation.values,
                    GridZ=self.topo.dem,
                    power=self.config["idw_power"],
                )

            elif self.config["distribution"] == "dk":
                # detrended kriging
                self.dk = dk.DK(
                    self.metadata.utm_x.values,
                    self.metadata.utm_y.values,
                    self.metadata.elevation.values,
                    self.topo.X,
                    self.topo.Y,
                    self.topo.dem,
                    self.config,
                )

            elif self.config["distribution"] == "grid":
                # linear interpolation between points
                self.grid = grid.GRID(
                    self.config,
                    self.metadata.utm_x.values,
                    self.metadata.utm_y.values,
                    self.topo.X,
                    self.topo.Y,
                    mz=self.metadata.elevation.values,
                    GridZ=self.topo.dem,
                    mask=self.topo.mask,
                    metadata=self.metadata,
                )

            elif self.config["distribution"] == "kriging":
                # generic kriging
                self.kriging = kriging.KRIGE(
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

        # get the data for the desired stations
        # this will also order it correctly how air_temp was initialized
        data = data[self.stations]

        if np.sum(data.isnull()) == data.shape[0]:
            raise Exception("{}: All data values are NaN".format(self.DISTRIBUTION_KEY))

        if self.config['distribution'] == 'idw':
            if self.config['detrend']:
                v = self.idw.detrendedIDW(
                    data.values,
                    self.config['detrend_slope'],
                    zeros=zeros
                )
            else:
                v = self.idw.calculateIDW(data.values)

        elif self.config['distribution'] == 'dk':
            v = self.dk.calculate(data.values)

        elif self.config['distribution'] == 'grid':
            if self.config['detrend']:
                v = self.grid.detrendedInterpolation(
                    data,
                    self.config['detrend_slope'],
                    self.config['grid_method']
                )
            else:
                v = self.grid.calculateInterpolation(
                    data.values,
                    self.config['grid_method']
                )

        elif self.config['distribution'] == 'kriging':
            v, ss = self.kriging.calculate(data.values)
            setattr(self, '{}_variance'.format(self.DISTRIBUTION_KEY), ss)

        if other_attribute is not None:
            setattr(self, other_attribute, v)
        else:
            setattr(self, self.DISTRIBUTION_KEY, v)
