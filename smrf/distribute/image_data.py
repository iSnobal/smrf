import logging

import numpy as np
import pandas as pd

from smrf.data.load_topo import Topo
from smrf.spatial import grid, idw, kriging
from smrf.spatial.dk import dk


class ImageData:
    """
    Base class that will ensure all variables are distributed in the same manner.

    Attributes:
        VARIABLE: The name of the variable that a class will distribute
        config:   Parsed dictionary from the configuration file for the variable
        stations: The stations to be used for the variable, if set, in
                  alphabetical order
        metadata: The metadata Pandas dataframe containing the station
                  information from :mod:`smrf.data.loadData` or :mod:`smrf.data.loadGrid`
        topo:     Access to the variables stored in the Topo
        gridded:  Boolean indicator of whether the variable is gridded or not
        max:      Maximum allowed value for the variable
        min:      Minimum allowed value for the variable
    """
    VARIABLE = ''
    OUTPUT_VARIABLES = {}

    def __init__(self, config):
        if self.VARIABLE != '':
            setattr(self, self.VARIABLE, None)

        self.topo = None
        self.metadata = None

        self.config = config[self.VARIABLE]

        # check of gridded interpolation
        self.gridded = config.get('distribution', None) == 'grid'

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

    @property
    def output_variables(self):
        ov = {}
        for key, value in self.OUTPUT_VARIABLES.items():
            # Add the variable module name to the dict to identify outputs in NetCDF
            value['module'] = self.__class__.__module__.split('.')[-1]
            ov[key] = value
        return ov

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

    def initialize(self, topo: Topo, metadata: pd.DataFrame):
        """
        Args:
            topo: Topo class instance
            metadata: metadata Pandas dataframe containing the station metadata
                  from :mod:`smrf.data.loadData` or :mod:`smrf.data.loadGrid

        Attributes set:
            * :py:attr:`date_time`
            * :py:attr:`metadata`
            * :py:attr:`topo`
        """
        self._logger.debug("Initializing")
        self.metadata = metadata
        self.topo = topo

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
                        self.VARIABLE
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
            raise Exception("{}: All data values are NaN".format(self.VARIABLE))

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
            setattr(self, '{}_variance'.format(self.VARIABLE), ss)

        if other_attribute is not None:
            setattr(self, other_attribute, v)
        else:
            setattr(self, self.VARIABLE, v)
