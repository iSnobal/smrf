import numpy as np
from smrf.utils import utils

from smrf.distribute.variable_base import VariableBase
from .wind_ninja import WindNinjaModel
from .winstral import WinstralWindModel


class Wind(VariableBase):
    """
    Three distribution methods are available for the Wind class:

    1. Winstral and Marks 2002 method for maximum upwind slope (maxus)
    2. Import WindNinja simulations
    3. Standard interpolation
    """

    INTERP = "interp"
    VARIABLE = "wind"
    LOADED_DATA = ['wind_speed', 'wind_direction']

    # these are variables that can be output
    OUTPUT_VARIABLES = {
        "flatwind": {
            "units": "m/s",
            "standard_name": "flatwind_wind_speed",
            "long_name": "Simulated wind on a flat surface",
        },
        "wind_speed": {
            "units": "m/s",
            "standard_name": "wind_speed",
            "long_name": "Wind speed",
        },
        "wind_direction": {
            "units": "degrees",
            "standard_name": "wind_direction",
            "long_name": "Wind direction",
        },
    }

    def __init__(self, config):
        super().__init__(config)

        if self.model_type(self.INTERP):
            # Straight interpolation of the wind
            self.wind_model = self
            self.wind_model.flatwind = None
            self.wind_model.cellmaxus = None
            self.wind_model.dir_round_cell = None

        elif self.model_type(WindNinjaModel.MODEL_TYPE):
            self.wind_model = WindNinjaModel(self)

        elif self.model_type(WinstralWindModel.MODEL_TYPE):
            self.wind_model = WinstralWindModel(self)

    def model_type(self, wind_model: str) -> bool:
        """
        Check if given model is set on config

        Args:
            wind_model (str): name of the wind model to look up

        Returns:
            bool: True/False
        """
        return self.config.get('wind_model', None) == wind_model

    def initialize(self, topo, metadata):
        """
        See :mod:`smrf.distribute.ImageData.initialize` for documentation
        """
        super().initialize(topo, metadata)

        if not self.model_type(self.INTERP):
            self.wind_model.initialize()

    def distribute(self, data_speed, data_direction, t):
        """
        Distribute given a Panda's dataframe for a single time step. Calls
        :mod:`smrf.distribute.ImageData._distribute` for the `wind_model` chosen.

        Args:
            data_speed: Pandas dataframe for single time step from wind_speed
            data_direction: Pandas dataframe for single time step from
                wind_direction
            t: time stamp

        """

        self._logger.debug('{} Distributing wind_direction and wind_speed'
                           .format(data_speed.name))

        if self.model_type(self.INTERP):

            self._distribute(data_speed, other_attribute='wind_speed')

            # wind direction components at the station
            self.u_direction = np.sin(data_direction * np.pi/180)
            self.v_direction = np.cos(data_direction * np.pi/180)

            # distribute u_direction and v_direction
            self._distribute(self.u_direction,
                             other_attribute='u_direction_distributed')
            self._distribute(self.v_direction,
                             other_attribute='v_direction_distributed')

            # combine u and v to azimuth
            az = np.arctan2(self.u_direction_distributed,
                            self.v_direction_distributed)*180/np.pi
            az[az < 0] = az[az < 0] + 360
            self.wind_direction = az

        else:
            self.wind_model.distribute(data_speed, data_direction)

        for v in self.OUTPUT_OPTIONS:
            setattr(self, v, getattr(self.wind_model, v))

        # set min and max
        self.wind_speed = utils.set_min_max(self.wind_speed, self.min, self.max)
