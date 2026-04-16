from io import StringIO

import numpy as np

from smrf.envphys.air_temp import adjust_by_elevation
from smrf.utils import utils

from .variable_base import VariableBase

LOOKUP_TABLE_DATA = """
Month,Multiplier
1,0.8856932782309219
2,0.8
3,0.8363605713104769
4,0.9559375367893626
5,0.9653522175168274
6,1
7,1
8,1
9,1
10,1
11,1
12,0.9082160449160459
"""

LOOKUP_TABLE = np.genfromtxt(
    StringIO(LOOKUP_TABLE_DATA), delimiter=",", names=True, filling_values=np.nan
)


class AirTemperature(VariableBase):
    """
    Air temperature is a relatively simple variable to distribute as it does
    not rely on any other variables, but has many variables that depend on it.

    Air temperature typically has a negative trend with elevation and performs
    best when detrended. However, even with a negative trend, it is possible to
    have instances where the trend does not apply, for example a temperature
    inversion or cold air pooling.  These types of conditions will have
    unintended consequences on variables that use the distributed air
    temperature.
    """

    DISTRIBUTION_KEY = "air_temp"

    # these are variables that can be output
    OUTPUT_VARIABLES = {
        DISTRIBUTION_KEY: {
            "units": "degree_Celsius",
            "standard_name": "air_temperature",
            "long_name": "Air temperature",
        }
    }

    def distribute(self, data, timestep):
        """
        Distribute air temperature given a Panda's dataframe for a single time
        step. Calls :mod:`smrf.distribute.image_data.image_data._distribute`.

        Args:
            data: Pandas dataframe for a single time step from air_temp
            timestep: Current processed datetime

        """
        self._logger.debug("{} Distributing".format(data.name))

        self._distribute(data)
        month_multiplier = LOOKUP_TABLE["Multiplier"][timestep.month - 1]
        self.air_temp = utils.set_min_max(self.air_temp, self.min, self.max)
        self.air_temp = adjust_by_elevation(self.air_temp, self.dem, month_multiplier)
