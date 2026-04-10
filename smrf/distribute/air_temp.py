import numpy as np
import numexpr as ne

from numpy.polynomial import Polynomial
from io import StringIO

from smrf.utils import utils

from .variable_base import VariableBase

LOOKUP_TABLE_DATA = """
Month,lapse_rate_Obs,lapse_rate_Mod,Multiplier
1,-0.006201984627358696,-0.006111007494547572,1.0148874196099902
2,-0.00563857020046628,-0.006145329338965829,0.9175375133621676
3,-0.005871829845905024,-0.006940257032262386,0.8460536574667644
4,-0.005933293059229652,-0.007610275268632311,0.7796423716347333
5,-0.004950479875245373,-0.007080299713120592,0.6991907229677831
6,-0.004702532777890484,-0.007019169190013406,0.669955752680967
7,,,
8,,,
9,,,
10,-0.006053721545391756,-0.005568149203234655,1.0872053395902217
11,-0.005509139112981821,-0.006197981657095438,0.8888601834881148
12,-0.0048201525905793245,-0.005618270835268653,0.8579423690863839
"""

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
    LOOKUP_TABLE = data = np.genfromtxt(StringIO(LOOKUP_TABLE_DATA), delimiter=",", names=True, filling_values=np.nan)

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
        self.air_temp = utils.set_min_max(self.air_temp, self.min, self.max)
        self.adjust_lapse_rate(timestep)

    def adjust_lapse_rate(self, timestep):
        """
        Adjust the modeled lapse rate from the incoming data with a multiplier from
        observed data. This removes the modeled rate first and re-applies a new lapse
        rate.

        Args:
            timestep: Current processed datetime

        Changes :py:attr:`air_temp`
        """
        poly_temp = Polynomial.fit(self.dem.ravel(), self.air_temp.ravel(), 1)
        # Remove input data trend
        self.air_temp -= poly_temp(self.dem.ravel()).reshape(self.dem.shape)
        intercept, slope = poly_temp.convert().coef

        # Adjust to monthly observed lapse rate
        slope *= self.LOOKUP_TABLE["Multiplier"][timestep.month - 1]
        ne.evaluate(
            "air_temp + slope * dem + intercept",
            local_dict={
                "air_temp": self.air_temp,
                "slope": slope,
                "intercept": intercept,
                "dem": self.dem,
            },
            out=self.air_temp,
        )
