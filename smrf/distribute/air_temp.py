from .variable_base import VariableBase
from smrf.utils import utils


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

    def distribute(self, data):
        """
        Distribute air temperature given a Panda's dataframe for a single time
        step. Calls :mod:`smrf.distribute.image_data.image_data._distribute`.

        Args:
            data: Pandas dataframe for a single time step from air_temp

        """
        self._logger.debug("{} Distributing".format(data.name))

        self._distribute(data)
        self.air_temp = utils.set_min_max(self.air_temp, self.min, self.max)
