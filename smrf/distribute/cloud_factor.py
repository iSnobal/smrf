from .variable_base import VariableBase
from smrf.utils import utils


class CloudFactor(VariableBase):
    """
    Cloud factor is a relatively simple variable to distribute as it does not rely on any
    other variables.

    Cloud factor is calculated as the ratio between measured incoming
    solar radiation and modeled clear sky radiation. A value of 0 means
    no incoming solar radiation (or very cloudy) and a value of 1 means
    sunny.
    """

    VARIABLE = "cloud_factor"

    # these are variables that can be output
    OUTPUT_VARIABLES = {
        "cloud_factor": {
            "units": "None",
            "standard_name": "cloud_factor",
            "long_name": "cloud factor",
        }
    }

    def distribute(self, data):
        """
        Distribute cloud factor given a Panda's dataframe for a single time
        step. Calls :mod:`smrf.distribute.ImageData._distribute`.

        Args:
            data: Pandas dataframe for a single time step from cloud_factor
        """

        self._logger.debug('{} Distributing cloud_factor'.format(data.name))

        self._distribute(data)
        self.cloud_factor = utils.set_min_max(
            self.cloud_factor, self.min, self.max
        )
