import numpy as np

from .variable_base import VariableBase


class SoilTemperature(VariableBase):
    """
    Soil temperature is simply set to a constant value during initialization.
    If soil temperature measurements are available, the values can be
    distributed using the distribution methods.
    """

    DISTRIBUTION_KEY = "soil_temp"

    # these are variables that can be output
    OUTPUT_VARIABLES = {
        DISTRIBUTION_KEY: {
            "units": "degree_Celsius",
            "standard_name": "soil_temperature",
            "long_name": "Soil temperature",
        }
    }

    def distribute(self):
        """
        No distribution is performed on soil temperature at the moment, method
        simply passes.

        Args:
            None
        """
        self.soil_temp = float(self.config['temp']) * np.ones(self.topo.dem.shape)
