import numpy as np

from .image_data import ImageData


class SoilTemperature(ImageData):
    """
    Soil temperature is simply set to a constant value during initialization.
    If soil temperature measurements are available, the values can be
    distributed using the distribution methods.
    """

    VARIABLE = "soil_temp"

    # these are variables that can be output
    OUTPUT_VARIABLES = {
        VARIABLE: {
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
