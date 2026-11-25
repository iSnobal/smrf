from datetime import datetime

import numpy as np
import pandas as pd

from .thermal_hrrr import ThermalHRRR
from .variable_base import VariableBase


class ThermalWRF(ThermalHRRR):
    """
    Thermal radiation from WRF longwave on the topo grid.
    """

    def initialize(self, metadata: pd.DataFrame) -> None:
        """
        Set up the interpolation from the grid cells to the topo grid.
        """
        VariableBase.initialize(self, metadata)

    def distribute(
        self, date_time: datetime, forcing_data: pd.Series, air_temp: np.ndarray
    ) -> None:
        """
        Thermal radiation for a single time step.

        Parameters
        ----------
        date_time : datetime
            Time step being distributed
        forcing_data : pd.Series
            Longwave for each grid cell, shape (grid cells,)
        air_temp : ndarray
            Distributed air temperature in degrees C, shape (ny, nx)
        """
        self._distribute(forcing_data)
        super().distribute(date_time, self.thermal, air_temp)
