import pandas as pd

from smrf.data.load_topo import Topo
from smrf.envphys.constants import EMISS_TERRAIN, STEF_BOLTZ, FREEZE

from .variable_base import VariableBase


class ThermalHRRR(VariableBase):
    """
    Calculate thermal radiation based of the HRRR
    Downwelling Longwave Radiation Flux (DLWRF) and corrected by the sky view factor.

    .. math::
        LW_{in} = V_f \\times DLWRF + (1 - V_f) \\times \\epsilon \\sigma T_a^4

        \\sigma = Stefan Boltzmann constant
        \\epsilon = Emissivity of the terrain
        \\T_a = Air temperature in Kelvin
    """

    VARIABLE = "thermal"
    INI_VARIABLE = "hrrr_thermal"
    GRIB_NAME = "DLWRF"

    OUTPUT_VARIABLES = {
        VARIABLE: {
            "units": "watt/m2",
            "standard_name": "thermal_radiation",
            "long_name": "Thermal (longwave) radiation",
        },
    }

    def initialize(self, topo: Topo, metadata: pd.DataFrame) -> None:
        """
        Trimmed down version of the base class as there is no need to initialize
        the interpolation method.
        """
        self._logger.debug("Initializing")
        self.metadata = metadata
        self.topo = topo

    def distribute(self, date_time, forcing_data, air_temp):
        self._logger.debug('%s Distributing HRRR thermal' % date_time)
        self.thermal = (self.sky_view_factor * forcing_data) + (
            1 - self.sky_view_factor
        ) * EMISS_TERRAIN * STEF_BOLTZ * (air_temp + FREEZE) ** 4
