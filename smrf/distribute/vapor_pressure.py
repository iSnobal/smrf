
import numpy as np

from .variable_base import VariableBase
from smrf.envphys.core import envphys_c
from smrf.utils import utils


class VaporPressure(VariableBase):
    """
    Vapor pressure is provided as an argument and is calculated from coincident
    air temperature and relative humidity measurements using utilities such as
    :mod:`smrf.envphys.vapor_pressure.rh2vp`. The vapor pressure is distributed
    instead of the relative humidity as it is an absolute measurement of the
    vapor within the atmosphere and will follow elevational trends (typically
    negative).  Were as relative humidity is a relative measurement which
    varies in complex ways over the topography.  From the distributed vapor
    pressure, the dew point is calculated for use by other distribution
    methods. The dew point temperature is further corrected to ensure that it
    does not exceed the distributed air temperature.
    """

    DISTRIBUTION_KEY = "vapor_pressure"

    # these are variables that can be output
    OUTPUT_VARIABLES = {
        "vapor_pressure": {
            "units": "pascal",
            "standard_name": "vapor_pressure",
            "long_name": "Vapor pressure",
        },
        "dew_point": {
            "units": "degree_Celsius",
            "standard_name": "dew_point_temperature",
            "long_name": "Dew point temperature",
        },
        "precip_temp": {
            "units": "degree_Celsius",
            "standard_name": "precip_temperature",
            "long_name": "Precip temperature",
        },
    }

    def __init__(self, config):
        super().__init__(config)

        self.precip_temp_method = config["precip"]["precip_temp_method"]

    def distribute(self, data, ta):
        """
        Distribute air temperature given a Panda's dataframe for a single time
        step. Calls :mod:`smrf.distribute.ImageData._distribute`.

        The following steps are performed when distributing vapor pressure:

        1. Distribute the point vapor pressure measurements
        2. Calculate dew point temperature using
            :mod:`smrf.envphys.core.envphys_c.cdewpt`
        3. Adjust dew point values to not exceed the air temperature

        Args:
            data: Pandas dataframe for a single time step from precip
            ta: air temperature numpy array that will be used for calculating
                dew point temperature

        """

        self._logger.debug('%s -- Distributing vapor_pressure' % data.name)

        # calculate the vapor pressure
        self._distribute(data)

        # set the limits
        self.vapor_pressure = utils.set_min_max(self.vapor_pressure,
                                                self.min,
                                                self.max)

        # calculate the dew point
        self._logger.debug('%s -- Calculating dew point' % data.name)

        # use the core_c to calculate the dew point
        dpt = np.zeros_like(self.vapor_pressure, dtype=np.float64)
        envphys_c.cdewpt(
            self.vapor_pressure,
            dpt,
            self.config["dew_point_tolerance"],
            self.config["threads"],
        )

        # find where dpt > ta
        ind = dpt >= ta

        if (np.sum(ind) > 0):  # or np.sum(indm) > 0):
            dpt[ind] = ta[ind] - 0.2

        self.dew_point = dpt

        # calculate wet bulb temperature
        if self.precip_temp_method == 'wet_bulb':
            # initialize timestep wet_bulb
            wet_bulb = np.zeros_like(self.vapor_pressure, dtype=np.float64)
            # calculate wet_bulb
            envphys_c.cwbt(
                ta,
                dpt,
                self.dem,
                wet_bulb,
                self.config["dew_point_tolerance"],
                self.config["threads"],
            )
            # # store last time step of wet_bulb
            # self.wet_bulb_old = wet_bulb.copy()
            # store in precip temp for use in precip
            self.precip_temp = wet_bulb
        else:
            self.precip_temp = dpt
