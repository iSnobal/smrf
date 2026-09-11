import numpy as np
import numpy.typing as npt

from smrf.envphys.core import envphys_c


def saturation_vapor_pressure(air_temperature: npt.NDArray) -> npt.NDArray:
    """
    Saturation vapor pressure using Goff-Gratch formulations for given temperatures.

    See the C functions `sati` and `satw` in topotherm.c for more references.

    Args:
        air_temperature: temperature in Kelvin

    Returns:
        Saturated vapor pressure
    """
    temperature_k = np.ascontiguousarray(air_temperature, dtype=np.float64)
    output = np.zeros_like(temperature_k)

    envphys_c.svp_for_temperatures(temperature_k, output)

    return output


def idewpt(vp):
    """
    Calculate the dew point given the vapor pressure

    Args:
        vp - array of vapor pressure values in [Pa]

    Returns:
        dewpt - array same size as vp of the calculated
            dew point temperature [C] (see Dingman 2002).

    """

    # ensure that vp is a numpy array
    vp = np.array(vp)

    # take the log and convert to kPa
    vp = np.log(vp / float(1000))

    # calculate the vapor pressure
    Td = (vp + 0.4926) / (0.0708 - 0.00421 * vp)

    return Td


def rh2vp(ta: npt.NDArray, rh: npt.NDArray) -> npt.NDArray:
    """
    Calculate the vapor pressure given the air temperature and relative humidity

    Args:
        ta: array of air temperature in [C]
        rh: array of relative humidity from 0-100 [%]

    Returns:
        vapor pressure
    """

    if rh.flat[0] >= 1.0:
        rh = rh / 100.0

    satvp = saturation_vapor_pressure(ta + 273.15)

    return satvp * rh


def svp_for_celsius(t_in_c: npt.NDArray) -> npt.NDArray:
    """
    Calculate the saturation vapor pressure for temperatures in Celsius

    Args:
        t_in_c: array of temperatures in [C]

    Returns:
        vapor_pressure
    """

    return saturation_vapor_pressure(t_in_c + 273.15)
