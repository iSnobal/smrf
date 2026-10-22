from datetime import datetime

from smrf.envphys import sunang
from smrf.envphys.solar.irradiance import direct_solar_irradiance
from smrf.envphys.solar.twostream import twostream


def model_solar(date_time: datetime, lat: float, lon: float, tau=0.2) -> float:
    """
    Model solar radiation at a point
    Combines sun angle, solar and two stream

    Args:
        date_time - datetime object
        lat - latitude
        lon - longitude
        tau - optical depth

    Returns:
        corrected solar radiation
    """

    # determine the sun angle
    cos_z, az, rad_vec = sunang.sunang(date_time, lat, lon)

    # calculate the solar irradiance
    solar_irradiance = direct_solar_irradiance(date_time, [0.28, 2.8])

    # calculate the two stream model value
    two_stream = twostream(cos_z, solar_irradiance, tau=tau)

    # Each key is an array with length 1
    return two_stream["irradiance_at_bottom"][0]
