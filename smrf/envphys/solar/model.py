from smrf.envphys import sunang
from smrf.envphys.solar.irradiance import direct_solar_irradiance
from smrf.envphys.solar.twostream import twostream


def model_solar(dt, lat, lon, tau=0.2, tzone=0):
    """
    Model solar radiation at a point
    Combines sun angle, solar and two stream

    Args:
        dt - datetime object
        lat - latitude
        lon - longitude
        tau - optical depth
        tzone - time zone

    Returns:
        corrected solar radiation
    """

    # determine the sun angle
    cosz, az, rad_vec = sunang.sunang(dt, lat, lon)

    # calculate the solar irradiance
    sol = direct_solar_irradiance(dt, [0.28, 2.8])

    # calculate the two stream model value
    R = twostream(cosz, sol, tau=tau)

    return R['irradiance_at_bottom']
