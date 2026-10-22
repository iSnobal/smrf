import numpy as np
import pandas as pd

from smrf.envphys.solar.model import model_solar

MIN_SOLAR = 50

def cf_cloud(beam, diffuse, cf):
    """
    Correct beam and diffuse irradiance for cloud attenuation at a single
    time, using input clear-sky global and diffuse radiation calculations
    supplied by locally modified toporad or locally modified stoporad

    Args:
        beam: global irradiance
        diffuse: diffuse irradiance
        cf: cloud attenuation factor - actual irradiance / clear-sky irradiance

    Returns:
        c_grad: cloud corrected gobal irradiance
        c_drad: cloud corrected diffuse irradiance

    20150610 Scott Havens - adapted from cloudcalc.c
    """

    # define some constants
    CRAT1 = 0.15
    CRAT2 = 0.99
    CCOEF = 1.38

    # cloud attenuation, beam ratio is reduced
    bf_c = CCOEF * (cf - CRAT1) ** 2
    c_grad = beam * cf
    c_brad = c_grad * bf_c
    c_drad = c_grad - c_brad

    # extensive cloud attenuation, no beam
    ind = cf <= CRAT1
    c_brad[ind] = 0
    c_drad[ind] = c_grad[ind]

    # minimal cloud attenution, no beam ratio reduction
    ind = cf > CRAT2
    c_drad[ind] = diffuse[ind] * cf[ind]
    c_brad[ind] = c_grad[ind] - c_drad[ind]

    return c_grad, c_drad


def get_hrrr_cloud(
    df_solar: pd.DataFrame,
    lat: float,
    lon: float,
) -> pd.DataFrame:
    """
    Take the combined solar from HRRR and use the two stream calculation
    at the specific HRRR pixels to find the cloud_factor.

    Args:
        df_solar - solar dataframe from hrrr
        lat - basin lat
        lon - basin lon

    Returns:
        df_cf - cloud factor dataframe in same format as df_solar input
    """

    # Clear-sky solar at the basin lat/lon for each time. The index is converted
    # to UTC without a timezone because ephemeris reads the hour, minute, and
    # second fields directly
    dates = df_solar.index.tz_convert(None).to_numpy()
    clear_sky_solar = np.float32(
        [model_solar(pd.Timestamp(dt), lat, lon) for dt in dates]
    )
    basin_solar = pd.DataFrame(
        np.broadcast_to(clear_sky_solar[:, None], df_solar.shape),
        index=df_solar.index,
        columns=df_solar.columns,
    )

    # Zero out low solar (night and near sunrise/sunset), where the cloud
    # factor is difficult to calculate
    low_sun = basin_solar < MIN_SOLAR
    basin_solar = basin_solar.mask(low_sun, 0)
    df_solar = df_solar.mask(low_sun, 0)

    # Masked values are 0 / 0 at night, which gives NaN
    df_cf = df_solar / basin_solar

    # Fill night gaps by linear interpolation, then the nearest value at the ends
    df_cf = df_cf.interpolate(method="linear").ffill().bfill()

    return df_cf.clip(0.0, 1.0)
