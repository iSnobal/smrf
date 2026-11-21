import numpy as np
import pandas as pd

from smrf.envphys.solar.model import model_solar

# Low-sun cutoff (W/m^2) on clear-sky shortwave; no cloud factor is computed below it.
LOW_SOLAR_THRESHOLD = 50


def cf_cloud(
    beam: np.ndarray, diffuse: np.ndarray, cf: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Correct beam and diffuse irradiance for cloud attenuation at a single
    time, using input clear-sky global and diffuse radiation calculations
    supplied by locally modified toporad or locally modified stoporad

    Args:
        beam: global irradiance
        diffuse: diffuse irradiance
        cf: cloud attenuation factor - actual irradiance / clear-sky irradiance

    Returns:
        c_grad: cloud corrected global irradiance
        c_drad: cloud corrected diffuse irradiance
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

    # heavy cloud attenuation, no beam
    ind = cf <= CRAT1
    c_brad[ind] = 0
    c_drad[ind] = c_grad[ind]

    # minimal cloud attenuation, no beam ratio reduction
    ind = cf > CRAT2
    c_drad[ind] = diffuse[ind] * cf[ind]
    c_brad[ind] = c_grad[ind] - c_drad[ind]

    return c_grad, c_drad


def cloud_factor(solar_ratio: pd.DataFrame, clear_sky: pd.DataFrame) -> pd.DataFrame:
    """
    Cloud factor from the ratio of incoming to clear-sky shortwave.

    Low-sun times are masked on clear-sky shortwave, interpolated in time, and
    clipped to [0, 1].

    Parameters
    ----------
    solar_ratio : pd.DataFrame
        Ratio of incoming to clear-sky shortwave.
    clear_sky : pd.DataFrame
        Clear-sky shortwave, same shape as solar_ratio.

    Returns
    -------
    pd.DataFrame
        Cloud factor, same shape as solar_ratio.
    """
    solar_ratio = solar_ratio.mask(clear_sky < LOW_SOLAR_THRESHOLD)

    # Fill gaps in time; edge gaps take the first or last valid value
    filled = solar_ratio.interpolate(method="linear").ffill().bfill()

    return filled.clip(0.0, 1.0)


def get_hrrr_cloud(
    incoming_solar: pd.DataFrame,
    lat: float,
    lon: float,
) -> pd.DataFrame:
    """
    Cloud factor (HRRR / clear-sky shortwave) at each grid cell.

    Clear-sky is computed once at the basin lat/lon, so it is uniform across
    the grid. Low-sun times are interpolated in time and clipped to [0, 1].
    The index must be timezone-aware.

    Args:
        incoming_solar: HRRR shortwave, indexed by timezone-aware datetimes.
        lat: basin latitude.
        lon: basin longitude.

    Returns:
        Cloud factor with the same index and columns as df_solar.
    """
    # Convert to naive UTC; ephemeris reads hour, minute, and second directly
    date_times = incoming_solar.index.tz_convert(None).to_numpy()

    # Clear-sky shortwave at the basin lat/lon, one value per timestep
    clear_sky_solar = np.float32(
        [model_solar(pd.Timestamp(dt), lat, lon) for dt in date_times]
    )

    # Same clear-sky value for every grid cell
    basin_solar = pd.DataFrame(
        np.broadcast_to(clear_sky_solar[:, None], incoming_solar.shape),
        index=incoming_solar.index,
        columns=incoming_solar.columns,
    )

    solar_ratio = incoming_solar / basin_solar
    return cloud_factor(solar_ratio, basin_solar)
