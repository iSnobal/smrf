import numexpr as ne
import numpy as np


def solar_veg_beam(
    direct_radiation: np.ndarray,
    vegetation_height: np.ndarray,
    illumination_angles: np.ndarray,
    k: np.ndarray,
) -> np.ndarray:
    """
    Apply the vegetation correction to the beam irradiance (direct radiation)
    using equation (2) from Link and Marks (1999) for all cells where the sun is visible.

    .. math::
        S_b = S_{b,o} * exp[ -k * h / cos_z ]

        S_b: Direct radiation corrected for vegetation
        S_{b,o}: Incoming direct radiation (unobstructed)
        k: Vegetation attenuation coefficient
        h: Vegetation canopy height
        cos_z: Illumination angles

    Args:
        direct_radiation: Incoming direct radiation
        vegetation_height: Height of the vegetation canopy
        illumination_angles: Illumination angles masked for shade
        k: Vegetation attenuation coefficient

    Returns:
        Direct radiation corrected for vegetation.
    """
    return ne.evaluate(
        "where("
        " illumination_angles > 0,"
        " direct_radiation * exp(-k * vegetation_height / illumination_angles), "
        " direct_radiation"
        ")"
    )


def solar_veg_diffuse(diffuse_radiation: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """
    Apply the vegetation correction to the diffuse irradiance using equation (1)
    from Link and Marks (1999)

    .. math::
        S_{d} = tau * S_{d,o}

        S_d: Diffuse radiation corrected for vegetation
        tau: Optical transmissivity of the canopy
        S_{d,o}: Incoming diffuse radiation (unobstructed)

    Args:
        diffuse_radiation: Incoming diffuse radiation
        tau: Optical transmissivity of the canopy

    Returns:
        Diffuse radiation corrected for vegetation.
    """
    return ne.evaluate("tau * diffuse_radiation")
