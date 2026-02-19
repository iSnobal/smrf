import numexpr as ne

from smrf.envphys.constants import EMISS_VEG, FREEZE, STEF_BOLTZ  # noqa


def thermal_correct_canopy(th, ta, tau, veg_height, height_thresh=2):
    """
    Correct thermal radiation for vegetation for pixels where the height
    is above a threshold. This ensures that the open areas don't get this applied.
    Vegetation temp is assumed to be at air temperature.
    Equations from Link and Marks 1999 :cite:`Link&Marks:1999`

    Args:
        th: thermal radiation
        ta: air temperature [C]
        tau: transmissivity of the canopy
        veg_height: vegetation height for each pixel
        height_thresh: threshold hold for height to say that there is veg in
            the pixel

    Returns:
        Vegetation corrected thermal radiation
    """

    return ne.evaluate(
        "where(veg_height > height_thresh, "
        "tau * th + (1 - tau) * (STEF_BOLTZ * EMISS_VEG * (ta + FREEZE)**4), "
        "th)"
    )
