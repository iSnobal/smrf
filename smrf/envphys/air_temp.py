import numexpr as ne
import numpy as np
import numpy.typing as npt
from numpy.polynomial import Polynomial

# Constants for logistic curve
ELEVATION_Z0 = 400
CURVE_K = 0.01


def adjust_lapse_rate(
    air_temp: npt.NDArray, dem: npt.NDArray, month_multiplier: float
) -> npt.NDArray:
    """
    Adjust the modeled lapse rate from the incoming data with a multiplier from
    observed data. This removes the modeled rate first and re-applies a new lapse
    rate.

    :argument
        air_temp: Air temperature to correct
        dem: Elevation data corresponding to the air temperature
        month_multiplier: Monthly lapse rate adjustment

    :returns
        Numpy array with adjusted air temperature
    """
    poly_temp = Polynomial.fit(dem.ravel(), air_temp.ravel(), 1)
    # Remove input data trend
    air_temp -= poly_temp(dem.ravel()).reshape(dem.shape)
    intercept, slope = poly_temp.convert().coef

    # Adjust to monthly observed lapse rate
    slope *= month_multiplier
    ne.evaluate(
        "air_temp + slope * dem + intercept",
        local_dict={
            "air_temp": air_temp,
            "slope": slope,
            "intercept": intercept,
            "dem": dem,
        },
        out=air_temp,
    )

    return air_temp


def adjust_by_elevation(
    air_temp: npt.NDArray, dem: npt.NDArray, month_multiplier: float
) -> npt.NDArray:
    """
    Apply a variable correction based on reference and DEM elevation and using a
    logistic curve.

    .. math::
        b = a / (1 + exp{-k * (z - z0)})

    :argument
        air_temp: Air temperature to correct
        dem: Elevation data corresponding to the air temperature
        month_multiplier: Monthly lapse rate adjustment

    :returns
        Numpy array with adjusted air temperature
    """
    lapse_rate_adjusted = adjust_lapse_rate(air_temp.copy(), dem, month_multiplier)
    # Maximum difference to the original, this changes every day based on calculated lapse rate
    # This is `a` in the equation
    max_difference = np.max(lapse_rate_adjusted - air_temp)

    # Adjust the originally cubic interpolated data
    # by the logistic curve lapse rate (a function of reference z and elevation)
    ne.evaluate(
        "air_temp + (max_difference / (1 + exp(-K * (dem - ELEVATION_Z0))))",
        local_dict={
            "air_temp": air_temp,
            "max_difference": max_difference,
            "K": CURVE_K,
            "dem": dem,
            "ELEVATION_Z0": ELEVATION_Z0,
        },
        out=air_temp,
    )

    return air_temp
