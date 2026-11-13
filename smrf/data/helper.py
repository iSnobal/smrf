import pandas as pd
import utm


def add_utm_columns(df: pd.DataFrame, force_zone_number: int) -> pd.DataFrame:
    """
    Add the UTM x and y columns to a dataframe from its latitude and longitude.

    Args:
        df: pandas dataframe with fields latitude and longitude
        force_zone_number: UTM zone number to project into

    Returns:
        df: the same dataframe with fields 'utm_x' and 'utm_y' added
    """
    p = utm.from_latlon(
        df["latitude"].values,
        df["longitude"].values,
        force_zone_number=force_zone_number,
    )
    df["utm_x"] = p[0]
    df["utm_y"] = p[1]
    return df
