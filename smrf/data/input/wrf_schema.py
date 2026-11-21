"""
Names and formats of the fields in WRF netcdf input files.
"""

from collections.abc import Iterable
from types import MappingProxyType

TIME_FORMAT = "%Y-%m-%d_%H:%M:%S"

# Input types: standard WRF output and the WUS-D3 hourly output
WRF = "wrf"
WUS_D3 = "wus_d3"

# Field key to WRF name, for each input type. Input types are checked in this
# order, and the first one whose WRF names are all in the file is used. WUS-D3
# comes first, so a file with both sets of fields is read as WUS-D3.
FIELDS = MappingProxyType(
    {
        WUS_D3: MappingProxyType(
            {
                "air_temp": "T2",
                "humidity": "Q2",
                "surface_pressure": "PSFC",
                "precip": "RAINNC",
                "u10": "U10",
                "v10": "V10",
                "shortwave": "SWDNB",
                "clear_shortwave": "SWDNBC",
                "longwave": "LWDNB",
            }
        ),
        WRF: MappingProxyType(
            {
                "air_temp": "T2",
                "dew_point": "DWPT",
                "u10": "UGRD",
                "v10": "VGRD",
                "cloud_factor": "CLDFRA",
                "precip": "RAINNC",
            }
        ),
    }
)

# Field keys with a vertical dimension, which are kept in full when read
LEVEL_FIELDS = frozenset({"cloud_factor"})

# Grid field names, which give the latitude, longitude and elevation of each cell
XLAT = "XLAT"
XLONG = "XLONG"
HGT = "HGT"

GRID_VARIABLES = (XLAT, XLONG, HGT)


def missing_fields(input_type: str, file_variables: set[str]) -> set[str]:
    """
    Get the WRF names an input type needs that are not in the file.

    Parameters
    ----------
    input_type : str
        Key of an input type in FIELDS.
    file_variables : set of str
        Names of the variables in the file.

    Returns
    -------
    set of str
        WRF names of the input type that are not in file_variables.
    """
    return set(FIELDS[input_type].values()) - file_variables


def detect_input_type(variables: Iterable[str]) -> str:
    """
    Determine the WRF input type from the variables in the file.

    WRF, WUS-D3, and other WRF-based data sets have different variable names.

    Parameters
    ----------
    variables : Iterable[str]
        Names of the variables in the file.

    Returns
    -------
    str
        Key of the input type in FIELDS with the fewest missing WRF names. Ties go
        to the first input type in FIELDS.

    Raises
    ------
    ValueError
        No input type in FIELDS has all of its WRF names in the file. The message
        names the closest input type and the names it is missing.
    """
    file_variables = set(variables)
    best_match = min(
        FIELDS,
        key=lambda input_type: len(missing_fields(input_type, file_variables)),
    )
    missing = sorted(missing_fields(best_match, file_variables))
    if missing:
        raise ValueError(
            f"File matches no WRF input type. Closest is {best_match}, missing "
            f"{', '.join(missing)}"
        )
    return best_match
