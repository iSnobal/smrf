from dataclasses import dataclass


@dataclass
class HrrrVariable:
    """
    Class holding information to load HRRR variables from
    GRIB files and maps to the corresponding SMRF variable.
    """
    level: str
    grib_identifier: str
    grib_keys: dict
    smrf_map: dict


# HRRR variables
FIRST_HOUR = {'stepRange': ['1', '0-1']}
SIXTH_HOUR = {'stepRange': ['6', '5-6']}
SURFACE = 'surface'
HAG = 'heightAboveGround'

HRRR_SURFACE = HrrrVariable(
    level=SURFACE,
    grib_identifier='shortName',
    grib_keys={
        'level': 0,
        'typeOfLevel': SURFACE,
    },
    smrf_map={
        'orog': 'elevation',
        'sdswrf': 'short_wave',
        'tp': 'precip_int',
    }
)
HRRR_HAG_2 = HrrrVariable(
    level=HAG,
    grib_identifier='cfVarName',
    grib_keys={
        'level': 2,
        'typeOfLevel': HAG,
    },
    smrf_map={
        'r2': 'relative_humidity',
        't2m': 'air_temp',
    },
)
HRRR_HAG_10 = HrrrVariable(
    level=HAG,
    grib_identifier='cfVarName',
    grib_keys={
        'level': 10,
        'typeOfLevel': HAG,
    },
    smrf_map={
        'u10': 'wind_u',
        'v10': 'wind_v',

    },
)
