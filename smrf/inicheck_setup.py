import os

from .version import __version__ as SMRF_VERSION

# Names that configure inicheck, re-exported by smrf/__init__.py
__core_config__ = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "framework", "CoreConfig.ini")
)
__recipes__ = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "framework", "recipes.ini")
)
__config_changelog__ = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "framework", "changelog.ini")
)

__config_titles__ = {
    "topo": "Files for DEM and vegetation",
    "time": "Dates to run model",
    "stations": "Stations to use",
    "csv": "CSV section configurations",
    "gridded": "Gridded datasets configurations",
    "air_temp": "Air temperature distribution",
    "vapor_pressure": "Vapor pressure distribution",
    "wind": "Wind speed and wind direction distribution",
    "precip": "Precipitation distribution",
    "albedo": "Albedo distribution",
    "cloud_factor": "Cloud Factor - Fraction used to limit solar radiation Cloudy (0) - Sunny (1)",
    "solar": "Solar radiation",
    "thermal": "Incoming thermal radiation",
    "soil_temp": "Soil temperature",
    "output": "Output variables",
    "system": "System variables and Logging",
}

__config_header__ = f"Config File for SMRF {SMRF_VERSION}\n"
__config_checkers__ = "utils.checkers"
