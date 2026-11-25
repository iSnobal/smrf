"""
Tolerances for comparing test results with gold reference values.

Gold and test outputs are both NetCDF files written by the output module,
so the values compared are already rounded when stored.

Each name says what the check guards and has platform logic if needed.
"""

from sys import platform

# Gold NetCDF comparisons
# GOLD_RTOL matches the significant digits set in output files.
# See :attr:`smrf.output.output_netcdf.OutputNetcdf.LEAST_SIGNIFICANT_DIGITS`
GOLD_RTOL = 1e-4
# Default for gold checks that pass no tolerance
GOLD_DEFAULT_RTOL = 1e-10

# Topo (test_topo.py)
TOPO_RTOL = 1e-7
TOPO_FACTOR_ATOL = 0.025
TOPO_SLOPE_ATOL_LINUX = 1e-12
# macOS slope differs from Linux beyond 1e-12 (commit af882382)
TOPO_SLOPE_ATOL_DARWIN = 1e-6
TOPO_SLOPE_ATOL = (
    TOPO_SLOPE_ATOL_DARWIN if platform == "darwin" else TOPO_SLOPE_ATOL_LINUX
)

# Coordinates
# UTM coordinates in meters, compared to 3 places or decimals, which allows
# about 0.0005 m (places) or 0.0015 m (decimals).
UTM_PLACES = 3
# Basin latitude and longitude in degrees, compared to 6 decimals, which
# allows about 1.5e-6 degrees.
TOPO_LATLON_DECIMAL = 6

# WRF input
WRF_ATOL = 1e-3

# Model physics
# Wet-bulb: the iteration stops within dew_point_tolerance, 0.01 in the test
# config, so compare to 2 decimals.
WET_BULB_DECIMAL = 2
# Snow density in kg/m^3, written as whole numbers, so compare to 0 decimals,
# which allows about 1.5 kg/m^3.
SNOW_DENSITY_DECIMAL = 0
# Solar radiation: reference values are whole W/m^2.
TOPORAD_ATOL = 1
# Two-stream: reference values are printed to six digits.
TWOSTREAM_RTOL = 1e-5

# Math checks
# Within one percent of a rounded reference value.
PERCENT_RTOL = 1e-2
# Exact round-trip checks.
EXACT_RTOL = 1e-12
