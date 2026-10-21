from netCDF4 import Dataset


def add_proj(nc_obj, map_meta=None):
    """
    Adds the projection using two different methods. One from an internet
    source using the epsg value and the other from an existing file containing
    it.
    Args:
        nc_obj: netCDF4 dataset object needing the projection information
        map_meta: Pass in a dictionary of the map meta data directly
    Returns:
        nc_obj: Original nc_bj plus the projection information
    """
    # Create a variable called projection
    nc_obj.createVariable("projection", "S1")
    nc_obj["projection"].setncatts(map_meta)

    # Adding coordinate system info to
    for name, var in nc_obj.variables.items():
        # Assume all 2D+ vars are the same projection
        if "x" in var.dimensions and "y" in var.dimensions:
            nc_obj[name].setncatts({"grid_mapping": "projection"})

        elif name.lower() in ["x", "y"]:
            # Set a standard name, which is required for recognizing
            # projections
            nc_obj[name].setncatts(
                {"standard_name": f"projection_{name.lower()}_coordinate"}
            )
            # Set the units
            nc_obj[name].setncatts({"units": "meters"})

    return nc_obj


def add_proj_from_file(nc_to_copy):
    """
    Use a netcdf file converted from a tif using gdal to retrieve the
    projection information

    Args:
        nc_to_copy: Path to netcdf file to copy the projection information from

    Returns:
        map_meta: dictionary of attributes to add for spatial reference
    """
    with Dataset(nc_to_copy) as ds:
        # Sometimes the projection is called transverse_mercator
        if "transverse_mercator" in ds.variables:
            map_meta = parse_wkt(ds["transverse_mercator"].getncattr("spatial_ref"))
        elif "projection" in ds.variables:
            map_meta = parse_wkt(ds["projection"].getncattr("spatial_ref"))
        else:
            raise Exception("No projection information found in the nc_to_copy")

    return map_meta


def parse_wkt(epsg_string):
    """
    Processes the epsg string returned from the URL request.

    Args:
        epsg_str: String received from the epsg request.

    Returns:
        map_meta: dictionary of projection data to be added to the netcdf
    """
    map_meta = {}

    # Add more projection parsers here
    if "utm" in epsg_string.lower():
        map_meta = gather_utm_meta(epsg_string)

    return map_meta


def gather_utm_meta(epsg_str):
    """
    Use if the EPSG data is associated to UTM
    Gathers the data and returns a dictionary of data and attributes that need
    to be added to the netcdf based on
    https://www.unidata.ucar.edu/software/thredds/current/netcdf-java/reference/StandardCoordinateTransforms.html

    Args:
        epsg_str: String received from the epsg request.

    Returns:
        map_meta: dictionary of UTM projection data to be added to the netcdf
    """
    meta = epsg_str.lower()

    map_meta = {
        "grid_mapping_name": "universal_transverse_mercator",
        "utm_zone_number": None,
        "semi_major_axis": None,
        "inverse_flattening": None,
        "spatial_ref": epsg_str,
        "_CoordinateTransformType": "Projection",
        "_CoordinateAxisTypes": "GeoX GeoY",
    }

    # Assign the zone number
    zone_str = meta.split("zone")[1]
    map_meta["utm_zone_number"] = float(
        (strip_chars(zone_str.split(",")[0])).strip()[-2:]
    )

    # Assign the semi_major_axis
    axis_string = meta.split("spheroid")[1]
    map_meta["semi_major_axis"] = float(axis_string.split(",")[1])

    # Assing the flattening
    map_meta["inverse_flattening"] = float(strip_chars(axis_string.split(",")[2]))

    return map_meta


def strip_chars(edit_str, bad_chars='[(){}<>,"_]=\nns'):
    """
    Written to strip out unwanted chars from the proj strings received
    back from the URL for the EPSG requests.

    Args:
        edit_str: String containing unwanted chars
        bad_chars: String of chars to be removed

    Returns:
        result: The edit_str without any of the chars in bad_chars
    """

    return "".join([s for s in edit_str if s not in bad_chars])
