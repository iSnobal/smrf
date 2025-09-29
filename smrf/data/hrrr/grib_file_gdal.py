from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator, Tuple

from osgeo import gdal
from smrf.data.load_topo import Topo

gdal.UseExceptions()


class GribMetadata:
    """
    GRIB_REF_TIME is in UTC, indicated by the 'Z' in field GRIB_IDS
    GRIB_VALID_TIME is the timestamp indicating the 'up to'
    """
    VARIABLE_NAME = "GRIB_ELEMENT"
    DATETIME = "GRIB_VALID_TIME"


class GribFileGdal:
    """
    Load data from the High Resolution Rapid Refresh (HRRR) model using the GDAL library.
    """
    DEFAULT_ALGORITHM = "cubic"

    # Mapping from string input to GDAL resampling algorithms
    RESAMPLING_METHODS = {
        "nearest": gdal.GRA_NearestNeighbour,
        "bilinear": gdal.GRA_Bilinear,
        DEFAULT_ALGORITHM: gdal.GRA_Cubic,
        "cubic_spline": gdal.GRA_CubicSpline,
    }

    def __init__(self, topo: Topo, resample_method:str):
        self.resample_method = self.RESAMPLING_METHODS[resample_method]
        self.topo = topo

    @staticmethod
    def get_grib_metadata(grib_file: str) -> Tuple[dict[str, int], dict[str, datetime]]:
        """
        Create mapping of GRIB variable name to band number and a second dict with
        the datetime it is valid for.

        This date times are properly calculated for forecast the first and sixth hour.

        :return:
            Tuple - Dict with band numbers and dict with datetime. Both times the HRRR
                    variable names are the keys.
        """
        band_map = {}
        valid_time = {}

        with gdal.Open(grib_file, gdal.GA_ReadOnly) as grib:
            for band in range(1, grib.RasterCount + 1):
                metadata = grib.GetRasterBand(band).GetMetadata()
                band_map[metadata[GribMetadata.VARIABLE_NAME]] = band
                valid_time[metadata[GribMetadata.VARIABLE_NAME]] = (
                    datetime.fromtimestamp(
                        int(metadata[GribMetadata.DATETIME])
                    ).astimezone(timezone.utc)
                )

        return band_map, valid_time

    @contextmanager
    def gdal_warp_and_cut(self, in_file: str) -> Generator[gdal.Dataset, None, None]:
        """
        Cut and warp the band for given grib file to the topo bounds and projection

        :param in_file: str - HRRR file to load

        :returns:
            gdal.Dataset in a context block
        """
        options = gdal.WarpOptions(
            dstSRS=self.topo.gdal_attributes.srs,
            outputBoundsSRS=self.topo.gdal_attributes.srs,
            outputBounds=self.topo.gdal_attributes.outputBounds,
            xRes=self.topo.gdal_attributes.xRes,
            yRes=self.topo.gdal_attributes.yRes,
            resampleAlg=self.resample_method,
            multithread=True,
            format='MEM',
        )

        # Using a blank '' destination warps the file in memory
        dataset = gdal.Warp('', in_file, options=options)

        yield dataset

        # Release the memory
        dataset.FlushCache()
        dataset = None # noqa

    def load(self, variables: list[str], grib_file: str) -> dict:
        """
        Get data from file for given variables.

        :param variables: List of variables to extract
        :param grib_file: Path to HRRR grib2 file

        :returns:
            Dict with HRRR variable names as keys and values from the grib file as
            numpy array interpolated to the topo grid.
        """
        data = {}
        band_map, _valid_time = self.get_grib_metadata(grib_file)

        with self.gdal_warp_and_cut(grib_file) as dataset:
            for variable in variables:
                band_number = band_map[variable]
                data[variable] = dataset.GetRasterBand(band_number).ReadAsArray()

        return data
