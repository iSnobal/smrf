from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator, Tuple

from osgeo import gdal, osr
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
        # Topo related
        self._dstSRS = None
        self._outputBoundsSRS = None
        self._outputBounds = None
        self._xRes = None
        self._yRes = None

        self.resample_method = self.RESAMPLING_METHODS[resample_method]

        self.get_spatial_info(topo.file)


    def get_spatial_info(self, filename: str) -> None:
        """
        Grabbing the spatial information (projection, bounds) from the first dataset
        and set instance variables to use with the warp and cut call.

        :param filename: Path to topo.nc file
        """
        spatial_info = osr.SpatialReference()

        with gdal.Open(filename, gdal.GA_ReadOnly) as topo:
            with gdal.Open(topo.GetSubDatasets()[0][0], gdal.GA_ReadOnly) as dataset:
                spatial_info.SetFromUserInput(dataset.GetProjection())

                self._dstSRS = self.gdal_osr_authority(spatial_info)
                self._outputBoundsSRS = self.gdal_osr_authority(spatial_info)
                self._outputBounds = self.gdal_output_bounds(dataset)
                self._xRes = dataset.GetGeoTransform()[1]
                self._yRes = dataset.GetGeoTransform()[1]

    @staticmethod
    def gdal_osr_authority(spatial_info: osr.SpatialReference) -> str:
        """
        Construct and projection string with format "EPSG:XXXXX" from the given
        spatial reference.

        :param spatial_info: Spatial reference object of the topo file

        :return: str - Spatial reference
        """
        return f"{spatial_info.GetAuthorityName(None)}:{spatial_info.GetAuthorityCode(None)}"

    @staticmethod
    def gdal_output_bounds(topo: gdal.Dataset) -> list[float]:
        """
        Get bounding box of the topo file in the order of [xmin, ymin, xmax, ymax]

        :param topo: Opened topo file as GDAL dataset

        :return: list of corner coordinates
        """
        geo_transform = topo.GetGeoTransform()
        return [
            geo_transform[0],
            geo_transform[3] + geo_transform[5] * topo.RasterYSize,
            geo_transform[0] + geo_transform[1] * topo.RasterXSize,
            geo_transform[3]
        ]

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
            dstSRS=self._dstSRS,
            outputBoundsSRS=self._outputBoundsSRS,
            outputBounds=self._outputBounds,
            xRes=self._xRes,
            yRes=self._yRes,
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
