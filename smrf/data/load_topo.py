import logging
from dataclasses import dataclass

import numpy as np
from netCDF4 import Dataset
from osgeo import gdal, osr
from topocalc import gradient
from topocalc.viewf import viewf
from utm import to_latlon


@dataclass
class GdalAttributes:
    # Helper class to store attributes from GDAL:
    #  * Spatial Reference
    #  * Domain Bounds
    #  * Spatial Resolution (X and Y)
    srs: str
    outputBounds: list
    xRes: float
    yRes: float


class Topo:
    """
    Class for topo images and processing those images. Images are:
    - DEM
    - Mask
    - veg type
    - veg height
    - veg k
    - veg tau

    Inputs to topo are the topo section of the config file

    """

    IMAGES = ['dem', 'mask', 'veg_type', 'veg_height', 'veg_k', 'veg_tau']

    def __init__(self, topoConfig):
        self.topoConfig = topoConfig

        self._logger = logging.getLogger(__name__)
        self._logger.info('Reading [TOPO] and making stoporad input')

        self.readNetCDF()

        # Set attributes used for GDAL warp and cut
        self.gdal_attributes = self.read_gdal_attributes()

        # calculate the gradient
        self.gradient()

    @property
    def file(self):
        return self.topoConfig['filename']

    def readNetCDF(self):
        """
        Read in the images from the config file where the file
        listed is in netcdf format
        """

        # read in the images
        f = Dataset(self.file, 'r')

        # netCDF>1.4.0 returns as masked arrays even if no missing values
        # are present. This will ensure that if the array has no missing
        # values, a normal numpy array is returned
        f.set_always_mask(False)

        if 'projection' not in f.variables.keys():
            raise IOError("Topo input files must have projection information")

        self.readImages(f)

        # get some general information about the model domain from the dem
        self.nx = f.dimensions['x'].size
        self.ny = f.dimensions['y'].size

        # create the x,y vectors
        self.x = f.variables['x'][:]
        self.y = f.variables['y'][:]
        [self.X, self.Y] = np.meshgrid(self.x, self.y)

        # There is not a great NetCDF convention on direction for the y-axis.
        # So there is the possibility that the dy will be positive or negative.
        # For the gradient calculations this needs to be absolute spacing.
        self.dx = np.mean(np.diff(self.x))
        self.dy = np.abs(np.mean(np.diff(self.y)))

        # Calculate the center of the basin
        self.cx, self.cy = self.get_center(f, mask_name='mask')

        # Is the modeling domain in the northern hemisphere
        self.northern_hemisphere = self.topoConfig['northern_hemisphere']

        # Assign the UTM zone
        self.zone_number = int(f.variables['projection'].utm_zone_number)

        # Calculate the lat long
        self.basin_lat, self.basin_long = to_latlon(
            self.cx,
            self.cy,
            self.zone_number,
            northern=self.northern_hemisphere)

        self._logger.info('Domain center in UTM Zone {:d} = {:0.1f}m, {:0.1f}m'
                          ''.format(self.zone_number, self.cx, self.cy))
        self._logger.info('Domain center as Latitude/Longitude = {:0.5f}, '
                          '{:0.5f}'.format(self.basin_lat, self.basin_long))

        # Load or calculate the sky view factor
        if 'sky_view_factor' in f.variables:
            self.sky_view_factor = f['sky_view_factor'][:]
            self.terrain_config_factor = f['terrain_config_factor'][:]
            f.close()
        else:
            f.close()
            self.calculate_sky_view_factor()

    def read_gdal_attributes(self) -> GdalAttributes:
        """
        Get the spatial information from the topo for warping and cutting via GDAL.

        :returns
            GdalAttributes - attributes read via GDAL
        """
        spatial_info = osr.SpatialReference()

        with gdal.Open(self.file, gdal.GA_ReadOnly) as topo:
            with gdal.Open(topo.GetSubDatasets()[0][0], gdal.GA_ReadOnly) as dataset:
                spatial_info.SetFromUserInput(dataset.GetProjection())

                return GdalAttributes(
                    srs = Topo.gdal_osr_authority(spatial_info),
                    outputBounds = self.gdal_output_bounds(dataset),
                    xRes = dataset.GetGeoTransform()[1],
                    yRes = dataset.GetGeoTransform()[1],
                )

    def readImages(self, f):
        """Read images from the netcdf and set as attributes in the Topo class

        Args:
            f: netcdf dataset object
        """

        # netCDF files are stored typically as 32-bit float, so convert
        # to double or int
        for v_smrf in self.IMAGES:

            if v_smrf in f.variables.keys():
                if v_smrf == 'veg_type':
                    result = f.variables[v_smrf][:].astype(int)
                else:
                    result = f.variables[v_smrf][:].astype(np.float64)

            setattr(self, v_smrf, result)

    def get_center(self, ds, mask_name=None):
        '''
        Function returns the basin center in the native coordinates of the
        a netcdf object.

        The incoming data set must contain at least and x, y and optionally
        whatever mask name the user would like to use for calculating .
        If no mask name is provided then the entire domain is used.

        Args:
            ds: netCDF4.Dataset object containing at least x,y, optionally
                    a mask variable name
            mask_name: variable name in the dataset that is a mask where 1 is
                    in the mask
        Returns:
            tuple: x,y of the data center in the datas native coordinates
        '''
        x = ds.variables['x'][:]
        y = ds.variables['y'][:]

        # Calculate the center of the basin
        if mask_name is not None:
            mask_id = np.argwhere(ds.variables[mask_name][:] == 1)

            # Tuple is required for an upcoming deprecation in numpy
            idx = tuple([mask_id[:, 1]])
            idy = tuple([mask_id[:, 0]])

            x = x[idx]
            y = y[idy]

        return x.mean(), y.mean()

    def gradient(self):
        """
        Calculate the gradient and aspect
        """

        func = self.topoConfig['gradient_method']

        # calculate the gradient and aspect
        g, a = getattr(gradient, func)(
            self.dem, self.dx, self.dy, aspect_rad=True)
        self.slope_radians = g

        # following IPW convention for slope as sin(Slope)
        self.sin_slope = np.sin(g)
        self.aspect = a

    def calculate_sky_view_factor(self):
        """
        Calculate the sky_view_factor and terrain_config_factor and store
        with the topo config file. Saves time when running through a water
        year.
        """
        self.gradient()

        svf, tcf = viewf(
            self.dem,
            self.dx,
            nangles=self.topoConfig['sky_view_factor_angles'],
            sin_slope=self.sin_slope,
            aspect=self.aspect
        )

        topo = Dataset(self.topoConfig['filename'], 'r+')

        sky_view_factor = topo.createVariable(
            'sky_view_factor', 'f8', ('y', 'x',), zlib=True
        )
        sky_view_factor.setncattr(
            'long_name',
            f"Sky view factor for "
            f"{self.topoConfig['sky_view_factor_angles']} angles"
        )

        terrain_config_factor = topo.createVariable(
            'terrain_config_factor', 'f8', ('y', 'x',), zlib=True
        )
        terrain_config_factor.setncattr(
            'long_name',
            f"Terrain config factor for "
            f"{self.topoConfig['sky_view_factor_angles']} angles"
        )

        slope = topo.createVariable(
            'slope', 'f8', ('y', 'x',), zlib=True
        )
        slope.setncattr('long_name', f"Slope angle (degrees)")

        sky_view_factor[:, :] = svf
        terrain_config_factor[:, :] = tcf
        slope[:, :] = np.degrees(self.slope_radians)

        topo.close()

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
