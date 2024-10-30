import copy

import xarray as xr


class GribFile():
    """
    Class to load a GRIB2 file from disk.

    The VAR_MAP class constants holds a mapping for currently available
    variables that are loadable from a file.
    """
    SUFFIX = 'grib2'

    CELL_SIZE = 3000  # in meters

    SURFACE = {
        'level': 0,
        'typeOfLevel': 'surface',
    }
    SURFACE_VARIABLES = {
        'precip_int': {
            'name': 'Total Precipitation',
            'shortName': 'tp',
            **SURFACE,
        },
        'short_wave': {
            'stepType': 'instant',
            'cfVarName': 'sdswrf',
            **SURFACE,
        },
        'elevation': {
            'cfVarName': 'orog',
            **SURFACE,
        }
    }
    # HAG - Height Above Ground
    HAG_2 = {
        'level': 2,
        'typeOfLevel': 'heightAboveGround',
    }
    HAG_2_VARIABLES = {
        'air_temp': {
            'cfName': 'air_temperature',
            'cfVarName': 't2m',
            **HAG_2,
        },
        'relative_humidity': {
            'cfVarName': 'r2',
            **HAG_2,
        },
    }
    WIND_U = 'wind_u'
    WIND_V = 'wind_v'
    WIND_VARIABLES = [WIND_U, WIND_V]
    HAG_10 = {
        'level': 10,
        'typeOfLevel': 'heightAboveGround',
    }
    HAG_10_VARIABLES = {
        WIND_U: {
            'cfVarName': 'u10',
            **HAG_10,
        },
        WIND_V: {
            'cfVarName': 'v10',
            **HAG_10,
        },
    }
    VAR_MAP = {
        **SURFACE_VARIABLES,
        **HAG_2_VARIABLES,
        **HAG_10_VARIABLES,
    }
    VARIABLES = VAR_MAP.keys()

    def __init__(self, external_logger=None):
        self._bbox = None
        self.log = external_logger

    @property
    def bbox(self):
        return self._bbox

    @bbox.setter
    def bbox(self, value):
        self._bbox = value

    @property
    def variable_map(self):
        return copy.deepcopy(self.VAR_MAP)

    @staticmethod
    def longitude_east(longitude):
        """
        HRRR references the longitudes starting from the east.
        When cropping to a longitude, the reference coordinate needs
        to be adjusted for that.

        :param longitude:
        :return: longitude from east
        """
        return longitude % 360

    def load(self, file, var_map):
        """
        Get valid HRRR data using Xarray

        Args:
            file:    Path to grib2 file to open
            var_map: Var map of variables to load from file

        Returns:
            Array with Xarray Datasets for each variable and
            cropped to bounding box
        """

        variable_data = []

        self.log.debug('Reading {}'.format(file))

        # open just one dataset at a time
        for key, params in var_map.items():

            # TODO - Remove special casing and harden logic around 6th forecast
            # hour
            if key == 'precip_int':
                file = file.replace(self.SUFFIX, 'apcp06.' + self.SUFFIX)
            else:
                file = file.replace('apcp06.', '')

            data = xr.open_dataset(
                file,
                engine='cfgrib',
                backend_kwargs={
                    'filter_by_keys': params,
                    'indexpath': '',  # Don't create an .idx file when reading
                }
            )

            if len(data) > 1:
                raise Exception('More than one grib variable returned')

            data = data.where(
                (data.latitude >= self.bbox[1]) &
                (data.latitude <= self.bbox[3]) &
                (data.longitude >= self.longitude_east(self.bbox[0])) &
                (data.longitude <= self.longitude_east(self.bbox[2])),
                drop=True
            )

            # Remove some dimensions so all read variables can
            # be combined into one dataset
            del data[params['typeOfLevel']]
            del data['step']

            # rename the data variable
            variable = params.get('cfVarName') or params.get('shortName')
            data = data.rename({variable: key})

            # Make the time an index coordinate
            data = data.assign_coords(time=data['valid_time'])
            data = data.expand_dims('time')
            del data['valid_time']

            variable_data.append(data)

            data.close()

        return variable_data
