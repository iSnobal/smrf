import xarray as xr

from .grib_file_variables import (FIRST_HOUR, HRRR_HAG_10, HRRR_HAG_2,
                                  HRRR_SURFACE, SIXTH_HOUR)


class GribFileXarray:
    """
    Class to load a GRIB2 file from disk using Xarray.
    """
    SUFFIX = 'grib2'

    CELL_SIZE = 3000  # in meters

    HRRR_VARIABLES = [HRRR_SURFACE, HRRR_HAG_2, HRRR_HAG_10]

    def __init__(self, external_logger=None):
        self._bbox = None
        self.log = external_logger

    @property
    def bbox(self):
        return self._bbox

    @bbox.setter
    def bbox(self, value):
        self._bbox = value

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

    @staticmethod
    def _prepare_for_merge(dataset, new_names, level):
        """
        Remove some dimensions so all read variables can be combined into
        one dataset later

        :param dataset:   Dataset to remove information from
        :param new_names: Dict - Map old to new variable names
        :param level:     String - Level name attribute to remove

        :return:
          xr.Dataset - Prepared Dataset for merge
        """
        if len(dataset.variables) == 0:
            return None

        del dataset[level]
        del dataset['step']

        # rename the grib variable name to a SMRF recognized variable name
        dataset = dataset.rename(new_names)

        # Make the time an index coordinate
        dataset = dataset.assign_coords(time=dataset['valid_time'])
        dataset = dataset.expand_dims('time')
        del dataset['valid_time']

        return dataset

    def _load_variable_level(
        self, file, filter_by_keys, smrf_mapping, level_name=None
    ):
        """
        Load HRRR variables with given arguments.
        Uses variable_key value if a level value is not passed.

        :param file:           String - File to load
        :param filter_by_keys: Dict - Arguments to pass to xarray
        :param smrf_mapping:   String - Dict key of mapped HRRR-SMRF variables
        :param level_name:     String - HRRR level name

        :return:
            xr.Dataset
        """
        return (
            # Prepare for merging of all variables in a successive step
            self._prepare_for_merge(
                # Read the file
                xr.open_dataset(
                    file,
                    engine='cfgrib',
                    backend_kwargs={
                        'filter_by_keys': filter_by_keys,
                        'indexpath': '',  # Don't create an .idx file when reading
                    }
                ),
                smrf_mapping,
                level_name,
            )
        )

    @staticmethod
    def _first_or_sixth_variable(smrf_map, sixth_hour_variables):
        """
        Split the HRRR variables by requested forecast hour

        :param smrf_map: Dict - Mapping HRRR to SMRF keys
        :param sixth_hour_variables: list - List of variables to load from the
                                     sixth forecast hour
        :return:
            (List, List) - Variables to load for first, sixth hour
        """
        if sixth_hour_variables is not None:
            first_hour_variables = [
                hrrr_key for hrrr_key, smrf_key in smrf_map.items()
                if smrf_key not in sixth_hour_variables
            ]
            sixth_hour_variables = [
                hrrr_key for hrrr_key, smrf_key in smrf_map.items()
                if smrf_key in sixth_hour_variables
            ]
            return first_hour_variables, sixth_hour_variables
        else:
            return list(smrf_map.keys()), None

    def load(
        self, file, sixth_hour_file, sixth_hour_variables=None, load_wind=False
    ):
        """
        Get valid HRRR data using Xarray

        :param file:                 Path to grib2 file to open
        :param sixth_hour_file:      Path to HRRR grib file of the sixth hour
                                     forecast
        :param sixth_hour_variables: List of variables that are loaded from the
                                     sixth hour forecast. Default: None
        :param load_wind:            Flag to indicate loading the wind fields
                                     Default: False

        Returns:
            Array with Xarray Datasets for each variable and
            cropped to bounding box
        """
        variable_data = []
        # For checking that we loaded all the variables we requested
        loaded_variables = []

        self.log.debug('Reading {}'.format(file))

        for variable in self.HRRR_VARIABLES:
            if variable == HRRR_HAG_10 and not load_wind:
                continue

            first_hour, sixth_hour = self._first_or_sixth_variable(
                variable.smrf_map, sixth_hour_variables
            )
            variable_data.append(
                self._load_variable_level(
                    file,
                    {
                        variable.grib_identifier: first_hour,
                        **variable.grib_keys,
                        **FIRST_HOUR,
                    },
                    {key: value for key, value in variable.smrf_map.items()
                     if key in first_hour},
                    variable.level,
                )
            )
            loaded_variables += first_hour

            if sixth_hour is not None and sixth_hour:
                variable_data.append(
                    self._load_variable_level(
                        sixth_hour_file,
                        {
                            variable.grib_identifier: sixth_hour,
                            **variable.grib_keys,
                            **SIXTH_HOUR,
                        },
                        {key: value for key, value in variable.smrf_map.items()
                         if key in sixth_hour},
                        variable.level,
                    )
                )
                loaded_variables += sixth_hour

        try:
            variable_data = xr.combine_by_coords(
                variable_data, combine_attrs='drop'
            )
        except TypeError:
            self.log.error('Not all grib files were successfully read')
            raise Exception()

        if len(variable_data.data_vars) is not len(loaded_variables):
            self.log.error(
                'Not all requested variables were found in the grib files'
            )
            raise Exception()

        variable_data = variable_data.where(
            (variable_data.latitude >= self.bbox[1]) &
            (variable_data.latitude <= self.bbox[3]) &
            (variable_data.longitude >= self.longitude_east(self.bbox[0])) &
            (variable_data.longitude <= self.longitude_east(self.bbox[2])),
            drop=True
        )

        return variable_data
