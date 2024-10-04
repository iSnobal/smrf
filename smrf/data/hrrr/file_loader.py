import os
from datetime import timedelta

import pandas as pd
import utm
import xarray as xr

from .file_handler import FileHandler
from .grib_file import GribFile


class FileLoader():
    """
    Load data from local HRRR files.
    Currently supports loading from Grib format.
    """
    # Maximum hour that local files will be attempted to be read if a previous
    # hour could not be found or successfully loaded.
    MAX_FORECAST_HOUR = 6
    NEXT_HOUR = timedelta(hours=1)

    def __init__(self,
                 file_dir,
                 file_type='grib2',
                 external_logger=None,
                 load_wind=False,
         ):
        """
        :param file_dir:        Base directory to location of files
        :param file_type:       Determines how to read the files.
                                Default: grib2
        :param external_logger: (Optional) Specify an existing logger instance
        :load_wind:             Flag to load HRRR wind data (Default: False)
        """
        self.log = external_logger

        self.file_type = file_type
        self._var_map = self.load_var_map(load_wind)
        self.file_dir = file_dir

    @property
    def file_dir(self):
        return self._file_dir

    @file_dir.setter
    def file_dir(self, value):
        self._file_dir = value

    @property
    def file_type(self):
        return self._file_loader.SUFFIX

    @file_type.setter
    def file_type(self, value):
        if value == GribFile.SUFFIX:
            self._file_loader = GribFile(external_logger=self.log)
        else:
            raise Exception('Unknown file type argument')

    @property
    def file_loader(self):
        return self._file_loader

    def load_var_map(self, load_wind=False):
        """
        Filter and return the desired HRRR variables to read

        Args:
            load_wind: (Boolean) Whether to load HRRR wind data.
                       (Default: False)

        Returns:
            Dict - HRRR variables with keys mapped to SMRF variable names.
        """
        if not load_wind:
            variables = [
                v for v in self.file_loader.VARIABLES
                if v not in self.file_loader.WIND_VARIABLES
            ]
        else:
            variables = self.file_loader.VARIABLES

        return {
            key: self.file_loader.VAR_MAP[key] for key in variables
        }

    def get_saved_data(self, start_date, end_date, bbox, utm_zone_number):
        """
        Get the saved data from above for a particular time and a particular
        bounding box.

        Args:
            start_date: datetime for the start of the data loading period
            end_date:   datetime for the end of the data loading period
            bbox:       list of  [lonmin, latmin, lonmax, latmax]
            utm_zone_number: UTM zone number to convert datetime to

        Returns:
            List containing dataframe for the metadata adn for each read
            variable.
        """

        if start_date > end_date:
            raise ValueError('start_date before end_date')

        self.file_loader.bbox = bbox

        self.log.info('Getting saved data')
        self.get_data(start_date, end_date)

        return self.convert_to_dataframes(utm_zone_number)

    def get_data(self, start_date, end_date):
        """
        Get the HRRR data for given start and end date.
        Read data is stored on instance attribute.

        Args:
            start_date: datetime for the start of the data loading period
            end_date:   datetime for the end of the data loading period
        """
        date = start_date
        data = []

        while date <= end_date:
            self.log.debug('Reading file for date: {}'.format(date))
            forecast_data = None

            # make sure we get a working file. This allows for six tries,
            # accounting for the fact that we start at forecast hour 1
            file_time = date
            for fx_hr in range(1, self.MAX_FORECAST_HOUR + 1):
                day_folder, file_name = FileHandler.folder_and_file(
                    file_time, fx_hr, self.file_type
                )

                try:
                    if self.file_type == GribFile.SUFFIX:
                        base_path = os.path.abspath(self.file_dir)
                        file = os.path.join(base_path, day_folder, file_name)
                        if os.path.exists(file):
                            forecast_data = self.file_loader.load(
                                file, self._var_map
                            )
                        else:
                            self.log.error('  No file for {}'.format(file))

                except Exception as e:
                    self.log.debug(e)
                    self.log.debug(
                        '  Could not load forecast hour {} for date {} '
                        'successfully'.format(fx_hr, date)
                    )

                if fx_hr == self.MAX_FORECAST_HOUR:
                    raise IOError(
                        'Not able to find good file for {}'
                        .format(file_time.strftime('%Y-%m-%d %H:%M'))
                    )

                if forecast_data is not None:
                    data += forecast_data
                    break

            date += self.NEXT_HOUR

        try:
            self.data = xr.combine_by_coords(data)
        except Exception as e:
            self.log.debug(e)
            self.log.debug(
                '  Could not combine forecast data for given dates: {} - {}'
                    .format(start_date, end_date)
            )

    def convert_to_dataframes(self, utm_zone_number):
        """
        Convert the xarray's to a pandas dataframes

        Args:
            utm_zone_number: UTM zone number to convert datetime to

        Returns
            Tuple of metadata and dataframe
        """
        metadata = None
        dataframe = {}

        for key, value in self._var_map.items():
            if self.file_type == GribFile.SUFFIX:
                df = self.data[key].to_dataframe()
            else:
                df = self.data[value].to_dataframe()
                key = value

            # convert from a row multi-index to a column multi-index
            df = df.unstack(level=[1, 2])

            # Get the metadata using the elevation variables
            if key == 'elevation':
                if self.file_type == GribFile.SUFFIX:
                    value = key

                metadata = []
                for mm in ['latitude', 'longitude', value]:
                    dftmp = df[mm].copy()
                    dftmp.columns = self.format_column_names(dftmp)
                    dftmp = dftmp.iloc[0]
                    dftmp.name = mm
                    metadata.append(dftmp)

                metadata = pd.concat(metadata, axis=1)
                metadata = metadata.apply(
                    FileLoader.apply_utm,
                    args=(utm_zone_number,),
                    axis=1
                )
                metadata.rename(columns={value: key}, inplace=True)

            else:
                df = df.loc[:, key]

                df.columns = self.format_column_names(df)
                df.index.rename('date_time', inplace=True)

                df.dropna(axis=1, how='all', inplace=True)
                df.sort_index(axis=0, inplace=True)
                dataframe[key] = df

                # manipulate data in necessary ways
                if key == 'air_temp':
                    dataframe['air_temp'] -= 273.15
                if key == 'cloud_factor':
                    dataframe['cloud_factor'] = \
                        1 - dataframe['cloud_factor'] / 100

        # the metadata may have more columns than the dataframes
        c = []
        for key in dataframe.keys():
            c.extend(list(dataframe[key].columns.values))

        metadata = metadata[metadata.index.isin(list(set(c)))]

        return metadata, dataframe

    @staticmethod
    def format_column_names(dataframe):
        """
        Make new names for the columns as grid_y_x

        :param dataframe:
        :return: Array - New column names including the y and x GRIB pixel
                         index. Example: grid_0_1 for y at 0 and x at 1
        """
        return [
            'grid_{c[0]}_{c[1]}'.format(c=col)
            for col in dataframe.columns.to_flat_index()
        ]

    @staticmethod
    def apply_utm(dataframe, utm_zone_number):
        """
        Ufunc to calculate the utm from lat/lon for a series

        Args:
            dataframe: pandas series with fields latitude and longitude
            utm_zone_number: Zone number to force to

        Returns:
            Pandas series entry with fields 'utm_x' and 'utm_y' filled
        """
        # HRRR has longitude reporting in degrees from the east
        dataframe['longitude'] -= 360

        (dataframe['utm_x'], dataframe['utm_y'], *unused) = utm.from_latlon(
            dataframe['latitude'],
            dataframe['longitude'],
            force_zone_number=utm_zone_number
        )

        return dataframe
