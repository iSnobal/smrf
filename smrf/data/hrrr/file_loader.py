import os
from datetime import datetime, timedelta
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import utm
import xarray as xr
from smrf.data.load_topo import Topo

from .file_handler import FileHandler
from .grib_file_gdal import GribFileGdal
from .grib_file_xarray import GribFileXarray


class FileLoader:
    """
    Load data from local HRRR GRIB files.
    """
    NEXT_HOUR = timedelta(hours=1)
    SIXTH_HOUR = 6
    NAME_SUFFIX = 'grib2'

    def __init__(
        self,
        file_dir,
        forecast_hour,
        sixth_hour_variables=None,
        load_gdal: list[str] = None,
        gdal_algorithm: str = None,
        external_logger=None,
        load_wind=False,
    ):
        """
        :param file_dir:        Base directory to location of files
        :param forecast_hour:   HRRR forecast hour to load forcing data from
        :param sixth_hour_variables:   HRRR forecast hour to load data from
                                the sixth forecast hour
        :param load_gdal:       Variable list to load via GDAL
        :param gdal_algorithm:  Interpolation algorithm to use for GDAL
        :param external_logger: (Optional) Specify an existing logger instance
        :param load_wind:       Flag to load HRRR wind data (Default: False)
        """
        self.log = external_logger

        self.file_dir = file_dir

        self._load_wind = load_wind
        self._forecast_hour = forecast_hour
        self._sixth_hour_variables = sixth_hour_variables
        self._load_gdal = (load_gdal or [])
        self._gdal_algorithm = (gdal_algorithm or GribFileGdal.DEFAULT_ALGORITHM)

    @property
    def file_dir(self):
        return self._file_dir

    @file_dir.setter
    def file_dir(self, value):
        self._file_dir = os.path.abspath(value)

    def data_for_time_and_topo(
        self,
        start_date: datetime,
        bbox: list[float],
        topo: Topo,
        utm_zone_number: int,
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Load variables from HRRR either using Xarray (default) or GDAL.

        :param start_date: datetime - Day to load
        :param bbox: list - Bounding box of the domain to load (used with Xarray)
        :param topo: Topo - Instance of the loaded topo file (used with GDAL)
        :param utm_zone_number: Int - UTM zone number to convert dataframe (Xarray)

        :return:
          Tuple - Dataframe with metadata and dictionary with variable names as keys
        """
        metadata, data = self.xarray(start_date, bbox, utm_zone_number)
        if len(self._load_gdal) > 0:
            data |= self.gdal(start_date, topo)

        return metadata, data


    def _get_file_path(self, file_time, forecast_hour):
        """
        Construct an absolute file path to a HRRR file

        :param file_time:       Date and time of file to load
        :param forecast_hour:   Forecast hour to load

        :return: (String) Absolute file path
        """
        day_folder, file_name = FileHandler.folder_and_file(
            file_time, forecast_hour, self.NAME_SUFFIX
        )
        return os.path.join(self.file_dir, day_folder, file_name)

    def _check_sixth_hour_presence(self, date):
        """
        Check that a sixth hour file is present when a variable is configured
        to be loaded from there

        :param date:        Date to load

        :return: (Boolean) Whether the sixth hour file is present
        """
        if self._sixth_hour_variables:
            file = self._get_file_path(date, self.SIXTH_HOUR)
            if os.path.exists(file):
                return file
            else:
                return False
        else:
            return True

    def xarray(
        self,
        date: datetime,
        bbox: list[float],
        utm_zone_number: int,
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Get the data for given time range and a specified bounding box using Xarray.
        Read data is stored on instance attribute.

        Args:
            date:            datetime for the start of the data loading period
            bbox:            list of  [lonmin, latmin, lonmax, latmax]
            utm_zone_number: UTM zone number to convert datetime to

        Returns:
            List containing dataframe for the metadata and for each read
            variable.
        """
        file_loader = GribFileXarray(external_logger=self.log)
        file_loader.bbox = bbox

        self.log.info("Getting saved data")

        self.log.debug("Reading file for date: {}".format(date))

        # Filename of the default configured forecast hour
        default_file = self._get_file_path(date, self._forecast_hour)
        # Filename for variables that are mapped to the sixth
        # forecast hour
        sixth_hour_file = self._check_sixth_hour_presence(date)

        try:
            if os.path.exists(default_file) and sixth_hour_file:
                data = file_loader.load(
                    file=default_file,
                    load_wind=self._load_wind,
                    sixth_hour_file=sixth_hour_file,
                    sixth_hour_variables=self._sixth_hour_variables,
                )
            else:
                raise FileNotFoundError(
                    "  Not able to find file for datetime: {}".format(
                        date.strftime("%Y-%m-%d %H:%M")
                    )
                )
        except Exception as e:
            self.log.error(
                "  Could not load forecast for date {} successfully".format(date)
            )
            raise e

        try:
            return self.convert_to_dataframes(data, utm_zone_number)
        except Exception as e:
            self.log.debug(
                '  Could not combine forecast data for given date: {} '
                .format(date)
            )
            raise e

    def convert_to_dataframes(
        self, data: xr.DataArray, utm_zone_number: int
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Convert the xarray's to a pandas dataframes

        Args:
            data: Xarray data object to convert
            utm_zone_number: UTM zone number to convert datetime to

        Returns
            Tuple of metadata and dataframe
        """
        metadata = None
        dataframe = {}

        for variable in list(data.data_vars):
            df = data[variable].to_dataframe()

            # convert from a row multi-index to a column multi-index
            df = df.unstack(level=[1, 2])

            # Get the metadata using the elevation variables
            if variable == "elevation":
                metadata = []
                for mm in ["latitude", "longitude", variable]:
                    dftmp = df[mm].copy()
                    dftmp.columns = self.format_column_names(dftmp)
                    dftmp = dftmp.iloc[0]
                    dftmp.name = mm
                    metadata.append(dftmp)

                metadata = pd.concat(metadata, axis=1)
                metadata = metadata.apply(
                    FileLoader.apply_utm, args=(utm_zone_number,), axis=1
                )
            else:
                df = df.loc[:, variable]

                df.columns = self.format_column_names(df)
                df.index.rename("date_time", inplace=True)

                df.dropna(axis=1, how="all", inplace=True)
                df.sort_index(axis=0, inplace=True)
                dataframe[variable] = df

                # manipulate data in necessary ways
                if variable == "air_temp":
                    dataframe["air_temp"] -= 273.15
                if variable == "cloud_factor":
                    dataframe["cloud_factor"] = 1 - dataframe["cloud_factor"] / 100

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

    def gdal(self, date: datetime, topo: Topo) -> dict[str, np.ndarray]:
        """
        Load data for given date and time and topo config using GDAL

        Args:
            date: datetime for the start of the data loading period
            topo: Instance of Topo class

        :return
            Dict - Loaded variable data transformed to topo
        """
        file_loader = GribFileGdal(topo, self._gdal_algorithm)

        return file_loader.load(
            self._load_gdal, self._get_file_path(date, self.SIXTH_HOUR)
        )
