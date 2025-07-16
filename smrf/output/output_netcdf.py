import logging
import os
from datetime import datetime

import netCDF4 as nc
import numpy as np
from spatialnc.proj import add_proj, add_proj_from_file

from smrf import __version__


class OutputNetcdf:
    """
    Class OutputNetcdf() to output values to a netCDF file
    """

    type = 'netcdf'
    fmt = '%Y-%m-%d %H:%M:%S'
    COMPRESSION = dict(compression="zlib", complevel=4)
    DIMENSIONS = ("time", "y", "x")

    def __init__(self, variable_dict, topo, time, outConfig):
        """
        Initialize the OutputNetcdf() class

        Args:
            variable_dict: dict of variable information
            topo: loadTopo instance
            time: time configuration
            outConfig: out configuration
        """

        self._logger = logging.getLogger(__name__)

        # go through the variable list and make full file names
        for v in variable_dict:
            variable_dict[v]['file_name'] = \
                str(variable_dict[v]['out_location'] + '.nc')

        self.variable_dict = variable_dict

        # process the time section
        self.run_time_step = int(time['time_step'])
        self.out_frequency = int(outConfig['frequency'])
        self.outConfig = outConfig

        # determine the x,y vectors for the netCDF file
        x = topo.x
        y = topo.y
        self.mask = topo.mask

        self.date_time = {}
        now_str = datetime.now().strftime(self.fmt)

        # Retrieve projection information from topo
        map_meta = add_proj_from_file(topo.topoConfig["filename"])

        precision = self.outConfig["netcdf_output_precision"][0]

        for v in self.variable_dict:
            f = self.variable_dict[v]

            if os.path.isfile(f["file_name"]):
                self._logger.warning(
                    "Opening {}, data may be overwritten!".format(
                        f["file_name"]
                    )
                )

                # open in append mode
                s = nc.Dataset(f["file_name"], "a")
                h = "[{}] Data added or updated".format(now_str)
                setattr(s, "last_modified", h)

                if "projection" not in s.variables.keys():
                    s = add_proj(s, map_meta=map_meta)

            else:
                self._logger.debug("Creating %s" % f["file_name"])
                s = nc.Dataset(
                    f["file_name"], "w", format="NETCDF4", clobber=True
                )

                # Dimensions
                s.createDimension(self.DIMENSIONS[0], None)
                s.createDimension(self.DIMENSIONS[1], y.shape[0])
                s.createDimension(self.DIMENSIONS[2], x.shape[0])

                # Variables
                time_var = s.createVariable(
                    "time", "f4", (self.DIMENSIONS[0]), **self.COMPRESSION
                )  # type: ignore
                y_var = s.createVariable(
                    "y", "f", self.DIMENSIONS[1], **self.COMPRESSION
                )  # type: ignore
                x_var = s.createVariable(
                    "x", "f", self.DIMENSIONS[2], **self.COMPRESSION
                )  # type: ignore
                variable = s.createVariable(
                    f["variable"],
                    precision,
                    self.DIMENSIONS,
                    least_significant_digit=4,
                    **self.COMPRESSION,
                )  # type: ignore

                # Attributes
                time_var.units = "hours since {}".format(time["start_date"])
                time_var.calendar = "standard"
                time_var.time_zone = time["time_zone"]
                time_var.long_name = "time"

                y_var.units = "meters"
                y_var.description = "UTM, north south"
                y_var.long_name = "y coordinate"

                x_var.units = "meters"
                x_var.description = "UTM, east west"
                x_var.long_name = "x coordinate"

                variable.module = f["module"]
                variable.units = f["info"]["units"]
                variable.long_name = f["info"]["long_name"]

                # Global attribute
                s.setncattr_string("Conventions", "CF-1.6")
                s.setncattr_string("dateCreated", now_str)
                s.setncattr_string(
                    "title",
                    "Distributed {0} data from SMRF".format(
                        f["info"]["long_name"]
                    ),
                )
                s.setncattr_string(
                    "history", "[{}] Create netCDF4 file".format(now_str)
                )
                s = add_proj(s, map_meta=map_meta)

                s.variables['y'][:] = y
                s.variables['x'][:] = x

            s.setncattr_string('SMRF_version', __version__)
            s.close()

    def output(self, variable, data, date_time):
        """
        Output a time step

        Args:
            variable: variable name that will index into variable list
            data: the variable data
            date_time: the date time object for the time step
        """

        self._logger.debug(
            '{0} Writing variable {1} to netCDF'.format(date_time, variable)
        )

        f = nc.Dataset(
            self.variable_dict[variable]['file_name'], 'a', 'NETCDF4'
        )

        # the current time integer
        times = f.variables['time']
        t = nc.date2num(
            date_time.replace(tzinfo=None), times.units, times.calendar
        )

        if len(times) != 0:
            index = np.where(times[:] == t)[0]
            if index.size == 0:
                index = len(times)
            else:
                index = index[0]
        else:
            index = len(times)

        # insert the time and data
        f.variables['time'][index] = t

        if self.outConfig['mask_output']:
            f.variables[variable][index, :] = data*self.mask
        else:
            f.variables[variable][index, :] = data

        f.close()
