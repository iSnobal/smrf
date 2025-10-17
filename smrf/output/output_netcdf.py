import logging
import os
from datetime import datetime

import netCDF4 as nc
import numpy as np
from spatialnc.proj import add_proj, add_proj_from_file

from smrf import __version__
from smrf.data.load_topo import Topo

class OutputNetcdf:
    """
    Class OutputNetcdf() to output values to a netCDF file
    """

    type = "netcdf"
    FILE_EXTENSION = ".nc"
    fmt = "%Y-%m-%d %H:%M:%S"
    COMPRESSION = dict(compression="zlib", complevel=4)
    DIMENSIONS = ("time", "y", "x")

    def __init__(self, variable_info: dict, topo: Topo, time: dict, out_config: dict):
        """
        Initialize the OutputNetcdf() class

        Args:
            variable_info: dict of variable information
            topo: loadTopo instance
            time: time configuration
            out_config: out configuration
        """

        self._logger = logging.getLogger(__name__)

        self.variables_info = variable_info
        self.topo = topo
        self.out_config = out_config

        self.run_time_step = int(time["time_step"])
        self.out_frequency = int(out_config["frequency"])
        self.create_time = datetime.now().strftime(self.fmt)

        # Retrieve projection information from topo
        self.map_meta = add_proj_from_file(topo.file)

        for output_variable, variable_info in self.variables_info.items():
            file_name = variable_info["out_location"] + self.FILE_EXTENSION
            # Write the name back to the property for writing the actual with the `output` method
            variable_info["out_location"] = file_name

            if os.path.isfile(file_name):
                self.append_to_existing_file(file_name)
            else:
                self.create_new_file(file_name, variable_info, time)

    @property
    def topo_x(self):
        return self.topo.x

    @property
    def topo_y(self):
        return self.topo.y

    def append_to_existing_file(self, file_name):
        self._logger.warning("Opening {}, data may be overwritten!".format(file_name))

        # open in append mode
        with nc.Dataset(file_name, "a") as file:
            setattr(
                file,
                "last_modified",
                "[{}] Data added or updated".format(self.create_time),
            )

            if "projection" not in file.variables.keys():
                file = add_proj(file, map_meta=self.map_meta)

            file.setncattr_string("SMRF_version", __version__)

    def create_new_file(self, file_name, variable_info: dict, time: dict):
        self._logger.debug("Creating %s" % file_name)

        with nc.Dataset(file_name, "w", format="NETCDF4", clobber=True) as new_file:
            # Dimensions
            new_file.createDimension(self.DIMENSIONS[0], None)
            new_file.createDimension(self.DIMENSIONS[1], self.topo_y.shape[0])
            new_file.createDimension(self.DIMENSIONS[2], self.topo_x.shape[0])

            # Variables
            time_var = new_file.createVariable(
                "time", "f4", (self.DIMENSIONS[0]), **self.COMPRESSION
            )  # type: ignore
            y_var = new_file.createVariable(
                "y", "f", self.DIMENSIONS[1], **self.COMPRESSION
            )  # type: ignore
            x_var = new_file.createVariable(
                "x", "f", self.DIMENSIONS[2], **self.COMPRESSION
            )  # type: ignore
            variable = new_file.createVariable(
                variable_info["variable"],
                self.out_config["netcdf_output_precision"][0],
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

            variable.module = str(variable_info["module"])
            variable.units = variable_info["nc_attributes"]["units"]
            variable.long_name = variable_info["nc_attributes"]["long_name"]

            # Global attribute
            new_file.setncattr_string("Conventions", "CF-1.6")
            new_file.setncattr_string("dateCreated", self.create_time)
            new_file.setncattr_string(
                "title",
                "Distributed {0} data from SMRF".format(
                    variable_info["nc_attributes"]["long_name"]
                ),
            )
            new_file.setncattr_string(
                "history", "[{}] Create netCDF4 file".format(self.create_time)
            )
            new_file = add_proj(new_file, map_meta=self.map_meta)

            new_file.variables['y'][:] = self.topo_y
            new_file.variables['x'][:] = self.topo_x

            new_file.setncattr_string("SMRF_version", __version__)

    def output(self, variable, data, date_time):
        """
        Output a time step

        Args:
            variable: variable name that will index into variable list
            data: the variable data
            date_time: the date time object for the time step
        """

        self._logger.debug(
            "{0} Writing variable {1} to netCDF".format(date_time, variable)
        )

        with nc.Dataset(
            self.variables_info[variable]["out_location"], mode="a", format="NETCDF4"
        ) as file:
            # the current time integer
            times = file.variables["time"]
            t = nc.date2num(date_time.replace(tzinfo=None), times.units, times.calendar)

            existing_times = np.where(times[:] == t)[0]
            if existing_times.size > 0:
                index = existing_times[0]
            else:
                index = len(times)

            # insert the time and data
            file.variables["time"][index] = t

            if self.out_config["mask_output"]:
                data = data * self.topo.mask

            file.variables[variable][index, :] = data
