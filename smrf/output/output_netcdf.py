import logging
import os
from datetime import datetime
from pathlib import Path

import netCDF4 as nc
import numpy as np
from smrf import __version__
from smrf.data.load_topo import Topo
from spatialnc.proj import add_proj, add_proj_from_file


class OutputNetcdf:
    """
    Class OutputNetcdf() to output values to a netCDF file
    """

    type = "netcdf"
    FILE_EXTENSION = ".nc"
    fmt = "%Y-%m-%d %H:%M:%S"
    COMPRESSION = dict(compression="zlib", complevel=4)
    DIMENSIONS = ("time", "y", "x")

    def __init__(
        self, output_variables: dict, topo: Topo, time: dict, out_config: dict
    ):
        """
        Initialize the OutputNetcdf() class

        Args:
            output_variables: Dict of output variable name (key)
                              and the module (value) creating it
            topo:             Topo instance
            time:             Time configuration
            out_config:       Configuration from [output] section
        """

        self._logger = logging.getLogger(__name__)

        self.output_variables = output_variables
        self.topo = topo
        self.out_config = out_config
        self.out_location = out_config["out_location"]

        self.run_time_step = int(time["time_step"])
        self.out_frequency = int(out_config["frequency"])
        self.create_time = datetime.now().strftime(self.fmt)

        # Retrieve projection information from topo
        self.map_meta = add_proj_from_file(topo.file)

        for nc_variable, module in self.output_variables.items():
            file_name = self.file_name(nc_variable)

            if os.path.isfile(file_name):
                self.append_to_existing_file(file_name)
            else:
                self.create_new_file(file_name, nc_variable, module, time)

    @property
    def out_location(self):
        return self._out_location

    @out_location.setter
    def out_location(self, path: str) -> Path:
        self._out_location = Path(path)

    @property
    def topo_x(self):
        return self.topo.x

    @property
    def topo_y(self):
        return self.topo.y

    def file_name(self, file_name: str) -> str:
        """
        Construct a file name based on the module OUTPUT_VARIABLES information. The
        key is the file name and the primary variable.

        :param file_name: File name to use

        :return: str - file name
        """
        return self.out_location.joinpath(file_name + self.FILE_EXTENSION).as_posix()

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

    def create_new_file(self, file_name: str, nc_variable: str, module, time: dict):
        """
        Create a new netCDF that will be written to by the module

        :param file_name:   Absolute path to the file location
        :param nc_variable: Variable name of the NetCDF
        :param module:      ::module:smrf.distribute: Instance that produced the output data
        :param time:        Current time when file was created
        """
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
                nc_variable,
                self.out_config["netcdf_output_precision"],
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

            variable.module = str(module)
            variable.units = module.OUTPUT_VARIABLES[nc_variable]["units"]
            variable.long_name = module.OUTPUT_VARIABLES[nc_variable]["long_name"]

            # Global attribute
            new_file.setncattr_string("Conventions", "CF-1.6")
            new_file.setncattr_string("dateCreated", self.create_time)
            new_file.setncattr_string(
                "title",
                "Distributed {0} data from SMRF".format(
                    module.OUTPUT_VARIABLES[nc_variable]["long_name"]
                ),
            )
            new_file.setncattr_string(
                "history", "[{}] Create netCDF4 file".format(self.create_time)
            )
            new_file = add_proj(new_file, map_meta=self.map_meta)

            new_file.variables["y"][:] = self.topo_y
            new_file.variables["x"][:] = self.topo_x

            new_file.setncattr_string("SMRF_version", __version__)

    def output(self, date_time):
        """
        Output a time step

        Args:
            date_time: the date time object for the time step to be saved
        """
        for nc_variable, module in self.output_variables.items():
            # Get the data from the distribution class
            data = getattr(module, nc_variable)
            self._logger.debug(
                "{0} Writing variable {1} to from module {2} netCDF".format(
                    date_time, nc_variable, str(module)
                )
            )

            if data is None:
                data = np.zeros((self.topo.ny, self.topo.nx))

            with nc.Dataset(
                self.file_name(nc_variable), mode="a", format="NETCDF4"
            ) as file:
                # the current time integer
                times = file.variables["time"]
                t = nc.date2num(
                    date_time.replace(tzinfo=None), times.units, times.calendar
                )

                existing_times = np.where(times[:] == t)[0]
                if existing_times.size > 0:
                    index = existing_times[0]
                else:
                    index = len(times)

                # insert the time and data
                file.variables["time"][index] = t

                if self.out_config["mask_output"]:
                    data = data * self.topo.mask

                file.variables[nc_variable][index, :] = data
