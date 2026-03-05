from datetime import datetime

import numpy as np
import numpy.typing as npt

from smrf.envphys import albedo
from smrf.utils import utils

from .variable_base import VariableBase


class Albedo(VariableBase):
    """
    The visible (280-700nm) and infrared (700-2800nm) albedo follows the
    relationships described in Marks et al. (1992) :cite:`Marks&al:1992`. The
    albedo is a function of the time since last storm, the solar zenith angle,
    and grain size. The time since last storm is tracked on a pixel by pixel
    basis and is based on where there is significant accumulated distributed
    precipitation. This allows for storms to only affect a small part of the
    basin and have the albedo decay at different rates for each pixel.
    """

    DISTRIBUTION_KEY = "albedo"

    # these are variables that can be output
    OUTPUT_VARIABLES = {
        "albedo_vis": {
            "units": "None",
            "standard_name": "visible_albedo",
            "long_name": "Visible wavelength albedo",
        },
        "albedo_ir": {
            "units": "None",
            "standard_name": "infrared_albedo",
            "long_name": "Infrared wavelength albedo",
        },
    }

    # List of recognized variables in external NetCDF files
    # Reused from existing runs
    ALBEDO_VIS = "albedo_vis"
    ALBEDO_IR = "albedo_ir"
    # Supplied through external files
    ALBEDO_DIRECT = "albedo_direct"
    ALBEDO_DIFFUSE = "albedo_diffuse"

    MAX_ALBEDO = np.float32(1.0)

    EXTERNAL_SOURCE_VARIABLES = frozenset(
        [DISTRIBUTION_KEY, ALBEDO_VIS, ALBEDO_IR, ALBEDO_DIRECT, ALBEDO_DIFFUSE]
    )

    def __init__(self, config: dict, topo):
        """
        Initialize albedo()

        Args:
            config: configuration from [albedo] section
        """
        super().__init__(config, topo)

        # Variables for calculations of external files
        # Note that the 'albedo' attribute is set by the VariableBase class
        self.albedo_vis = None
        self.albedo_ir = None
        self.albedo_direct = None
        self.albedo_diffuse = None

        # Get the veg values for the decay methods. Date method uses self.veg
        # Hardy2000 uses self.litter
        for d in ["veg", "litter"]:
            v = {}

            matching = [s for s in self.config.keys() if "{0}_".format(d) in s]
            for m in matching:
                ms = m.split("_")
                v[ms[-1]] = self.config[m]

            # Create self.litter,self.veg
            setattr(self, d, v)

    def initialize(self, metadata):
        """
        Initialize the distribution, calls ImageData._initialize()

        Args:
            metadata: data dataframe containing the station data

        """
        super().initialize(metadata)

        if (
            self.config.get("decay_method", None) is None
            and self.config["source_files"] is None
        ):
            self._logger.warning("No decay method is set!")

    def distribute(
        self, current_time_step: datetime, cosz: npt.NDArray, storm_day: npt.NDArray
    ):
        """
        Calculate or load snow albedo for given time step. When calculated, the configured
        options from the albedo section in the .ini file is used. When loaded, values
        from the NetCDF files are read. File loading has precedence over the
        calculations.

        Instance variables are set as part of this step depending on the chosen method
        to generate albedo values. These will be later accessible to steps like the
        net solar calculation.
        List of possible instance variables:
         * :py:attr:`albedo`
         * :py:attr:`albedo_ir`
         * :py:attr:`albedo_vis`
         * :py:attr:`albedo_direct`
         * :py:attr:`albedo_diffuse`

        Args:
            current_time_step: Current time step in datetime object
            cosz: numpy array of the illumination angle for the current time
                step
            storm_day: numpy array of the decimal days since it last
                snowed at a grid cell

        """
        self._logger.debug("%s Distributing albedo" % current_time_step)

        # Only calculated when the sun is up
        if cosz is not None:
            if self.source_files is not None:
                # For files with VIS and IR values (typically a previous run)
                if (
                    self.ALBEDO_VIS in self.source_files.variables
                    and self.ALBEDO_IR in self.source_files.variables
                ):
                    self.albedo_vis = self.source_files.load(
                        self.ALBEDO_VIS, current_time_step
                    )
                    self.albedo_ir = self.source_files.load(
                        self.ALBEDO_IR, current_time_step
                    )
                # For files with direct and diffuse values (e.g. Hyperspectral)
                elif (
                    self.ALBEDO_DIRECT in self.source_files.variables
                    and self.ALBEDO_DIFFUSE in self.source_files.variables
                ):
                    self.albedo_direct = self.source_files.load(
                        self.ALBEDO_DIRECT, current_time_step
                    )
                    self.albedo_diffuse = self.source_files.load(
                        self.ALBEDO_DIFFUSE, current_time_step
                    )
                # File only contains broadband albedo (e.g. MODIS, SPIRES)
                elif self.DISTRIBUTION_KEY in self.source_files.variables:
                    self.albedo = self.source_files.load(
                        self.DISTRIBUTION_KEY, current_time_step
                    )
                else:
                    raise ValueError(
                        "Albedo files do not contain recognized albedo variable names"
                        f"  Only found: {self.source_files.variables}"
                    )
            else:
                alb_v, alb_ir = albedo.albedo(
                    storm_day,
                    cosz,
                    self.config["grain_size"],
                    self.config["max_grain"],
                    self.config["dirt"],
                )

                # Perform litter decay
                if self.config["decay_method"] == "date_method":
                    alb_v_d, alb_ir_d = albedo.decay_alb_power(
                        self.veg,
                        self.veg_type,
                        self.config["date_method_start_decay"],
                        self.config["date_method_end_decay"],
                        current_time_step,
                        self.config["date_method_decay_power"],
                        alb_v,
                        alb_ir,
                    )

                    alb_v = alb_v_d
                    alb_ir = alb_ir_d

                elif self.config["decay_method"] == "hardy2000":
                    alb_v_d, alb_ir_d = albedo.decay_alb_hardy(
                        self.litter, self.veg_type, storm_day, alb_v, alb_ir
                    )

                    alb_v = alb_v_d
                    alb_ir = alb_ir_d

                self.albedo_vis = utils.set_min_max(alb_v, self.min, self.max)
                self.albedo_ir = utils.set_min_max(alb_ir, self.min, self.max)
