from datetime import datetime
from typing import Tuple

import numexpr as ne
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
        super().__init__(config, topo)

        # Variables for calculations of external files
        # Note that the 'albedo' attribute is set by the VariableBase class
        self.albedo_vis = None
        self.albedo_ir = None
        self.albedo_direct = None
        self.albedo_diffuse = None
        self.burn_mask = None

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

        self.load_burn_mask()

        if (
            self.config.get("decay_method", None) is None
            and self.config["source_files"] is None
        ):
            self._logger.warning("No decay method is set!")

    def load_burn_mask(self):
        """
        Load a post fire burn mask from the topo file. If none is set all values
        will be marked as 0.

        Sets: :py:attr:`burn_mask`
        """
        burn_mask = getattr(self.topo, "burn_mask", None)
        if burn_mask is None:
            self.burn_mask = np.zeros_like(self.topo.dem)
        else:
            self.burn_mask = self.topo.burn_mask

    def distribute(
        self, current_time_step: datetime, cos_z: npt.NDArray, storm_day: npt.NDArray
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
            cos_z: Illumination angle for the current time step
            storm_day: Decimal days since it last snowed at a grid cell
        """
        self._logger.debug("%s Distributing albedo" % current_time_step)

        # Only calculated when the sun is up
        if cos_z is not None:
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
                    cos_z,
                    self.config["grain_size"],
                    self.config["max_grain"],
                    self.config["dirt"],
                )

                if self.config["decay_method"] == "date_method":
                    self._logger.debug("  Using date method decay")
                    current_hours, decay_hours = self.decay_window(current_time_step)
                    if current_hours >= 0:
                        alb_v, alb_ir = self.date_method(
                            alb_v, alb_ir, current_hours, decay_hours, storm_day
                        )

                elif self.config["decay_method"] == "hardy2000":
                    self._logger.debug("  Using hardy2000 decay")
                    alb_v, alb_ir = albedo.decay_alb_hardy(
                        self.litter, self.veg_type, storm_day, alb_v, alb_ir
                    )

                self.albedo_vis = utils.set_min_max(alb_v, self.min, self.max)
                self.albedo_ir = utils.set_min_max(alb_ir, self.min, self.max)

        else:
            self.albedo_vis = np.zeros(storm_day.shape)
            self.albedo_ir = np.zeros(storm_day.shape)

    def date_method(
        self,
        alb_v: npt.NDArray,
        alb_ir: npt.NDArray,
        current_hours: float,
        decay_hours: float,
        storm_day: npt.NDArray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply a power law decay to the albedo and an optional post fire decay if configured.
        The post fire decay uses the initial albedo value of the time decay method after
        a fresh snowfall.

        :param alb_v: Visibility albedo
        :param alb_ir: Infrared albedo
        :param current_hours: Time in hours since start of decay window
        :param decay_hours: Total hours of the decay window
        :param storm_day: Time (days) since last snowfall

        :return:
            alb_v, alb_ir : Decayed albedo for visible and infrared spectrum
        """
        # Create a mask of burned pixels with no new snowfall
        if self.config.get("post_fire", False):
            burned_no_snowfall = ne.evaluate(
                "(burn_mask == 1) & (storm_day > 0)",
                local_dict={"burn_mask": self.burn_mask, "storm_day": storm_day},
            )
        else:
            burned_no_snowfall = np.zeros_like(storm_day)

        alb_v, alb_ir = albedo.decay_alb_power(
            self.veg,
            self.veg_type,
            current_hours,
            decay_hours,
            self.config["date_method_decay_power"],
            alb_v,
            alb_ir,
            burned_no_snowfall,
        )

        if self.config.get("post_fire", False):
            self._logger.debug("  Applying post fire decay")
            alb_v, alb_ir = albedo.decay_burned(
                alb_v,
                alb_ir,
                storm_day,
                self.burn_mask,
                self.config.get("post_fire_k_burned", None),
            )

        return alb_v, alb_ir

    def decay_window(self, current_timestep: datetime) -> Tuple[float, float]:
        """
        Determine if the current time is within the decay window.
        When the window has not been reached, a return value of -1 is returned.
        When the window has been reached, the current and total decay hours are returned.

        :param current_timestep: Timestep to check if it is within the decay window

        :return: Tuple(current_hours, decay_hours)
            current_hours: Time in hours since start of decay window or -1 if outside the window
            decay_hours: Total hours of the decay window
        """
        # Calculate current hours past the start of the decay window
        current_difference = current_timestep - self.config["decay_start"]
        current_hours = (
            current_difference.days * 24.0 + current_difference.seconds / 3600.0
        )

        # Exit if we are before the window starts
        if current_hours < 0:
            return -1, 0

        # Calculate total time of decay window
        decay_difference = self.config["decay_end"] - self.config["decay_start"]
        decay_hours = decay_difference.days * 24.0 + decay_difference.seconds / 3600.0

        return current_hours, decay_hours
