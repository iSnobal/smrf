from datetime import datetime
from typing import Tuple

import numpy as np

from .image_data import ImageData
from smrf.envphys import albedo
from smrf.utils import utils


class Albedo(ImageData):
    """
    The visible (280-700nm) and infrared (700-2800nm) albedo follows the
    relationships described in Marks et al. (1992) :cite:`Marks&al:1992`. The
    albedo is a function of the time since last storm, the solar zenith angle,
    and grain size. The time since last storm is tracked on a pixel by pixel
    basis and is based on where there is significant accumulated distributed
    precipitation. This allows for storms to only affect a small part of the
    basin and have the albedo decay at different rates for each pixel.
    """

    VARIABLE = "albedo"

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

    def __init__(self, config):
        """
        Initialize albedo()

        Args:
            config: configuration from [albedo] section
        """
        super().__init__(config)

        # Get the veg values for the decay methods. Date method uses self.veg
        # Hardy2000 uses self.litter
        for d in ["veg", "litter"]:
            v = {}

            matching = [s for s in config.keys() if "{0}_".format(d) in s]
            for m in matching:
                ms = m.split("_")
                v[ms[-1]] = config[m]

            # Create self.litter,self.veg
            setattr(self, d, v)

    def initialize(self, topo, metadata):
        """
        Initialize the distribution, calls ImageData._initialize()

        Args:
            topo: smrf.data.loadTopo.Topo instance contain topo data/info
            metadata: data dataframe containing the station data

        """
        super().initialize(topo, metadata)

        self.burn_mask = topo.burn_mask

        if self.config["decay_method"] is None:
            self._logger.warning("No decay method is set!")

    def distribute(
        self, current_time_step: datetime, cosz: np.ndarray, storm_day: np.ndarray
    ) -> None:
        """
        Distribute air temperature given a Panda's dataframe for a single time
        step. Calls :mod:`smrf.distribute.ImageData._distribute`.

        Args:
            current_time_step: Current time step in datetime object
            cosz: Llumination angle for the current time step
            storm_day: Decimal days since it last snowed at a grid cell
        """

        self._logger.debug("%s Distributing albedo" % current_time_step)

        # only need to calculate albedo if the sun is up
        if cosz is not None:
            alb_v, alb_ir = albedo.albedo(
                storm_day,
                cosz,
                self.config["grain_size"],
                self.config["max_grain"],
                self.config["dirt"],
            )

            # Perform litter decay
            if self.config["decay_method"] == "date_method":
                current_hours, decay_hours = self.decay_window(current_time_step)
                if current_hours > 0:
                    alb_v, alb_ir = albedo.decay_alb_power(
                        self.veg,
                        self.veg_type,
                        current_hours,
                        decay_hours,
                        self.config["date_method_decay_power"],
                        alb_v,
                        alb_ir,
                    )
            elif self.config["decay_method"] == "post_fire":
                current_hours, decay_hours = self.decay_window(current_time_step)
                if current_hours > 0:
                    alb_v, alb_ir = albedo.decay_burned(
                        alb_v,
                        alb_ir,
                        storm_day,
                        self.burn_mask,
                        self.config["post_fire_k_burned"],
                        self.config["post_fire_k_unburned"],
                    )

            elif self.config["decay_method"] == "hardy2000":
                alb_v, alb_ir = albedo.decay_alb_hardy(
                    self.litter, self.veg_type, storm_day, alb_v, alb_ir
                )

            self.albedo_vis = utils.set_min_max(alb_v, self.min, self.max)
            self.albedo_ir = utils.set_min_max(alb_ir, self.min, self.max)

        else:
            self.albedo_vis = np.zeros(storm_day.shape)
            self.albedo_ir = np.zeros(storm_day.shape)

    def decay_window(self, current_timestep: datetime) -> Tuple[float, float]:
        # Calculate hour past start of decay
        current_difference = current_timestep - self.config["start_decay"]
        current_hours = (
            current_difference.days * 24.0 + current_difference.seconds / 3600.0
        )

        # Exit if we are before the window starts
        if current_hours < 0:
            return -1, 0

        # Calculate total time of decay
        decay_difference = self.config["end_decay"] - self.config["start_decay"]
        decay_hours = (
            decay_difference.days * 24.0 + decay_difference.seconds / 3600.0
        )

        return current_hours, decay_hours
