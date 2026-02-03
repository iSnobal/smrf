import numpy as np

from .variable_base import VariableBase
from smrf.envphys import albedo
from smrf.utils import utils


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
        "albedo": {
            "units": "None",
            "standard_name": "albedo_total",
            "long_name": "Blue Sky Broadband Albedo",
        },
        "albedo_direct": {
            "units": "None",
            "standard_name": "albedo_direct",
            "long_name": "Black Sky Broadband Albedo",
        },
        "albedo_diffuse": {
            "units": "None",
            "standard_name": "albedo_diffuse",
            "long_name": "White Sky Broadband Albedo",
        },
    }

    def __init__(self, config: dict, topo):
        """
        Initialize albedo()

        Args:
            config: configuration from [albedo] section
        """
        super().__init__(config, topo)

        # Get the veg values for the decay methods. Date method uses self.veg
        # Hardy2000 uses self.litter
        for d in ['veg', 'litter']:
            v = {}

            matching = [s for s in self.config.keys()
                        if "{0}_".format(d) in s]
            for m in matching:
                ms = m.split('_')
                v[ms[-1]] = self.config[m]

            # Create self.litter,self.veg
            setattr(self, d, v)

    def initialize(self, metadata):
        """
        Initialize the distribution, calls ImageData._initialize()

        Args:
            topo: smrf.data.loadTopo.Topo instance contain topo data/info
            metadata: data dataframe containing the station data

        """
        super().initialize(metadata)

        if self.config["decay_method"] is None:
            self._logger.warning("No decay method is set!")

    def distribute(self, current_time_step, cosz, storm_day, data=None):
        """
        Distribute air temperature given a Panda's dataframe for a single time
        step. Calls :mod:`smrf.distribute.ImageData._distribute`.

        Args:
            current_time_step: Current time step in datetime object
            cosz: numpy array of the illumination angle for the current time
                step
            storm_day: numpy array of the decimal days since it last
                snowed at a grid cell

        """

        self._logger.debug('%s Distributing albedo' % current_time_step)

        # set defaults for night time
        self.albedo_vis = np.zeros(storm_day.shape)
        self.albedo_ir = np.zeros(storm_day.shape)
        self.albedo = np.zeros(storm_day.shape)
        self.albedo_direct = np.zeros(storm_day.shape)
        self.albedo_diffuse = np.zeros(storm_day.shape)

        # only need to calculate albedo if the sun is up
        if cosz is not None:
            
            # Grab external albedo data if present
            if data is not None:
                # Check for direct and diffuse data (this is best and should be used if both are given)
                if hasattr(data, 'albedo_direct') and hasattr(data, 'albedo_diffuse'):
                    self._logger.debug("Using external direct/diffuse albedo")
                    self.albedo_direct = data.albedo_direct.loc[current_time_step].values
                    self.albedo_diffuse = data.albedo_diffuse.loc[current_time_step].values
                    self.albedo = None

                # Fallback to broadband albedo (this is total blue sky albedo).
                elif hasattr(data, 'albedo'):
                    self._logger.debug("Using external broadband albedo")
                    self.albedo = data.albedo.loc[current_time_step].values
                    self.albedo_direct = None
                    self.albedo_diffuse = None

            # Always populate decay based albedos
            alb_v, alb_ir = albedo.albedo(
                storm_day, cosz,
                self.config['grain_size'],
                self.config['max_grain'],
                self.config['dirt'])

            # Perform litter decay
            if self.config['decay_method'] == 'date_method':
                alb_v_d, alb_ir_d = albedo.decay_alb_power(
                    self.veg,
                    self.veg_type,
                    self.config['date_method_start_decay'],
                    self.config['date_method_end_decay'],
                    current_time_step,
                    self.config['date_method_decay_power'],
                    alb_v, alb_ir)

                alb_v = alb_v_d
                alb_ir = alb_ir_d

            elif self.config['decay_method'] == 'hardy2000':
                alb_v_d, alb_ir_d = albedo.decay_alb_hardy(
                    self.litter,
                    self.veg_type,
                    storm_day,
                    alb_v,
                    alb_ir)

                alb_v = alb_v_d
                alb_ir = alb_ir_d

            self.albedo_vis = utils.set_min_max(alb_v, self.min, self.max)
            self.albedo_ir = utils.set_min_max(alb_ir, self.min, self.max)