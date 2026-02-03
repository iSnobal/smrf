from datetime import datetime

import numpy as np
import numexpr as ne
from smrf.envphys.solar.toporad import mask_for_shade
from smrf.envphys.solar.toposplit import TopoSplit
from smrf.distribute.albedo import Albedo

from .variable_base import VariableBase


class SolarHRRR(VariableBase):
    """
    Calculate solar radiation based of the following HRRR variables:
        * DSWRF - Downwelling Shortwave Radiation Flux
        * VBDSF - Visible Beam Downward Solar Flux
        * VDDSF - Visible Diffuse Downward Solar Flux

    .. math::
        Toposplit = DNI * cos(\\theta) * S + DHI * V_f

        DNI = Direct Normal Irradiance
        \\theta = solar zenith angle
        S = Topographic Shading
        DHI = Diffuse Horizontal Irradiance
        V_f = Sky View Factor
    """

    DISTRIBUTION_KEY = "solar"
    DEFAULT_OUTPUT = "net_solar"
    INI_VARIABLE = "hrrr_solar"
    # HRRR Variable names
    ## Downwelling Shortwave Radiation Flux
    DSWRF = "DSWRF"
    ## Visible Beam Downward Solar Flux
    VBDSF = "VBDSF"
    ## Visible Diffuse Downward Solar Flux
    VDDSF = "VDDSF"

    GRIB_VARIABLES = [DSWRF, VBDSF, VDDSF]

    # Minimum value to calculate radiation for
    MIN_RADIATION = 1

    OUTPUT_PREFIX = "solar_"
    OUTPUT_VARIABLES = {
        OUTPUT_PREFIX + "ghi_vis": {
            "units": "watt/m2",
            "standard_name": "global_horizontal_irradiance",
            "long_name": "GHI from beam and diffuse in the visible wavelengths",
            "source": "HRRR Grib",
        },
        OUTPUT_PREFIX + "dhi": {
            "units": "watt/m2",
            "standard_name": "diffuse_horizontal_irradiance",
            "long_name": "",
        },
        OUTPUT_PREFIX + "dni": {
            "units": "watt/m2",
            "standard_name": "direct_normal_irradiance",
            "long_name": "",
        },
        INI_VARIABLE: {
            "units": "watt/m2",
            "standard_name": "surface_downwelling_shortwave_flux_in_air",
            "long_name": "HRRR DSWRF - Topographically downscaled",
        },
        OUTPUT_PREFIX + "k": {
            "units": "None",
            "standard_name": "diffuse_fraction",
            "long_name": "Fraction of diffuse/global in the visible wavelengths",
        },
        DEFAULT_OUTPUT: {
            "units": "watt/m2",
            "standard_name": "net_solar_radiation",
            "long_name": "Net solar radiation",
        },
    }

    def __init__(self, config, topo):
        super().__init__(config, topo)

        self.toposplit = TopoSplit(
            self.sky_view_factor, self.MIN_RADIATION, self.threads
        )

        # Class attributes holding output data
        self.solar_ghi_vis = None
        self.solar_dni = None
        self.solar_dhi = None
        self.solar_k = None
        self.direct = None
        self.diffuse = None
        self.hrrr_solar = None
        self.net_solar = None

    @classmethod
    def is_requested(cls, config_variables: set) -> bool:
        requested = super().is_requested(config_variables)
        return requested or cls.INI_VARIABLE in config_variables

    def distribute(
        self,
        timestep: datetime,
        hrrr_data: dict,
        cos_z: float,
        azimuth: float,
        illumination_angles: np.ndarray,
        albedo: Albedo,
    ) -> None:
        """
        Calculate solar radiation based on the Toposplit model presented in
        Olson et al. (202X). This model downscales the incoming HRRR radiation based
        on topographic influences.
        Steps involved are:
            * Calculate global hemispherical irradiance (GHI) using a flat surface
            * Calculate the diffuse fraction (k)
            * Determine the diffuse horizontal irradiance (DHI)
            * Correct the visible direct irradiance by zenith angle and topographic shading
            * Correct the diffuse horizontal irradiance by the sky-view factor
            * Calculate the new total incoming solar radiation

        Final step is to calculate the net solar radiation by using an empirical
        relationship for the visible and infrared albedo that was calculated for regions
        across the Western US.

        Changes the following attributes:
         * :py:attr: solar_ghi_vis
         * :py:attr: solar_dni
         * :py:attr: solar_dhi
         * :py:attr: solar_k
         * :py:attr: hrrr_solar
         * :py:attr: net_solar

        Parameters
        ----------
        timestep:            Current processed time step
        hrrr_data:           Dictionary of loaded and interpolated HRRR data
        cos_z:               Solar zenith angle (cos)
        azimuth:             Solar azimuth angles
        illumination_angles: Angles calculated with topocalc.illumination_angle
        albedo:              Instance of :py:class:`smrf.distribute.albedo.Albedo`

        """
        self._logger.debug("%s Distributing HRRR solar" % timestep)

        # Skip calculations if the sun is down
        if cos_z <= 0:
            empty = np.zeros_like(self.sky_view_factor)
            self.solar_ghi_vis = empty
            self.solar_dni = empty
            self.solar_dhi = empty
            self.solar_k = empty
            self.hrrr_solar = empty
            self.direct = empty
            self.diffuse = empty
            self.net_solar = empty

            return

        illumination_angles, _horizon_angles = mask_for_shade(
            cos_z, azimuth, illumination_angles, self.topo
        )

        results = self.toposplit.calculate(
            hrrr_data[self.DSWRF],
            hrrr_data[self.VBDSF],
            hrrr_data[self.VDDSF],
            np.float64(cos_z),
            illumination_angles,
        )

        self.solar_ghi_vis = results["ghi_vis"]
        self.solar_k = results["k"]
        self.solar_dhi = results["dhi"]
        self.solar_dni = results["dni"]
        self.direct = results["direct"]
        self.diffuse = results["diffuse"]

        self.calculate_net_solar(albedo)

    def calculate_net_solar(self, albedo: Albedo) -> None:
        """
        Calculate net solar based on a visible and infrared albedo ratio calculated
        for a Northern Latitude location in the Western US.

        :param albedo: Instance of :py:class:`smrf.distribute.albedo.Albedo`
        """ 
        
        # Variables for numexpr 
        params = {
            'VIS_RATIO': np.float32(0.54),
            'IR_RATIO': np.float32(0.46),
            'MAX_ALBEDO': np.float32(1.0),
            'direct': self.direct.astype(np.float32, copy=False, order='C'),
            'diffuse': self.diffuse.astype(np.float32, copy=False, order='C'),
            'albedo_vis': albedo.albedo_vis.astype(np.float32, copy=False, order='C'),
            'albedo_ir': albedo.albedo_ir.astype(np.float32, copy=False, order='C'),
        }

        params['hrrr_solar'] = ne.evaluate(
            "direct + diffuse", local_dict=params, casting='safe'
        )
        self.hrrr_solar = params['hrrr_solar']

        # 1. If direct and diffuse external albedos, use these.
        if albedo.albedo_direct is not None and albedo.albedo_diffuse is not None:
            params['alb_dir'] = albedo.albedo_direct.astype(np.float32, copy=False)
            params['alb_diff'] = albedo.albedo_diffuse.astype(np.float32, copy=False)

            self.net_solar = ne.evaluate(
                "direct * alb_dir + diffuse * alb_diff",
                local_dict=params, casting='safe'
            )   
        # 2. Else if, we have total broadband albedo (blue sky albedo), use this.  
        elif albedo.albedo is not None:
            params['alb_total'] = albedo.albedo.astype(np.float32, copy=False)
            
            self.net_solar = ne.evaluate(
                "hrrr_solar * alb_total",
                local_dict=params, casting='safe'
            )     

        # 3. Else, fall back to the base model that always is running.
        else:
            self.net_solar = ne.evaluate(
                "hrrr_solar * (MAX_ALBEDO - (VIS_RATIO * albedo_vis + IR_RATIO * albedo_ir))",
                local_dict=params, casting='safe'
            )
