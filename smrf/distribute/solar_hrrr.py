from datetime import datetime

import numpy as np
from smrf.envphys.solar.toporad import mask_for_shade

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

    OUTPUT_PREFIX = "solar_"
    OUTPUT_VARIABLES = {
        OUTPUT_PREFIX + "ghi": {
            "units": "watt/m2",
            "standard_name": "global_horizontal_irradiance",
            "long_name": "HRRR DSWRF - Interpolated to topographic resolution",
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

        # Class attributes holding output data
        self.ghi_vis = None
        self.dni = None
        self.dhi = None
        self.k = None
        self.solar = None
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
        albedo_vis: np.ndarray,
        albedo_ir: np.ndarray,
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
        albedo_vis:          Visible albedo
        albedo_ir:           Infrared albedo

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
            self.net_solar = empty

            return

        illumination_angles, _horizon_angles = mask_for_shade(
            cos_z, azimuth, illumination_angles, self.topo
        )

        # Early morning/late evening hours will have a zero or negative values
        # from interpolation in some pixels. Masking all values less than 1 W/m^2.
        ghi_mask = hrrr_data[self.DSWRF] <= 1
        dswrf = np.where(ghi_mask, 0, hrrr_data[self.DSWRF])
        direct_mask = np.logical_or(ghi_mask, hrrr_data[self.VBDSF] <= 1)
        direct_normal = np.where(direct_mask, 1, hrrr_data[self.VBDSF])
        diffuse_mask = np.logical_or(ghi_mask, hrrr_data[self.VDDSF] <= 1)
        diffuse_horizontal = np.where(diffuse_mask, 0, hrrr_data[self.VDDSF])

        # Calculate GHI; Equation (1)
        self.solar_ghi_vis = direct_normal * cos_z + diffuse_horizontal

        # Obtain K based on visible diffuse fraction; Equation (2)
        self.solar_k = diffuse_horizontal / self.solar_ghi_vis

        # Global diffuse fraction derived from global incoming (DSWRF); Equation (3)
        # DHI - Diffuse Horizontal Irradiance
        self.solar_dhi = dswrf * self.solar_k

        # Global direct; Equation (4)
        # DNI - Direct Normal Irradiance
        self.solar_dni = (dswrf * (1 - self.solar_k)) / cos_z

        # Toposplit model
        self.hrrr_solar = (
            self.solar_dni * illumination_angles +
            self.solar_dhi * self.sky_view_factor
        )

        # Net solar
        # Simulations with SBDART showed the ratios used in the below calculation
        # to get to broadband albedo using a split at 700nm for VIS vs IR. These values
        # also assumed clear-sky conditions and a lat/lon over the Western US.
        self.net_solar = self.hrrr_solar * ( 1 -(0.54 * albedo_vis + 0.46 * albedo_ir))
