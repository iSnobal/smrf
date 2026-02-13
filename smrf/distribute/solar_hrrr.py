from datetime import datetime

import numpy as np
import numexpr as ne

from smrf.distribute.albedo import Albedo
from smrf.envphys.solar import vegetation
from smrf.envphys.solar.net_solar import NetSolar
from smrf.envphys.solar.toporad import mask_for_shade
from smrf.envphys.solar.toposplit import TopoSplit

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

        if self.config.get("correct_veg", False):
            self.correct_vegetation(illumination_angles)

        params = {
            "direct": self.direct.astype(np.float32, copy=False, order="C"),
            "diffuse": self.diffuse.astype(np.float32, copy=False, order="C"),
        }
        self.hrrr_solar = ne.evaluate(
            "direct + diffuse", local_dict=params, casting="safe"
        )

        self.calculate_net_solar(albedo)

    def calculate_net_solar(self, albedo: Albedo) -> None:
        """
        Calculate net solar based on the set instance variable in the Albedo class.

        :param albedo: Instance of :py:class:`smrf.distribute.albedo.Albedo`
        """
        if albedo.albedo_vis is not None and albedo.albedo_ir is not None:
            self._logger.debug("Calculating net solar using vis and ir albedo")
            self.net_solar = NetSolar.broadband_from_vis_ir(self.hrrr_solar, albedo)
        elif albedo.albedo_diffuse is not None and albedo.albedo_direct is not None:
            self._logger.debug("Calculating net solar using diffuse and direct albedo")
            self.net_solar = NetSolar.albedo_diffuse_and_direct(
                direct=self.direct, diffuse=self.diffuse, albedo=albedo
            )
        elif albedo.albedo is not None:
            self._logger.debug("Calculating net solar using broadband albedo")
            self.net_solar = NetSolar.broadband_albedo(self.hrrr_solar, albedo)
        else:
            raise RuntimeError("Albedo variables are not set")

    def correct_vegetation(self, illumination_angles: np.ndarray) -> None:
        """
        Adjust the direct and diffuse components for vegetation.

        Args:
            illumination_angles: Illumination angles from :py:mod:`smrf.envphys.radiation.sunang`
        """
        self.direct = vegetation.solar_veg_beam(
            self.direct, self.veg_height, illumination_angles, self.veg_k
        )
        self.diffuse = vegetation.solar_veg_diffuse(self.diffuse, self.veg_tau)
