from datetime import datetime
from typing import Tuple

import numpy as np
from smrf.data.load_topo import Topo
from smrf.envphys.constants import (
    GRAVITY,
    IR_MAX,
    IR_MIN,
    MOL_AIR,
    SEA_LEVEL,
    STD_AIRTMP,
    STD_LAPSE,
    VISIBLE_MAX,
    VISIBLE_MIN,
)
from smrf.envphys.solar.irradiance import direct_solar_irradiance
from smrf.envphys.solar.twostream import twostream
from smrf.envphys.thermal.topotherm import hysat
from topocalc.horizon import horizon


def check_wavelengths(wavelength_range):
    if wavelength_range[0] >= VISIBLE_MIN and wavelength_range[1] <= VISIBLE_MAX:
        return
    elif wavelength_range[0] >= IR_MIN and wavelength_range[1] <= IR_MAX:
        return
    else:
        raise ValueError(
            "stoporad wavelength range not within visible or IR wavelengths"
        )


def mask_for_shade(
    cos_z: float,
    azimuth: float,
    illumination_angles: np.ndarray,
    topo: Topo,
    horizon_angles: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mask the illumination angles for shaded areas

    Args:
        cos_z:                   Solar zenith angle (cos)
        azimuth:                 Solar azimuth angles
        illumination_angles:     Calculated angles from topocalc
        topo:                    Loaded topo file
        horizon_angles:          Horizon angles (if previously calculated)

    Returns:
        Tuple:
        * Illumination angles with shaded areas masked out
        * Calculated horizon angle for azimuth
    """
    if horizon_angles is None:
        horizon_angles = horizon(azimuth, topo.dem, topo.dx)

    sun_position = np.tan(np.pi / 2 - np.arccos(cos_z))
    no_sun_mask = np.tan(np.abs(horizon_angles)) > sun_position

    shaded_angles = illumination_angles.copy()
    shaded_angles[no_sun_mask] = 0

    return shaded_angles, horizon_angles


def stoporad(
    date_time: datetime,
    topo: Topo,
    cos_z: float,
    azimuth: float,
    illumination_angles: np.ndarray,
    albedo_surface: np.ndarray,
    wavelength_range: list,
    tau_elevation=100,
    tau=0.2,
    omega=0.85,
    scattering_factor=0.3,
    horizon_angles=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The horizon angles are returned for subsequent calls to this method to re-use and
    save compute time.

    Args:
        date_time:               Current processed time step
        topo:                    Loaded topo file
        cos_z:                   Solar zenith angle (cos)
        azimuth:                 Solar azimuth angles
        illumination_angles:     Calculated angles from topocalc
        albedo_surface:          Calculated albedo from time step
        wavelength_range:        Spectrum to calculate for
        tau_elevation
        tau
        omega
        scattering_factor
        horizon_angles:          Horizon angles (if previously calculated with this method)

    Returns:
        Tuple: Beam radiation, diffuse radiation, and horizon angles
    """
    check_wavelengths(wavelength_range)

    # Check if sun is up
    if cos_z < 0:
        return np.zeros_like(topo.dem), np.zeros_like(topo.dem), np.zeros_like(topo.dem)

    else:
        solar_irradiance = direct_solar_irradiance(date_time, w=wavelength_range)

        # Get beam and diffuse radiation
        evrad = Elevrad(
            topo.dem,
            solar_irradiance,
            cos_z,
            tau_elevation=tau_elevation,
            tau=tau,
            omega=omega,
            scattering_factor=scattering_factor,
            surface_albedo=np.mean(albedo_surface),
        )

        shade, horizon_angles = mask_for_shade(cos_z, azimuth, illumination_angles, topo, horizon_angles)

        # Correct topographically
        trad_beam, trad_diff = toporad(
            evrad.beam,
            evrad.diffuse,
            shade,
            topo.sky_view_factor,
            topo.terrain_config_factor,
            cos_z,
            surface_albedo=albedo_surface,
        )

    return trad_beam, trad_diff, horizon_angles


def toporad(
    beam,
    diffuse,
    illumination_angles,
    sky_view_factor,
    terrain_config_factor,
    cos_z,
    surface_albedo=0.0,
):
    """
    Topographically-corrected solar radiation. Calculates the topographic
    distribution of solar radiation at a single time, using input beam and
    diffuse radiation calculates supplied by elevrad.

    Args:
        beam (np.array): beam radiation
        diffuse (np.array): diffuse radiation
        illumination_angles (np.array): local illumination angles
        sky_view_factor (np.array): sky view factor
        terrain_config_factor (np.array): terrain configuration factor
        cos_z (float): cosine of the zenith
        surface_albedo (float/np.array, optional): surface albedo.
            Defaults to 0.0.

    Returns:
        tuple: beam and diffuse radiation corrected for terrain
    """

    # adjust diffuse radiation accounting for sky view factor
    drad = diffuse * sky_view_factor

    # add reflection from adjacent terrain
    drad = (
        drad
        + (diffuse * (1 - sky_view_factor) + beam * cos_z)
        * terrain_config_factor
        * surface_albedo
    )

    # global radiation is diffuse + incoming_beam * cosine of local
    # illumination * angle
    rad = drad + beam * illumination_angles

    return rad, drad


class Elevrad:
    """
    Beam and diffuse radiation from elevation.
    :py:class:`Elevrad` is essentially the spatial or grid version of the twostream
    command.

    Args:
        elevation (np.array): DEM elevations in meters
        solar_irradiance (float): from direct_solar_irradiance
        cos_z (float): cosine of zenith angle
        tau_elevation (float, optional): Elevation [m] of optical depth
                                        measurement. Defaults to 100.
        tau (float, optional): optical depth at tau_elevation. Defaults to 0.2.
        omega (float, optional): Single scattering albedo. Defaults to 0.85.
        scattering_factor (float, optional): Scattering asymmetry parameter.
                                            Defaults to 0.3.
        surface_albedo (float, optional): Mean surface albedo. Defaults to 0.5.
    """

    def __init__(self, elevation, solar_irradiance, cos_z, **kwargs):
        """
        Args:
            elevation (np.array): DEM elevation in meters
            solar_irradiance (float): from direct_solar_irradiance
            cos_z (float): cosine of zenith angle
            kwargs: tau_elevation, tau, omega, scattering_factor,
                    surface_albedo
        """
        # For calculations
        self.beam = None
        self.diffuse = None

        # Defaults
        self.tau_elevation = 100.0
        self.tau = (0.2,)
        self.omega = 0.85
        self.scattering_factor = 0.3
        self.surface_albedo = 0.5

        # Get user specific values and overwrite defaults
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        self.elevation = elevation
        self.solar_irradiance = solar_irradiance
        self.cos_z = cos_z

        self.calculate()

    def calculate(self):
        """
        Perform the calculations

        Sets:
         * :py:attr:`beam`
         * :py:attr:`diffuse`
        """

        # reference pressure (at reference elevation, in km)
        reference_pressure = hysat(
            SEA_LEVEL,
            STD_AIRTMP,
            STD_LAPSE,
            self.tau_elevation / 1000,
            GRAVITY,
            MOL_AIR,
        )

        # Convert each elevation in look-up table to pressure, then to optical
        # depth over the modeling domain
        pressure = hysat(
            SEA_LEVEL, STD_AIRTMP, STD_LAPSE, self.elevation / 1000, GRAVITY, MOL_AIR
        )
        tau_domain = self.tau * pressure / reference_pressure

        # twostream over the optical depth of the domain
        two_stream = twostream(
            self.cos_z,
            self.solar_irradiance,
            tau=tau_domain,
            omega=self.omega,
            g=self.scattering_factor,
            R0=self.surface_albedo,
        )

        # calculate beam and diffuse
        self.beam = self.solar_irradiance * two_stream["direct_transmittance"]
        self.diffuse = (
            self.solar_irradiance
            * self.cos_z
            * (two_stream["transmittance"] - two_stream["direct_transmittance"])
        )
