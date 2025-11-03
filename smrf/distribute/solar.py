from datetime import datetime
from typing import Tuple

import numpy as np

from smrf.envphys.constants import IR_WAVELENGTHS, VISIBLE_WAVELENGTHS
from smrf.envphys.solar import cloud, toporad, vegetation
from smrf.utils import utils

from .variable_base import VariableBase


class Solar(VariableBase):
    """
    Multiple steps are required to estimate solar radiation:

    1. Terrain corrected clear sky radiation
    2. Adjust solar radiation for vegetation effects
    3. Calculate net radiation using the albedo

    The Image Processing Workbench (IPW) includes a utility ``stoporad`` to
    model terrain corrected clear sky radiation over the DEM. Within
    ``stoporad``, the radiation transfer model ``twostream`` simulates the
    clear sky radiation on a flat surface for a range of wavelengths through
    the atmosphere :cite:`Dozier:1980` :cite:`Dozier&Frew:1981`
    :cite:`Dubayah:1994`. Terrain correction using the DEM adjusts for terrain
    shading and splits the clear sky radiation into beam and diffuse radiation.

    The second step requires sites measuring solar radiation. The measured
    solar radiation is compared to the modeled clear sky radiation from
    ``twostream``. The cloud factor is then the measured incoming solar
    radiation divided by the modeled radiation.  The cloud factor can be
    computed on an hourly timescale if the measurement locations are of high
    quality. For stations that are less reliable, we recommend calculating a
    daily cloud factor which divides the daily integrated measured radiation by
    the daily integrated modeled radiation.  This helps to reduce the problems
    that may be encountered from instrument shading, instrument calibration, or
    a time shift in the data. The calculated cloud factor at each station can
    then be distributed using any of the method available in
    :mod:`smrf.spatial`. Since the cloud factor is not explicitly controlled
    by elevation like other variables, the values may be distributed without
    detrending to elevation. The modeled clear sky radiation (both beam and
    diffuse)  are adjusted for clouds using
    :mod:`smrf.envphys.radiation.cf_cloud`.

    The third step adjusts the cloud corrected solar radiation for vegetation
    affects, following the methods developed by Link and Marks (1999)
    :cite:`Link&Marks:1999`. The direct beam radiation  is corrected by:

    .. math::
        R_b = S_b * exp( -\\mu h / cos \\theta )

    where :math:`S_b` is the above canopy direct radiation, :math:`\\mu` is
    the extinction coefficient (:math:`m^{-1}`), :math:`h` is the canopy height
    (:math:`m`), :math:`\\theta` is the solar zenith angle, and :math:`R_b` is
    the canopy adjusted direct radiation. Adjusting the diffuse radiation is
    performed by:

    .. math::
        R_d = \\tau * R_d

    where :math:`R_d` is the diffuse adjusted radiation, :math:`\\tau` is the
    optical transmissivity of the canopy, and :math:`R_d` is the above canopy
    diffuse radiation. Values for :math:`\\mu` and :math:`\\tau` can be found
    in Link and Marks (1999) :cite:`Link&Marks:1999`, measured at study sites
    in Saskatchewan and Manitoba.

    The final step for calculating the net solar radiation requires the surface
    albedo from :mod:`smrf.distribute.albedo`. The net radiation is the sum of
    the of beam and diffuse canopy adjusted radiation multiple by one minus
    the albedo.
    """

    DISTRIBUTION_KEY = "solar"

    # these are variables that can be output
    OUTPUT_VARIABLES = {
        "clear_ir_beam": {
            "units": "watt/m2",
            "standard_name": "clear_sky_infrared_beam",
            "long_name": "Clear sky infrared beam solar radiation",
        },
        "clear_ir_diffuse": {
            "units": "watt/m2",
            "standard_name": "clear_sky_infrared_diffuse",
            "long_name": "Clear sky infrared diffuse solar radiation",
        },
        "clear_vis_beam": {
            "units": "watt/m2",
            "standard_name": "clear_sky_visible_beam",
            "long_name": "Clear sky visible beam solar radiation",
        },
        "clear_vis_diffuse": {
            "units": "watt/m2",
            "standard_name": "clear_sky_visible_diffuse",
            "long_name": "Clear sky visible diffuse solar radiation",
        },
        "cloud_ir_beam": {
            "units": "watt/m2",
            "standard_name": "cloud_infrared_beam",
            "long_name": "Cloud corrected infrared beam solar radiation",
        },
        "cloud_ir_diffuse": {
            "units": "watt/m2",
            "standard_name": "cloud_infrared_diffuse",
            "long_name": "Cloud corrected infrared diffuse solar radiation",
        },
        "cloud_vis_beam": {
            "units": "watt/m2",
            "standard_name": "cloud_visible_beam",
            "long_name": "Cloud corrected visible beam solar radiation",
        },
        "cloud_vis_diffuse": {
            "units": "watt/m2",
            "standard_name": "cloud_visible_diffuse",
            "long_name": "Cloud corrected visible diffuse solar radiation",
        },
        "net_solar": {
            "units": "watt/m2",
            "standard_name": "net_solar_radiation",
            "long_name": "Net solar radiation",
        },
        "veg_ir_beam": {
            "units": "watt/m2",
            "standard_name": "vegetation_infrared_beam",
            "long_name": "Vegetation corrected infrared beam solar radiation",
        },
        "veg_ir_diffuse": {
            "units": "watt/m2",
            "standard_name": "vegetation_infrared_diffuse",
            "long_name": "Vegetation corrected infrared diffuse solar \
                radiation",
        },
        "veg_vis_beam": {
            "units": "watt/m2",
            "standard_name": "vegetation_visible_beam",
            "long_name": "Vegetation corrected visible beam solar radiation",
        },
        "veg_vis_diffuse": {
            "units": "watt/m2",
            "standard_name": "vegetation_visible_diffuse",
            "long_name": "Vegetation corrected visible diffuse solar radiation",
        },
    }

    def __init__(self, config, topo):
        super().__init__(config, topo)

        # For calculations
        self.vis_beam = None
        self.vis_diffuse = None
        self.ir_beam = None
        self.ir_diffuse = None
        self.cloud_factor = None

        # Clear sky
        self.clear_vis_beam = None
        self.clear_vis_diffuse = None
        self.clear_ir_beam = None
        self.clear_ir_diffuse = None

        # Cloud corrected
        self.cloud_vis_beam = None
        self.cloud_vis_diffuse = None
        self.cloud_ir_beam = None
        self.cloud_ir_diffuse = None

        # Canopy corrected
        self.veg_vis_beam = None
        self.veg_vis_diffuse = None
        self.veg_ir_beam = None
        self.veg_ir_diffuse = None

        # Net Solar
        self.net_solar = None

    def distribute(
        self,
        date_time: datetime,
        cloud_factor: np.ndarray,
        illumination_angles: np.ndarray,
        cos_z: float,
        azimuth: float,
        albedo_vis: np.ndarray,
        albedo_ir: np.ndarray,
    ) -> None:
        """
        Distribute solar radiation given a Panda's dataframe for a single time
        step.

        If the sun is up, i.e. ``cos_z > 0``, then the following steps are
        performed:

        1. Model clear sky radiation
        2. Cloud correct with :mod:`!smrf.distribute.solar.solar.cloud_correct`
        3. vegetation correct with
            :mod:`!smrf.distribute.solar.solar.veg_correct`
        4. Calculate net radiation with
            :mod:`!smrf.distribute.solar.solar.calc_net`

        If sun is down, then initialized attributes will maintain `None`,
        signaling the output functions to put zeros in their place.

        Sets
         * :py:attr:`clear_vis_beam`
         * :py:attr:`clear_vis_diffuse`
         * :py:attr:`clear_ir_beam`
         * :py:attr:`clear_ir_diffuse`

        Args:
            date_time: Current processed datetime
            cloud_factor: Numpy array of the domain for cloud factor
            illumination_angles: Illumination angels from :py:mod:`topocalc.illumination_angle`
            cos_z: cosine of the zenith angle for the basin, from
                :py:mod:`smrf.envphys.radiation.sunang`
            azimuth: azimuth to the sun for the basin, from
                :py:mod:`smrf.envphys.radiation.sunang`
            albedo_vis: numpy array for visible albedo, from
                :py:mod:`smrf.distribute.albedo.Albedo.albedo_vis`
            albedo_ir: numpy array for infrared albedo, from
                :py:mod:`smrf.distribute.albedo.Albedo.albedo_ir`

        """

        self._logger.debug(f"{date_time} Distributing solar")

        # Only calculate solar if the sun is up
        if cos_z > 0:
            self.cloud_factor = cloud_factor.copy()

            # Clear sky radiation
            self.clear_ir_beam, self.clear_ir_diffuse, horizon_angles = (
                self.calc_stoporad(
                    date_time,
                    illumination_angles,
                    cos_z,
                    azimuth,
                    albedo_ir,
                    IR_WAVELENGTHS,
                )
            )

            self.ir_beam = self.clear_ir_beam.copy()
            self.ir_diffuse = self.clear_ir_diffuse.copy()

            self.clear_vis_beam, self.clear_vis_diffuse, horizon_angles = (
                self.calc_stoporad(
                    date_time,
                    illumination_angles,
                    cos_z,
                    azimuth,
                    albedo_vis,
                    VISIBLE_WAVELENGTHS,
                    horizon_angles,
                )
            )

            self.vis_beam = self.clear_vis_beam.copy()
            self.vis_diffuse = self.clear_vis_diffuse.copy()

            # Correct clear sky for cloud cover
            if self.config["correct_cloud"]:
                self.cloud_correct()
                # Copy output to store as output files when requested
                self.cloud_vis_beam = self.vis_beam.copy()
                self.cloud_vis_diffuse = self.vis_diffuse.copy()
                self.cloud_ir_beam = self.ir_beam.copy()
                self.cloud_ir_diffuse = self.ir_diffuse.copy()

            # Correct cloud for vegetation
            if self.config["correct_veg"]:
                self.veg_correct(illumination_angles)
                # Copy output to store as output files when requested
                self.veg_vis_beam = self.vis_beam.copy()
                self.veg_vis_diffuse = self.vis_diffuse.copy()
                self.veg_ir_beam = self.ir_beam.copy()
                self.veg_ir_diffuse = self.ir_diffuse.copy()

            # Calculate net radiation
            self.calc_net(albedo_vis, albedo_ir)

        else:
            self._logger.debug("Sun is down, see you in the morning!")

    def cloud_correct(self):
        """
        Correct the modeled clear sky radiation for cloud cover using
        py:mod:`smrf.envphys.radiation.cloud.cf_cloud`.

        Sets
         * :py:attr:`cloud_vis_beam`
         * :py:attr:`cloud_vis_diffuse`
        """

        self._logger.debug("Correcting clear sky radiation for clouds")

        # Visible
        self.vis_beam, self.vis_diffuse = cloud.cf_cloud(
            self.vis_beam, self.vis_diffuse, self.cloud_factor
        )

        # IR
        self.ir_beam, self.ir_diffuse = cloud.cf_cloud(
            self.ir_beam, self.ir_diffuse, self.cloud_factor
        )

    def veg_correct(self, illumination_angles: np.ndarray) -> None:
        """
        Correct the cloud adjusted radiation for vegetation using
        :py:mod:`smrf.envphys.radiation.vegetation.veg_beam` and
        :py:mod:`smrf.envphys.radiation.vegetation.veg_diffuse`.

        Sets:
         * :py:attr:`veg_vis_beam`
         * :py:attr:`veg_vis_diffuse`
         * :py:attr:`veg_ir_beam`
         * :py:attr:`veg_ir_diffuse`

        Args:
            illumination_angles: Illumination angles from :py:mod:`smrf.envphys.radiation.sunang`
        """

        self._logger.debug("Correcting radiation for vegetation")

        # Visible
        ## Correct beam
        self.vis_beam = vegetation.solar_veg_beam(
            self.vis_beam, self.veg_height, illumination_angles, self.veg_k
        )

        ## Correct diffuse
        self.vis_diffuse = vegetation.solar_veg_diffuse(self.vis_diffuse, self.veg_tau)

        # IR
        ## Correct beam
        self.ir_beam = vegetation.solar_veg_beam(
            self.ir_beam, self.veg_height, illumination_angles, self.veg_k
        )

        ## Correct diffuse
        self.ir_diffuse = vegetation.solar_veg_diffuse(self.ir_diffuse, self.veg_tau)

    def calc_net(self, albedo_vis: np.ndarray, albedo_ir: np.ndarray,) -> None:
        """
        Calculate the net radiation using the vegetation adjusted radiation.
        Sets :py:attr:`net_solar`.

        Args:
            albedo_vis: numpy array for visible albedo, from
                :mod:`smrf.distribute.albedo.Albedo.albedo_vis`
            albedo_ir: numpy array for infrared albedo, from
                :mod:`smrf.distribute.albedo.Albedo.albedo_ir`
        """

        self._logger.debug("Calculating net radiation")

        # Visible
        vv_n = (self.vis_beam + self.vis_diffuse) * (1 - albedo_vis)
        vv_n = utils.set_min_max(vv_n, self.min, self.max)

        # IR
        vir_n = (self.ir_beam + self.ir_diffuse) * (1 - albedo_ir)
        vir_n = utils.set_min_max(vir_n, self.min, self.max)

        # Net Solar
        self.net_solar = vv_n + vir_n
        self.net_solar = utils.set_min_max(self.net_solar, self.min, self.max)

    def calc_stoporad(
        self,
        date_time: datetime,
        illumination_angles: np.ndarray,
        cos_z: float,
        azimuth: float,
        albedo_surface: np.ndarray,
        wavelength_range=VISIBLE_WAVELENGTHS,
        horizon_angles=None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run stoporad for the given date_time and wavelength range.

        Args:
            date_time (datetime): datetime object
            illumination_angles (np.array): numpy array of cosing of local illumination
                angles
            cos_z (float): cosine of the zenith angle for the basin
            azimuth (float): azimuth to the sun for the basin
            albedo_surface (np.array): albedo should match wavelengths
                specified
            wavelength_range (list, optional): wavelengths to integrate over.
                Defaults to [0.28, 0.7].
            horizon_angles (list, optional): Cached calculated angles.

        Returns:
            Tuple: clear sky beam and diffuse radiation
        """

        return toporad.stoporad(
            date_time,
            self.topo,
            cos_z,
            azimuth,
            illumination_angles,
            albedo_surface,
            wavelength_range=wavelength_range,
            horizon_angles=horizon_angles,
            tau_elevation=self.config["clear_opt_depth"],
            tau=self.config["clear_tau"],
            omega=self.config["clear_omega"],
            scattering_factor=self.config["clear_gamma"],
        )
