import numexpr as ne
import numpy as np
import numpy.typing as npt

from smrf.distribute.albedo import Albedo


class NetSolar:
    VIS_ALBEDO_RATIO = np.float32(0.54)
    IR_ALBEDO_RATIO = np.float32(0.46)

    @staticmethod
    def broadband_albedo(solar: npt.NDArray, albedo: Albedo) -> npt.NDArray:
        """
        Calculate net solar based on a broadband albedo value.

        :param solar: Incoming shortwave radiation
        :param albedo: Instance of :py:class:`smrf.distribute.albedo.Albedo`

        :return:
            Numpy array with net solar radiation absorbed by the snowpack
        """
        params = {
            "MAX_ALBEDO": albedo.MAX_ALBEDO,
            "solar": solar,
            "albedo": albedo.albedo.astype(np.float32, copy=False, order="C"),
        }

        return ne.evaluate(
            "solar * (MAX_ALBEDO - albedo)", local_dict=params, casting="safe"
        )

    @staticmethod
    def broadband_from_vis_ir(solar: npt.NDArray, albedo: Albedo) -> npt.NDArray:
        """
        Calculate net solar based on a visible and infrared albedo ratio to
        compute broadband. The ratio is calculated for a Northern Latitude location
        in the Western US.

        :param solar: Incoming shortwave radiation
        :param albedo: Instance of :py:class:`smrf.distribute.albedo.Albedo`

        :return:
            Numpy array with net solar radiation absorbed by the snowpack
        """
        params = {
            "VIS_RATIO": NetSolar.VIS_ALBEDO_RATIO,
            "IR_RATIO": NetSolar.IR_ALBEDO_RATIO,
            "MAX_ALBEDO": albedo.MAX_ALBEDO,
            "solar": solar,
            "albedo_vis": albedo.albedo_vis.astype(np.float32, copy=False, order="C"),
            "albedo_ir": albedo.albedo_ir.astype(np.float32, copy=False, order="C"),
        }

        return ne.evaluate(
            "solar * (MAX_ALBEDO - (VIS_RATIO * albedo_vis + IR_RATIO * albedo_ir))",
            local_dict=params,
            casting="safe",
        )

    @staticmethod
    def albedo_diffuse_and_direct(
        direct: npt.NDArray, diffuse: npt.NDArray, albedo: Albedo
    ) -> npt.NDArray:
        """
        Calculate net solar by first applying the direct and diffuse albedo corrections
        and then computing the total net solar.

        :param direct: Numpy array with direct radiation
        :param diffuse: Numpy array with diffuse radiation
        :param albedo: Instance of :py:class:`smrf.distribute.albedo.Albedo`

        :return:
            Numpy array with net solar radiation absorbed by the snowpack
        """
        params = {
            "MAX_ALBEDO": albedo.MAX_ALBEDO,
            "direct": direct.astype(np.float32, copy=False, order="C"),
            "diffuse": diffuse.astype(np.float32, copy=False, order="C"),
            "albedo_direct": albedo.albedo_direct.astype(
                np.float32, copy=False, order="C"
            ),
            "albedo_diffuse": albedo.albedo_diffuse.astype(
                np.float32, copy=False, order="C"
            ),
        }

        return ne.evaluate(
            "direct * (MAX_ALBEDO - albedo_direct) + diffuse * (MAX_ALBEDO - albedo_diffuse)",
            local_dict=params,
            casting="safe",
        )
