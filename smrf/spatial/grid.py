"""
Distributed forcing data over a grid using interpolation
"""

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.spatial import Delaunay, KDTree

from smrf.utils.utils import grid_interpolate_deconstructed


class Grid:
    """
    Linear interpolation between grid points.
    """

    CONFIG_KEY = "grid"

    def __init__(
        self,
        config: dict,
        mx: np.ndarray,
        my: np.ndarray,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        mz: np.ndarray,
        grid_z: np.ndarray,
        mask: np.ndarray | None = None,
        metadata: pd.DataFrame | None = None,
    ) -> None:
        """
        Args:
            config: configuration for grid interpolation
            mx: x locations for the points
            my: y locations for the points
            mz: z locations for the points
            grid_x: x locations in grid to interpolate over
            grid_y: y locations in grid to interpolate over
            grid_z: z locations in grid to interpolate over
            mask: mask for those points to include in the detrending
                will be ignored if config['grid_mask'] is false
        """
        self.config = config

        # measurement point locations
        self.mx = mx
        self.my = my
        self.mz = mz
        self.npoints = len(mx)

        # grid information
        self.GridX = grid_x
        self.GridY = grid_y
        self.GridZ = grid_z

        self.metadata = metadata

        # precalculate the k nearest neighbors for the local detrend
        if config["grid_local"]:
            k = config["grid_local_n"]
            if k < 2:
                raise ValueError(
                    "grid_local_n must be at least 2 for the local detrend"
                )

            coords = metadata[["latitude", "longitude"]].to_numpy()
            tree = KDTree(coords, leafsize=k + 5)

            # Query k nearest neighbors. Returns shape (N, k)
            _distances, indices = tree.query(coords, k=k, workers=-1)

            self.neighbor_indices = indices
            self.neighbor_elevation = metadata["elevation"].to_numpy()[indices]

            # design matrix and pseudo-inverse depend only on elevation
            A = np.stack(
                (self.neighbor_elevation, np.ones_like(self.neighbor_elevation)),
                axis=-1,
            )
            self.pinv_A = np.linalg.pinv(A)

        self.tri = None

        # mask
        if config["grid_mask"]:
            assert mask.shape == grid_x.shape

            # Nearest cell by distance, works for south-to-north and north-to-south grids
            x = grid_x[0, :]
            y = grid_y[:, 0]
            xi = np.abs(x[None, :] - mx[:, None]).argmin(axis=1)
            yi = np.abs(y[None, :] - my[:, None]).argmin(axis=1)
            self.mask = mask.astype(bool)[yi, xi]
        else:
            self.mask = np.ones_like(self.mx, dtype=bool)

    def detrended_interpolation(self, data, constrain_trend=0, grid_method="linear"):
        """
        Interpolate using a detrended approach

        Args:
            data: data to interpolate
            grid_method: scipy.interpolate.griddata interpolation method
            constrain_trend: 1 zeros negative slopes, -1 zeros positive, 0 for none
        """
        # Columns in the data file need not match the metadata row order
        data = data.loc[self.metadata.index].to_numpy()

        if self.config["grid_local"]:
            rtrend = self.detrended_interpolation_local(
                data, constrain_trend, grid_method
            )
        else:
            rtrend = self.detrended_interpolation_mask(
                data, constrain_trend, grid_method
            )

        return rtrend

    def detrended_interpolation_local(
        self, data, constrain_trend=0, grid_method="linear"
    ):
        """
        Interpolate using a detrended approach

        Args:
            data: data to interpolate
            grid_method: scipy.interpolate.griddata interpolation method
            constrain_trend: 1 zeros negative slopes, -1 zeros positive, 0 for none
        """
        # values and elevations of each cell's k neighbors, shape (N, k)
        values = data[self.neighbor_indices]
        elevation = self.neighbor_elevation

        # batched least squares via the precomputed pseudo-inverse, shape (N, 2, 1)
        params = self.pinv_A @ values[..., None]
        slope = np.ascontiguousarray(params[:, 0, 0])
        intercept = np.ascontiguousarray(params[:, 1, 0])

        # apply trend constraints
        if constrain_trend == 1:
            neg = slope < 0
            slope[neg] = 0
            intercept[neg] = 0
        elif constrain_trend == -1:
            pos = slope > 0
            slope[pos] = 0
            intercept[pos] = 0

        if self.tri is None:
            xy = np.column_stack((self.metadata.utm_x, self.metadata.utm_y))
            self.tri = Delaunay(xy)

        # remove the elevation trend from each cell's own value
        dtrend = values[:, 0] - (elevation[:, 0] * slope + intercept)

        # Interpolate slope, intercept, and residuals
        grid_values = grid_interpolate_deconstructed(
            self.tri,
            np.column_stack((slope, intercept, dtrend)),
            (self.GridX, self.GridY),
            method=grid_method,
        )
        grid_slope, grid_intercept, idtrend = np.moveaxis(grid_values, -1, 0)

        rtrend = idtrend + grid_slope * self.GridZ + grid_intercept

        return rtrend

    def detrended_interpolation_mask(
        self, data, constrain_trend=0, grid_method="linear"
    ):
        """
        Interpolate using a detrended approach

        Args:
            data: data to interpolate
            grid_method: scipy.interpolate.griddata interpolation method
            constrain_trend: 1 zeros negative slopes, -1 zeros positive, 0 for none
        """
        pv = np.polyfit(self.mz[self.mask].astype(float), data[self.mask], 1)

        # apply trend constraints
        if constrain_trend == 1 and pv[0] < 0 or constrain_trend == -1 and pv[0] > 0:
            pv = np.array([0, 0])

        self.pv = pv

        # detrend the data
        el_trend = self.mz * pv[0] + pv[1]
        dtrend = data - el_trend

        # interpolate over the DEM grid
        idtrend = griddata(
            (self.mx, self.my), dtrend, (self.GridX, self.GridY), method=grid_method
        )

        # retrend the data
        rtrend = idtrend + pv[0] * self.GridZ + pv[1]

        return rtrend

    def calculate_interpolation(self, data, grid_method="linear"):
        """
        Interpolate over the grid

        Args:
            data: data to interpolate
            grid_method: Method for interpolation
        """
        g = griddata(
            (self.mx, self.my), data, (self.GridX, self.GridY), method=grid_method
        )

        return g
