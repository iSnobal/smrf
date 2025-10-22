"""
Distributed forcing data over a grid using interpolation
"""

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.interpolate.interpnd import _ndim_coords_from_arrays
from scipy.spatial import Delaunay, KDTree

from smrf.utils.utils import grid_interpolate_deconstructed


class Grid:
    """
    Linear interpolation between grid points.
    """

    CONFIG_KEY = "grid"

    def __init__(
        self,
        config,
        mx,
        my,
        grid_x,
        grid_y,
        mz=None,
        grid_z=None,
        mask=None,
        metadata=None,
    ):
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
                will be ignored if config['mask'] is false
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

        # local elevation gradient, precalculate the distance dataframe
        if config["grid_local"]:
            k = config["grid_local_n"]

            coords = metadata[["latitude", "longitude"]].to_numpy()
            tree = KDTree(coords, leafsize=k + 5)

            # Query k nearest neighbors.
            _distances, indices = tree.query(coords, k=k, workers=-1)

            dist_df = pd.DataFrame(
                metadata.index.to_numpy()[indices],
                index=metadata.index,
                columns=range(k),
            )
            dist_df.index.name = "cell_id"

            # stack and reset index
            df = dist_df.stack().reset_index()
            df = df.rename(columns={0: "cell_local"})
            df.drop("level_1", axis=1, inplace=True)

            # get the elevations
            df["elevation"] = metadata.loc[df.cell_local, "elevation"].values

            # now we have cell_id, cell_local and elevation for the whole grid
            self.full_df = df

            self.tri = None

        # mask
        self.mask = np.zeros_like(self.mx, dtype=bool)
        if config["grid_mask"]:
            assert mask.shape == grid_x.shape
            mask = mask.astype(bool)

            x = grid_x[0, :]
            y = grid_y[:, 0]
            for i, v in enumerate(mx):
                xi = np.argmin(np.abs(x - mx[i]))
                yi = np.argmin(np.abs(y - my[i]))

                self.mask[i] = mask[yi, xi]
        else:
            self.mask = np.ones_like(self.mx, dtype=bool)

    def detrended_interpolation(self, data, constrain_trend=0, grid_method="linear"):
        """
        Interpolate using a detrended approach

        Args:
            data: data to interpolate
            grid_method: scipy.interpolate.griddata interpolation method
            constrain_trend: Bool - Constrain the trend within configured bounds
        """
        if self.config["grid_local"]:
            rtrend = self.detrended_interpolation_local(data, constrain_trend, grid_method)
        else:
            rtrend = self.detrended_interpolation_mask(data, constrain_trend, grid_method)

        return rtrend

    @staticmethod
    def get_fit_params(group: pd.DataFrame) -> pd.Series:
        """
        Perform polyfit on each group and return the slope and intercept
        """
        slope, intercept = np.polyfit(group.elevation, group.data, 1)
        return pd.Series({"slope": slope, "intercept": intercept})

    def detrended_interpolation_local(self, data, constrain_trend=0, grid_method="linear"):
        """
        Interpolate using a detrended approach

        Args:
            data: data to interpolate
            grid_method: scipy.interpolate.griddata interpolation method
            constrain_trend: Bool - Constrain the trend within configured bounds
        """
        # take the new full_df and fill a data column
        df = self.full_df.copy()
        df["data"] = data[df["cell_local"]].values

        # Apply the custom function and aggregate the results
        df_fit_params = df.groupby("cell_id").apply(self.get_fit_params)

        # Merge the fit parameters with the original data,
        # ensuring only unique cell_id entries remain
        df = df.drop_duplicates(subset=["cell_id"], keep="first").set_index("cell_id")
        df = df.merge(df_fit_params, left_index=True, right_index=True)

        # apply trend constraints
        if constrain_trend == 1:
            df.loc[df["slope"] < 0, ["slope", "intercept"]] = 0
        elif constrain_trend == -1:
            df.loc[df["slope"] > 0, ["slope", "intercept"]] = 0

        # get triangulation
        if self.tri is None:
            xy = _ndim_coords_from_arrays((self.metadata.utm_x, self.metadata.utm_y))
            self.tri = Delaunay(xy)

        # interpolate the slope/intercept
        grid_slope = grid_interpolate_deconstructed(
            self.tri, df.slope.values[:], (self.GridX, self.GridY), method=grid_method
        )

        grid_intercept = grid_interpolate_deconstructed(
            self.tri,
            df.intercept.values[:],
            (self.GridX, self.GridY),
            method=grid_method,
        )

        # remove the elevation trend from the HRRR precip
        el_trend = df.elevation * df.slope + df.intercept
        dtrend = df.data - el_trend

        # interpolate the residuals over the DEM
        idtrend = grid_interpolate_deconstructed(
            self.tri, dtrend, (self.GridX, self.GridY), method=grid_method
        )

        # reinterpolate
        rtrend = idtrend + grid_slope * self.GridZ + grid_intercept

        return rtrend

    def detrended_interpolation_mask(self, data, constrain_trend=0, grid_method="linear"):
        """
        Interpolate using a detrended approach

        Args:
            data: data to interpolate
            grid_method: scipy.interpolate.griddata interpolation method
            constrain_trend: Bool - Constrain the trend within configured bounds
        """
        pv = np.polyfit(self.mz[self.mask].astype(float), data[self.mask], 1)

        # apply trend constraints
        if constrain_trend == 1 and pv[0] < 0:
            pv = np.array([0, 0])
        elif constrain_trend == -1 and pv[0] > 0:
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
