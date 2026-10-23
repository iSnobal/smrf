import unittest

import numpy as np
import numpy.testing as npt
import pandas as pd

from smrf.spatial.grid import Grid
from smrf.tests.smrf_config import SMRFConfig

GRID_X, GRID_Y = np.meshgrid(np.linspace(0, 1, 3), np.linspace(0, 1, 3))
GRID_Z = GRID_X + 10 * GRID_Y
METADATA = pd.DataFrame(
    {
        "latitude": [0.0, 0.0, 1.0, 1.0],
        "longitude": [0.0, 1.0, 0.0, 1.0],
        "utm_x": [0.0, 1.0, 0.0, 1.0],
        "utm_y": [0.0, 0.0, 1.0, 1.0],
        "elevation": [100.0, 200.0, 150.0, 250.0],
    },
    index=["station 1", "station 2", "station 3", "station 4"],
)
LOCAL_CONFIG = {"grid_local": True, "grid_local_n": 3, "grid_mask": False}
NON_LOCAL_CONFIG = {"grid_local": False, "grid_mask": False}


def _grid(config):
    return Grid(
        config,
        METADATA.utm_x.values,
        METADATA.utm_y.values,
        GRID_X,
        GRID_Y,
        METADATA.elevation.values,
        GRID_Z,
        metadata=METADATA,
    )


class TestGridInit(SMRFConfig, unittest.TestCase):
    def test_local_n_below_two_raises(self):
        """Local detrend rejects a neighbor count of 1, which cannot fit a trend"""
        config = {"grid_local": True, "grid_local_n": 1, "grid_mask": False}

        with self.assertRaises(ValueError):
            _grid(config)


class TestGridMask(SMRFConfig, unittest.TestCase):
    def test_mask_descending_y(self):
        """Mask lookup picks the right cells when the grid y-axis descends"""
        grid_x, grid_y = np.meshgrid([0.0, 1.0, 2.0], [2.0, 1.0, 0.0])
        mask = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
        mx = np.array([1.0, 0.0, 0.0, 2.0])
        my = np.array([1.0, 2.0, 0.0, 2.0])
        config = {"grid_local": False, "grid_mask": True}

        subject = Grid(
            config,
            mx,
            my,
            grid_x,
            grid_y,
            np.zeros(4),
            np.zeros_like(grid_x),
            mask=mask,
        )

        npt.assert_array_equal(subject.mask, [True, False, True, False])

    def test_mask_ascending_y(self):
        """Mask lookup picks the same cells as the descending case when y ascends"""
        grid_x, grid_y = np.meshgrid([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
        mask = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
        mx = np.array([1.0, 0.0, 0.0, 2.0])
        my = np.array([1.0, 2.0, 0.0, 2.0])
        config = {"grid_local": False, "grid_mask": True}

        subject = Grid(
            config,
            mx,
            my,
            grid_x,
            grid_y,
            np.zeros(4),
            np.zeros_like(grid_x),
            mask=mask,
        )

        npt.assert_array_equal(subject.mask, [True, False, True, False])

    def test_mask_nearest_cell(self):
        """Station between two grid cells takes the mask value of the nearest cell"""
        grid_x, grid_y = np.meshgrid([0.0, 1.0], [0.0, 1.0])
        mask = np.array([[0, 1], [0, 0]])
        mx = np.array([0.6, 0.4])
        my = np.array([0.1, 0.1])
        config = {"grid_local": False, "grid_mask": True}

        subject = Grid(
            config,
            mx,
            my,
            grid_x,
            grid_y,
            np.zeros(2),
            np.zeros_like(grid_x),
            mask=mask,
        )

        npt.assert_array_equal(subject.mask, [True, False])

    def test_mask_disabled_uses_all_stations(self):
        """All stations are used for the detrend when grid_mask is disabled"""
        subject = _grid(NON_LOCAL_CONFIG)

        npt.assert_array_equal(subject.mask, [True, True, True, True])


class TestGridLocalDetrend(SMRFConfig, unittest.TestCase):
    def test_detrended_interpolation_local_aligns_data_to_metadata(self):
        """Local detrend gives the same result when the data Series is in a different order"""
        subject = _grid(LOCAL_CONFIG)
        data = pd.Series([1.0, 2.0, 3.0, 5.0], index=METADATA.index)

        expected = subject.detrended_interpolation(data)
        result = subject.detrended_interpolation(data.iloc[::-1])

        npt.assert_allclose(result, expected)

    def test_detrended_interpolation_local_recovers_linear_trend(self):
        """Local detrend reproduces an exact elevation trend over the grid"""
        subject = _grid(LOCAL_CONFIG)
        data = pd.Series(2 * METADATA.elevation.values + 5, index=METADATA.index)

        result = subject.detrended_interpolation(data)

        npt.assert_allclose(result, 2 * GRID_Z + 5)


class TestGridMaskDetrend(SMRFConfig, unittest.TestCase):
    def test_detrended_interpolation_mask_aligns_data_to_metadata(self):
        """Mask detrend gives the same result when the data Series is in a different order"""
        subject = _grid(NON_LOCAL_CONFIG)
        data = pd.Series([1.0, 2.0, 3.0, 5.0], index=METADATA.index)

        expected = subject.detrended_interpolation(data)
        result = subject.detrended_interpolation(data.iloc[::-1])

        npt.assert_allclose(result, expected)

    def test_detrended_interpolation_mask_recovers_linear_trend(self):
        """Mask detrend reproduces an exact elevation trend over the grid"""
        subject = _grid(NON_LOCAL_CONFIG)
        data = pd.Series(2 * METADATA.elevation.values + 5, index=METADATA.index)

        result = subject.detrended_interpolation(data)

        npt.assert_allclose(result, 2 * GRID_Z + 5)

    def test_detrended_interpolation_mask_constrains_negative_slope(self):
        """Negative elevation slope is zeroed when constrain_trend is 1"""
        subject = _grid(NON_LOCAL_CONFIG)
        data = pd.Series(-2 * METADATA.elevation.values + 5, index=METADATA.index)

        subject.detrended_interpolation_mask(data.values, constrain_trend=1)

        npt.assert_array_equal(subject.pv, [0, 0])

    def test_detrended_interpolation_mask_keeps_positive_slope(self):
        """Positive elevation slope is kept when constrain_trend is 1"""
        subject = _grid(NON_LOCAL_CONFIG)
        data = pd.Series(2 * METADATA.elevation.values + 5, index=METADATA.index)

        subject.detrended_interpolation_mask(data.values, constrain_trend=1)

        npt.assert_allclose(subject.pv, [2, 5])


class TestGridInterpolation(SMRFConfig, unittest.TestCase):
    def test_calculate_interpolation_linear_surface(self):
        """Plain grid interpolation reproduces a linear surface over the grid"""
        subject = _grid(NON_LOCAL_CONFIG)
        data = 2 * METADATA.utm_x.values + 3 * METADATA.utm_y.values + 1

        result = subject.calculate_interpolation(data)

        npt.assert_allclose(result, 2 * GRID_X + 3 * GRID_Y + 1)
