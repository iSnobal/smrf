import unittest

import numpy as np
import pandas as pd

from smrf.envphys.solar.cloud import cloud_factor

INDEX = pd.date_range("2025-10-01 10:00", periods=4, freq="h")


class TestCloudFactor(unittest.TestCase):
    def test_low_sun_is_interpolated_from_neighbors(self):
        ratio = pd.DataFrame({"cell": [0.5, 9.0, 0.7, 0.9]}, index=INDEX)
        clear_sky = pd.DataFrame({"cell": [400.0, 10.0, 400.0, 400.0]}, index=INDEX)

        cloud = cloud_factor(ratio, clear_sky)

        np.testing.assert_allclose(cloud["cell"], [0.5, 0.6, 0.7, 0.9])

    def test_edge_low_sun_takes_first_valid_value(self):
        ratio = pd.DataFrame({"cell": [9.0, 0.4, 0.5, 0.6]}, index=INDEX)
        clear_sky = pd.DataFrame({"cell": [10.0, 400.0, 400.0, 400.0]}, index=INDEX)

        cloud = cloud_factor(ratio, clear_sky)

        np.testing.assert_allclose(cloud["cell"], [0.4, 0.4, 0.5, 0.6])

    def test_result_is_clipped_to_unit_interval(self):
        ratio = pd.DataFrame({"cell": [1.5, -0.2, 0.5, 0.5]}, index=INDEX)
        clear_sky = pd.DataFrame({"cell": [400.0, 400.0, 400.0, 400.0]}, index=INDEX)

        cloud = cloud_factor(ratio, clear_sky)

        np.testing.assert_allclose(cloud["cell"], [1.0, 0.0, 0.5, 0.5])
