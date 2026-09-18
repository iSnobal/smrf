import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.testing as npt

from smrf.distribute import VaporPressure
from smrf.tests.distribute import topo_mock

RAW_DATA_MOCK = np.ones((1, 1))
# Temperatures in C: 25, nan, 0, -10
TEMPERATURES = np.array(
    [[298.15, 274.15], [273.15, 263.15]], dtype=np.float64, order="C"
)

DATA_MOCK = MagicMock(thermal=RAW_DATA_MOCK)

CONFIG = {
    "time": {
        "start_date": "2025-09-20 00:00",
        "time_zone": "utc",
    },
    "vapor_pressure": {
        "dew_point_tolerance": 0.01,
        "threads": 1,
    },
    "precip": {
        "precip_temp_method": "wet_bulb",
    },
}


class TestVaporPressure(unittest.TestCase):
    def setUp(self):
        self.subject = VaporPressure(config=CONFIG, topo=topo_mock())

    @patch("smrf.distribute.VaporPressure._distribute")
    def test_wet_bulb_temperature(self, mock_distribute):
        self.subject.vapor_pressure = np.array(
            [[3167.9, 657.1], [611.3, 259.7]], dtype=np.float64, order="C"
        )

        self.subject.distribute(DATA_MOCK, TEMPERATURES)
        mock_distribute.assert_called_once()

        expected_vp = np.array(
            [[61.993499, 57.021897], [56.526332, 55.035582]],
            dtype=np.float64,
            order="C",
        )
        npt.assert_array_almost_equal(
            expected_vp,
            self.subject.precip_temp,
            decimal=2,
        )
