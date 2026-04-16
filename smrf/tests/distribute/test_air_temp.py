import unittest
from unittest import mock

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt

from smrf.distribute import AirTemperature
from smrf.tests.distribute import TOPO_MOCK

CONFIG = {
    "time": {
        "start_date": "2025-09-20 00:00",
        "time_zone": "utc",
    },
    "air_temp": {
        "distribution": "grid",
        "grid_local": True,
        "grid_local_n": 25,
    },
}
TIMESTEP = pd.to_datetime("2025-01-01 00:00:00")
DEM = np.array([[100, 150], [120, 200]])


class TestAirTemp(unittest.TestCase):
    def setUp(self):
        TOPO_MOCK.dem = DEM
        self.subject = AirTemperature(config=CONFIG, topo=TOPO_MOCK)

    @mock.patch.object(AirTemperature, "_distribute")
    @mock.patch("smrf.distribute.air_temp.adjust_by_elevation")
    def test_distribute(self, mock_adjust_by_elevation, mock_distribute):
        air_temp = pd.DataFrame([[1.0, 1.5], [1.2, 2.0]])
        air_temp.name = "air_temp"
        # Simulate the distribute call
        # It is mocked for this method and verified below
        self.subject.air_temp = air_temp

        self.subject.distribute(air_temp, TIMESTEP)

        mock_distribute.assert_called_once()
        call_args = mock_distribute.call_args[0]
        pdt.assert_frame_equal(air_temp, call_args[0])

        call_args = mock_adjust_by_elevation.call_args[0]

        pdt.assert_frame_equal(air_temp, call_args[0])
        npt.assert_equal(DEM, call_args[1])
        self.assertEqual(0.8856932782309219, call_args[2])
