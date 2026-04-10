import unittest

import numpy as np
from numpy.polynomial import Polynomial
import numpy.testing as npt
import pandas as pd

from smrf.distribute import AirTemperature
from smrf.tests.distribute import TOPO_MOCK

CONFIG = {
    "time": {
        "start_date": "2025-09-20 00:00",
        "time_zone": "utc",
    },
}
TIMESTEP = pd.to_datetime("2025-10-01 00:00:00")
DEM = np.array([[100, 150], [120, 200]])


class TestAirTemp(unittest.TestCase):
    def setUp(self):
        TOPO_MOCK.dem = DEM
        self.subject = AirTemperature(config=CONFIG, topo=TOPO_MOCK)

    def test_lapse_rate(self):
        orig_temp = np.array([[1., 2.], [3., 2.]])
        self.subject.air_temp = orig_temp.copy()

        self.subject.adjust_lapse_rate(TIMESTEP)

        poly_1st = Polynomial.fit(DEM.ravel(), orig_temp.ravel(), 1)
        month_lapse_rate = 1.0872053395902217

        # Math of the method
        orig_temp -= poly_1st(DEM.ravel()).reshape(DEM.shape)
        b, m = poly_1st.convert().coef
        slope = m * month_lapse_rate

        orig_temp += slope * DEM + b

        # Allow floating point noise
        npt.assert_allclose(self.subject.air_temp, orig_temp, atol=1e-15, rtol=0)
