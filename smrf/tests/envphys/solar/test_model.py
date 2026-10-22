import unittest
from datetime import datetime

from smrf.envphys import sunang
from smrf.envphys.solar.irradiance import direct_solar_irradiance
from smrf.envphys.solar.model import model_solar
from smrf.envphys.solar.twostream import twostream

# Center of the Reynolds Mountain East test basin (Idaho, USA)
BASIN_LAT = 43.0647537
BASIN_LON = -116.7539542


class TestModelSolar(unittest.TestCase):
    def test_sun_up(self):
        # Around local solar noon in January (19:00 UTC)
        date_time = datetime(2026, 1, 1, 19)

        cosz, _azimuth, _rad_vec = sunang.sunang(date_time, BASIN_LAT, BASIN_LON)
        self.assertGreater(cosz, 0.0)

        expected = twostream(
            cosz, direct_solar_irradiance(date_time, [0.28, 2.8]), tau=0.3
        )["irradiance_at_bottom"][0]

        result = model_solar(date_time, BASIN_LAT, BASIN_LON, tau=0.3)

        self.assertGreater(result, 0.0)
        self.assertEqual(result, expected)

    def test_sun_down_returns_zero(self):
        # Night time in January (06:00 UTC)
        date_time = datetime(2026, 1, 1, 6)

        cosz, _azimuth, _rad_vec = sunang.sunang(date_time, BASIN_LAT, BASIN_LON)
        self.assertLess(cosz, 0.0)

        result = model_solar(date_time, BASIN_LAT, BASIN_LON)

        self.assertEqual(result, 0)
