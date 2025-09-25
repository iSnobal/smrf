import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from smrf.data.hrrr_grib import InputGribHRRR
from smrf.data.load_topo import Topo
from smrf.distribute.wind.wind_ninja import WindNinjaModel


class TestInputGribHRRR(unittest.TestCase):
    TOPO_MOCK = MagicMock(spec=Topo, instance=True)
    BBOX = [1, 2, 3, 4]
    START_DATE = pd.to_datetime('2021-01-01 00:00 UTC')
    END_DATE = pd.to_datetime('2021-01-02')
    SMRF_CONFIG = {"gridded": {}, "output": {"variables": []}}

    def setUp(self):
        self.hrrr_input = InputGribHRRR(
            self.START_DATE,
            self.END_DATE,
            topo=self.TOPO_MOCK,
            bbox=self.BBOX,
            config=self.SMRF_CONFIG,
        )

    def test_load_method_config(self):
        self.assertEqual(
            self.START_DATE,
            self.hrrr_input.start_date
        )
        self.assertEqual(
            self.START_DATE + pd.to_timedelta(20, 'minutes'),
            self.hrrr_input.end_date
        )
        self.assertEqual(None, self.hrrr_input.cloud_factor_memory)

    def test_load_wind(self):
        hrrr_input = InputGribHRRR(
            self.START_DATE, self.END_DATE,
            topo=self.TOPO_MOCK, bbox=self.BBOX,
            config={
                **self.SMRF_CONFIG,
                'wind': {'wind_model': 'other'}
            }
        )

        self.assertTrue(hrrr_input._load_wind)
        self.assertTrue('wind_speed' in hrrr_input.variables)
        self.assertTrue('wind_direction' in hrrr_input.variables)

    def test_skip_load_wind(self):
        hrrr_input = InputGribHRRR(
            self.START_DATE, self.END_DATE,
            topo=self.TOPO_MOCK, bbox=self.BBOX,
            config={
                **self.SMRF_CONFIG,
                'wind': {'wind_model': WindNinjaModel.MODEL_TYPE}
            }
        )

        self.assertFalse(hrrr_input._load_wind)
        self.assertCountEqual(
            InputGribHRRR.VARIABLES,
            hrrr_input.variables
        )

    @patch.object(InputGribHRRR, "timestep_dates")
    @patch.object(InputGribHRRR, "load")
    def test_load_timestep_sets_start_date(self, _mock_load, _mock_timestep_dates):
        self.hrrr_input.load_timestep(self.START_DATE)

        self.assertEqual(self.hrrr_input.start_date, self.START_DATE)

    @patch.object(InputGribHRRR, "timestep_dates")
    @patch.object(InputGribHRRR, "load")
    def test_load_timestep_calls_methods(self, mock_load, mock_timestep_dates):
        self.hrrr_input.load_timestep(self.START_DATE)

        mock_timestep_dates.assert_called_once()
        mock_load.assert_called_once()
