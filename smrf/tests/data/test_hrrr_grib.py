import unittest
from unittest.mock import MagicMock, Mock, patch, DEFAULT

import pandas as pd
from smrf.data.hrrr.grib_file_gdal import GribFileGdal
from smrf.data.hrrr_grib import InputGribHRRR
from smrf.data.load_topo import Topo
from smrf.distribute import ThermalHRRR
from smrf.distribute.wind.wind_ninja import WindNinjaModel


class TestInputGribHRRR(unittest.TestCase):
    TOPO_MOCK = MagicMock(spec=Topo, instance=True, zone_number=12)
    BBOX = [1, 2, 3, 4]
    START_DATE = pd.to_datetime("2021-01-01 00:00 UTC")
    END_DATE = pd.to_datetime("2021-01-02")
    SMRF_CONFIG = {
        "gridded": {
            "hrrr_directory": "/data/hrrr",
            "hrrr_forecast_hour": 1,
            "hrrr_sixth_hour_variables": ["var1"],
        },
        "output": {"variables": []},
    }

    def setUp(self):
        self.hrrr_input = InputGribHRRR(
            self.START_DATE,
            self.END_DATE,
            topo=self.TOPO_MOCK,
            bbox=self.BBOX,
            config=self.SMRF_CONFIG,
        )

    def test_init_default(self):
        self.assertEqual(self.START_DATE, self.hrrr_input.start_date)
        self.assertEqual(None, self.hrrr_input.cloud_factor_memory)
        self.assertEqual([], self.hrrr_input._load_gdal)
        self.assertEqual(
            GribFileGdal.DEFAULT_ALGORITHM, self.hrrr_input._gdal_algorithm
        )

    def test_init_load_gdal(self):
        hrrr_input = InputGribHRRR(
            self.START_DATE,
            self.END_DATE,
            topo=self.TOPO_MOCK,
            bbox=self.BBOX,
            config={
                "gridded": {
                    "hrrr_gdal_variables": ["VAR"],
                    "hrrr_gdal_algorithm": "nearest",
                },
                "output": {"variables": []},
            },
        )
        self.assertEqual(["VAR"], hrrr_input._load_gdal)
        self.assertEqual("nearest", hrrr_input._gdal_algorithm)

    def test_load_wind(self):
        hrrr_input = InputGribHRRR(
            self.START_DATE,
            self.END_DATE,
            topo=self.TOPO_MOCK,
            bbox=self.BBOX,
            config={**self.SMRF_CONFIG, "wind": {"wind_model": "other"}},
        )

        self.assertTrue(hrrr_input._load_wind)

    def test_skip_load_wind(self):
        hrrr_input = InputGribHRRR(
            self.START_DATE,
            self.END_DATE,
            topo=self.TOPO_MOCK,
            bbox=self.BBOX,
            config={
                **self.SMRF_CONFIG,
                "wind": {"wind_model": WindNinjaModel.MODEL_TYPE},
            },
        )

        self.assertFalse(hrrr_input._load_wind)

    @patch("smrf.data.hrrr_grib.FileLoader")
    @patch.object(InputGribHRRR, "parse_data")
    def test_load(self, mock_parse_data, mock_file_loader):
        file_loader = MagicMock()
        mock_file_loader.return_value = file_loader
        mock_data = {"variable": Mock(name="dataframe")}
        file_loader.data_for_time_and_topo.return_value = mock_data

        InputGribHRRR(
            self.START_DATE,
            self.END_DATE,
            topo=self.TOPO_MOCK,
            bbox=self.BBOX,
            config=self.SMRF_CONFIG,
        ).load()

        mock_file_loader.assert_called_once_with(
            external_logger=self.hrrr_input._logger,
            file_dir=self.SMRF_CONFIG["gridded"]["hrrr_directory"],
            forecast_hour=self.SMRF_CONFIG["gridded"]["hrrr_forecast_hour"],
            load_gdal=self.hrrr_input._load_gdal,
            gdal_algorithm=self.hrrr_input._gdal_algorithm,
            load_wind=self.hrrr_input._load_wind,
            sixth_hour_variables=self.SMRF_CONFIG["gridded"][
                "hrrr_sixth_hour_variables"
            ],
        )

        file_loader.data_for_time_and_topo.assert_called_once_with(
            start_date=self.START_DATE,
            bbox=self.BBOX,
            topo=self.TOPO_MOCK,
        )

        mock_parse_data.assert_called_once_with(mock_data)

    @patch("smrf.data.hrrr_grib.FileLoader")
    def test_get_metadata(self, mock_file_loader):
        file_loader = MagicMock()
        mock_file_loader.return_value = file_loader
        mock_data = MagicMock("metadata")
        file_loader.get_metadata.return_value = mock_data

        InputGribHRRR(
            self.START_DATE,
            self.END_DATE,
            topo=self.TOPO_MOCK,
            bbox=self.BBOX,
            config=self.SMRF_CONFIG,
        ).get_metadata()

        mock_file_loader.assert_called_once_with(
            external_logger=self.hrrr_input._logger,
            file_dir=self.SMRF_CONFIG["gridded"]["hrrr_directory"],
            forecast_hour=self.SMRF_CONFIG["gridded"]["hrrr_forecast_hour"],
        )

        file_loader.get_metadata.assert_called_once_with(
            date=self.START_DATE,
            bbox=self.BBOX,
            utm_zone_number=self.TOPO_MOCK.zone_number,
        )

    def test_parse_data(self):
        data = {
            "air_temp": pd.DataFrame(
                {
                    "date": pd.to_datetime("2025-01-01"),
                    "col_1": [1],
                    "col_2": [1],
                },
            ).set_index("date"),
            "relative_humidity": pd.DataFrame(
                {
                    "date": pd.to_datetime("2025-01-01"),
                    "col_1": [2],
                    "col_2": [2],
                },
            ).set_index("date"),
            "precip_int": pd.DataFrame(
                {
                    "date": pd.to_datetime("2025-01-01"),
                    "col_1": [3],
                    "col_2": [3],
                },
            ).set_index("date"),
        }

        subject = InputGribHRRR(
            self.START_DATE,
            self.END_DATE,
            topo=self.TOPO_MOCK,
            bbox=self.BBOX,
            config=self.SMRF_CONFIG,
        )
        subject._calculate_cloud_factor = False

        with patch.multiple(
            subject,
            calculate_wind=DEFAULT,
            check_cloud_factor=DEFAULT,
        ) as mocks:
            subject.parse_data(data)

            mocks["calculate_wind"].assert_called_once()
            mocks["check_cloud_factor"].assert_not_called()

        self.assertIsInstance(subject.air_temp, pd.DataFrame)
        self.assertIsInstance(subject.vapor_pressure, pd.DataFrame)
        self.assertIsInstance(subject.precip, pd.DataFrame)

    def test_parse_data_gdal_load(self):
        data = {
            "air_temp": pd.DataFrame(
                {
                    "date": pd.to_datetime("2025-01-01"),
                    "col_1": [1],
                    "col_2": [1],
                },
            ).set_index("date"),
            "relative_humidity": pd.DataFrame(
                {
                    "date": pd.to_datetime("2025-01-01"),
                    "col_1": [2],
                    "col_2": [2],
                },
            ).set_index("date"),
            "precip_int": pd.DataFrame(
                {
                    "date": pd.to_datetime("2025-01-01"),
                    "col_1": [3],
                    "col_2": [3],
                },
            ).set_index("date"),
            ThermalHRRR.GRIB_NAME: pd.DataFrame(
            {
                "date": pd.to_datetime("2025-01-01"),
                "col_1": [4],
                "col_2": [4],
            },
            ).set_index("date"),
        }

        self.SMRF_CONFIG["gridded"]["hrrr_gdal_variables"] = ["hrrr_thermal"]

        subject = InputGribHRRR(
            self.START_DATE,
            self.END_DATE,
            topo=self.TOPO_MOCK,
            bbox=self.BBOX,
            config=self.SMRF_CONFIG,
        )
        subject._calculate_cloud_factor = False

        with patch.multiple(
            subject,
            calculate_wind=DEFAULT,
            check_cloud_factor=DEFAULT,
        ) as mocks:
            subject.parse_data(data)

            mocks["calculate_wind"].assert_called_once()
            mocks["check_cloud_factor"].assert_not_called()

        self.assertIsInstance(subject.air_temp, pd.DataFrame)
        self.assertIsInstance(subject.vapor_pressure, pd.DataFrame)
        self.assertIsInstance(subject.precip, pd.DataFrame)
        self.assertIsInstance(subject.thermal, pd.DataFrame)
