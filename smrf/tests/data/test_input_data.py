from unittest import mock

import smrf
import smrf.data as smrf_data

from smrf.framework.model_framework import SMRF
from smrf.tests.smrf_test_case import SMRFTestCase
from smrf.tests.smrf_test_case_lakes import SMRFTestCaseLakes


class TestInputData(SMRFTestCase):
    def setUp(self):
        self.smrf = SMRF(self.base_config)
        self.smrf.load_topo()

    @mock.patch.object(smrf_data.InputCSV, 'load')
    def test_csv_type(self, mock_load):
        input_data = smrf_data.InputData(
            self.smrf.config,
            self.smrf.start_date,
            self.smrf.end_date,
            self.smrf.topo,
        )

        self.assertEqual(
            smrf_data.InputCSV.DATA_TYPE,
            input_data.data_type
        )
        self.assertEqual(
            self.smrf.start_date,
            input_data.start_date
        )
        self.assertEqual(
            self.smrf.end_date,
            input_data.end_date
        )

        mock_load.assert_called()

    def test_missing_data_type(self):
        del self.smrf.config['csv']
        with self.assertRaisesRegex(AttributeError, 'Missing required'):
            smrf_data.InputData(
                self.smrf.config,
                self.smrf.start_date,
                self.smrf.end_date,
                self.smrf.topo,
            )


class TestInputDataGridded(SMRFTestCaseLakes):
    def setUp(self):
        self.smrf = SMRF(self.base_config_copy())
        self.smrf.load_topo()

    def assert_parameters(self):
        for parameter in ['start_date', 'end_date', 'topo']:
            self.assertEqual(
                getattr(self.smrf, parameter),
                getattr(self.input_data, parameter)
            )

    @mock.patch.object(smrf_data.InputGribHRRR, 'load')
    def test_data_type(self, mock_load):
        self.input_data = smrf_data.InputData(
            self.smrf.config,
            self.smrf.start_date,
            self.smrf.end_date,
            self.smrf.topo,
        )
        self.assertEqual(
            smrf_data.InputGribHRRR.DATA_TYPE,
            self.input_data.data_type
        )
        self.assert_parameters()

        mock_load.assert_not_called()

    @mock.patch.object(smrf_data.InputNetcdf, 'load')
    def test_netcdf_type(self, mock_load):
        self.smrf.config['gridded']['data_type'] = 'netcdf'
        self.smrf.config['gridded']['netcdf_file'] = {}
        self.input_data = smrf_data.InputData(
            self.smrf.config,
            self.smrf.start_date,
            self.smrf.end_date,
            self.smrf.topo,
        )

        self.assertEqual(
            smrf_data.InputNetcdf.DATA_TYPE,
            self.input_data.data_type
        )
        self.assert_parameters()

        mock_load.assert_called()

    @mock.patch.object(smrf_data.InputWRF, 'load')
    def test_wrf_type(self, mock_load):
        self.smrf.config['gridded']['data_type'] = 'wrf'
        self.input_data = smrf_data.InputData(
            self.smrf.config,
            self.smrf.start_date,
            self.smrf.end_date,
            self.smrf.topo,
        )

        self.assertEqual(
            smrf_data.InputWRF.DATA_TYPE,
            self.input_data.data_type
        )
        self.assert_parameters()

        mock_load.assert_called()

    def test_unknown_data_type(self):
        self.smrf.config['gridded']['data_type'] = 'unknown'
        with self.assertRaisesRegex(AttributeError, 'Unknown gridded'):
            self.input_data = smrf_data.InputData(
                self.smrf.config,
                self.smrf.start_date,
                self.smrf.end_date,
                self.smrf.topo,
            )
