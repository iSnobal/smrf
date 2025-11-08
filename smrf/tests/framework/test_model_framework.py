from unittest.mock import MagicMock

import numpy as np
import pytz
from inicheck.tools import cast_all_variables
from pandas import to_datetime
from smrf import distribute
from smrf.data import Topo
from smrf.framework.model_framework import SMRF
from smrf.tests.smrf_test_case import SMRFTestCase
from smrf.tests.smrf_test_case_lakes import SMRFTestCaseLakes

TOPO_MOCK = MagicMock(spec=Topo, sky_view_factor=np.array([[10.], [20.]]), instance=True)


class TestModelFramework(SMRFTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.smrf = SMRF(cls.config_file)

    def test_start_date(self):
        self.assertEqual(
            self.smrf.start_date,
            to_datetime(self.smrf.config["time"]["start_date"], utc=True),
        )

    def test_end_date(self):
        self.assertEqual(
            self.smrf.end_date,
            to_datetime(self.smrf.config["time"]["end_date"], utc=True),
        )

    def test_time_zone(self):
        self.assertEqual(self.smrf.time_zone, pytz.UTC)

    def test_date_time(self):
        self.assertEqual(
            self.smrf.date_time[0], to_datetime("1998-01-14 15:00:00", utc=True)
        )
        self.assertEqual(
            self.smrf.date_time[-1], to_datetime("1998-01-14 19:00:00", utc=True)
        )
        self.assertEqual(self.smrf.date_time[0].tzname(), str(pytz.UTC))
        self.assertEqual(type(self.smrf.date_time), list)

    def test_assert_time_steps(self):
        self.assertEqual(self.smrf.time_steps, 5)


class TestModelFrameworkMST(SMRFTestCase):
    """
    Test timezone handling for MST.
    """

    TIMEZONE = pytz.timezone("MST")

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        base_config = cls.base_config_copy()
        base_config.cfg["time"]["time_zone"] = "MST"
        cls.smrf = SMRF(base_config)

    def test_timezone_error(self):
        base_config = self.base_config_copy()
        base_config.cfg["time"]["time_zone"] = "unkown"
        with self.assertRaises(Exception):
            SMRF(base_config)

    def test_start_date(self):
        self.assertEqual(
            self.smrf.start_date,
            to_datetime(self.smrf.config["time"]["start_date"]).tz_localize(
                self.TIMEZONE
            ),
        )

    def test_end_date(self):
        self.assertEqual(
            self.smrf.end_date,
            to_datetime(self.smrf.config["time"]["end_date"]).tz_localize(
                self.TIMEZONE
            ),
        )

    def test_time_zone(self):
        self.assertEqual(self.smrf.time_zone, self.TIMEZONE)

    def test_date_time(self):
        self.assertEqual(
            self.smrf.date_time[0],
            to_datetime("1998-01-14 15:00:00").tz_localize(self.TIMEZONE),
        )
        self.assertEqual(
            self.smrf.date_time[-1],
            to_datetime("1998-01-14 19:00:00").tz_localize(self.TIMEZONE),
        )
        self.assertEqual(self.smrf.date_time[0].tz.zone, self.TIMEZONE.zone)


class TestModelFrameworkLakes(SMRFTestCaseLakes):
    """
    Test initialization with a HRRR based run
    """

    def setUp(self):
        self.smrf = SMRF(self.base_config)
        self.smrf.topo = TOPO_MOCK

    def compare_distribute_keys(self, expected, smrf):
        """
        Helper method to compare the enqueued distribution variables so there are
        no duplicate entries
        """
        self.assertSetEqual(
            set(expected),
            set(smrf.distribute.keys() - ['soil_temp']), # Soil temperature is always enqueued
        )

    def test_init(self):
        self.assertTrue(self.smrf.gridded)
        self.assertEqual(True, self.smrf.load_hrrr)
        self.assertEqual([], self.smrf.config["gridded"]["hrrr_gdal_variables"])

    def test_distribute_default_gridded(self):
        self.smrf.create_distribution()

        self.compare_distribute_keys(
            [
                distribute.AirTemperature.DISTRIBUTION_KEY,
                distribute.VaporPressure.DISTRIBUTION_KEY,
                distribute.Wind.DISTRIBUTION_KEY,
                distribute.Precipitation.DISTRIBUTION_KEY,
                distribute.Thermal.DISTRIBUTION_KEY,
                distribute.Solar.DISTRIBUTION_KEY,
                distribute.Albedo.DISTRIBUTION_KEY,
                distribute.CloudFactor.DISTRIBUTION_KEY,
            ],
            self.smrf
        )
        for distribution_class in [
            distribute.AirTemperature,
            distribute.VaporPressure,
            distribute.Wind,
            distribute.Precipitation,
            distribute.Albedo,
            distribute.Thermal,
            distribute.Solar,
            distribute.CloudFactor,
            distribute.SoilTemperature,
        ]:
            self.assertIsInstance(
                self.smrf.distribute[distribution_class.DISTRIBUTION_KEY],
                distribution_class,
            )

    def test_distribute_hrrr_thermal(self):
        output_variables = ["hrrr_thermal"]
        config = self.base_config_copy()
        config.cfg["output"]["variables"] = output_variables
        config = cast_all_variables(config, config.mcfg)

        smrf = SMRF(config)
        smrf.topo = TOPO_MOCK

        self.assertEqual(set(output_variables), smrf.output_variables)
        self.assertEqual({}, smrf.distribute)

        smrf.create_distribution()

        self.compare_distribute_keys(
            [
                distribute.ThermalHRRR.DISTRIBUTION_KEY,
                distribute.AirTemperature.DISTRIBUTION_KEY,
            ],
            smrf
        )
        self.assertIsInstance(
            smrf.distribute[distribute.ThermalHRRR.DISTRIBUTION_KEY],
            distribute.ThermalHRRR,
        )
        # Special case handling for HRRR Thermal
        self.assertTrue(distribute.Thermal.DISTRIBUTION_KEY in smrf.output_variables)
        self.assertFalse(distribute.ThermalHRRR.INI_VARIABLE in smrf.output_variables)
        # Dependencies
        self.assertIsInstance(
            smrf.distribute[distribute.AirTemperature.DISTRIBUTION_KEY],
            distribute.AirTemperature,
        )

    def test_distribute_hrrr_solar(self):
        output_variables = [distribute.SolarHRRR.INI_VARIABLE]
        config = self.base_config_copy()
        config.cfg["output"]["variables"] = output_variables
        config = cast_all_variables(config, config.mcfg)

        smrf = SMRF(config)
        smrf.topo = TOPO_MOCK

        self.assertEqual(set(output_variables), smrf.output_variables)
        self.assertEqual({}, smrf.distribute)

        smrf.create_distribution()

        self.compare_distribute_keys(
            [
                distribute.Albedo.DISTRIBUTION_KEY,
                distribute.SolarHRRR.DISTRIBUTION_KEY,
                distribute.Precipitation.DISTRIBUTION_KEY,
                distribute.AirTemperature.DISTRIBUTION_KEY,
                distribute.VaporPressure.DISTRIBUTION_KEY,
            ],
            smrf
        )
        self.assertIsInstance(
            smrf.distribute[distribute.SolarHRRR.DISTRIBUTION_KEY],
            distribute.SolarHRRR,
        )
        self.assertTrue(distribute.SolarHRRR.INI_VARIABLE in smrf.output_variables)
        # Dependencies
        self.assertIsInstance(
            smrf.distribute[distribute.Precipitation.DISTRIBUTION_KEY],
            distribute.Precipitation,
        )
        self.assertIsInstance(
            smrf.distribute[distribute.Albedo.DISTRIBUTION_KEY],
            distribute.Albedo,
        )

    def test_solar_and_albedo_output(self):
        config = self.base_config_copy()
        config.cfg["output"]["variables"] = ["net_solar", "albedo_ir", "albedo_vis"]
        config = cast_all_variables(config, config.mcfg)

        smrf = SMRF(config)
        smrf.topo = TOPO_MOCK

        smrf.create_distribution()

        self.compare_distribute_keys(
            [
                distribute.Albedo.DISTRIBUTION_KEY,
                distribute.Solar.DISTRIBUTION_KEY,
                distribute.CloudFactor.DISTRIBUTION_KEY,
                distribute.Precipitation.DISTRIBUTION_KEY,
                distribute.AirTemperature.DISTRIBUTION_KEY,
                distribute.VaporPressure.DISTRIBUTION_KEY,
            ],
            smrf
        )
        self.assertIsInstance(
            smrf.distribute[distribute.Solar.DISTRIBUTION_KEY],
            distribute.Solar,
        )
        self.assertIsInstance(
            smrf.distribute[distribute.Albedo.DISTRIBUTION_KEY],
            distribute.Albedo,
        )

    def test_hrrr_solar_and_albedo_output(self):
        config = self.base_config_copy()
        config.cfg["output"]["variables"] = ["hrrr_solar", "albedo_ir", "albedo_vis"]
        config = cast_all_variables(config, config.mcfg)

        smrf = SMRF(config)
        smrf.topo = TOPO_MOCK

        smrf.create_distribution()

        self.compare_distribute_keys(
            [
                distribute.Albedo.DISTRIBUTION_KEY,
                distribute.SolarHRRR.DISTRIBUTION_KEY,
                distribute.Precipitation.DISTRIBUTION_KEY,
                distribute.AirTemperature.DISTRIBUTION_KEY,
                distribute.VaporPressure.DISTRIBUTION_KEY,
            ],
            smrf
        )
        self.assertIsInstance(
            smrf.distribute[distribute.SolarHRRR.DISTRIBUTION_KEY],
            distribute.SolarHRRR,
        )
        self.assertIsInstance(
            smrf.distribute[distribute.Albedo.DISTRIBUTION_KEY],
            distribute.Albedo,
        )

    def test_albedo_only_output(self):
        config = self.base_config_copy()
        config.cfg["output"]["variables"] = ["albedo_ir", "albedo_vis"]
        config = cast_all_variables(config, config.mcfg)

        smrf = SMRF(config)
        smrf.topo = TOPO_MOCK

        smrf.create_distribution()

        self.compare_distribute_keys([distribute.Albedo.DISTRIBUTION_KEY], smrf)
        self.assertIsInstance(
            smrf.distribute[distribute.Albedo.DISTRIBUTION_KEY],
            distribute.Albedo,
        )
