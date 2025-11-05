import pytz
from pandas import to_datetime

from unittest.mock import MagicMock

from smrf.framework.model_framework import SMRF
from smrf import distribute
from smrf.data import Topo

from smrf.tests.smrf_test_case import SMRFTestCase
from smrf.tests.smrf_test_case_lakes import SMRFTestCaseLakes

from inicheck.tools import cast_all_variables

TOPO_MOCK = MagicMock(spec=Topo, instance=True)

class TestModelFramework(SMRFTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.smrf = SMRF(cls.config_file)

    def test_start_date(self):
        self.assertEqual(
            self.smrf.start_date,
            to_datetime(self.smrf.config['time']['start_date'], utc=True)
        )

    def test_end_date(self):
        self.assertEqual(
            self.smrf.end_date,
            to_datetime(self.smrf.config['time']['end_date'], utc=True)
        )

    def test_time_zone(self):
        self.assertEqual(self.smrf.time_zone, pytz.UTC)

    def test_date_time(self):
        self.assertEqual(
            self.smrf.date_time[0],
            to_datetime('1998-01-14 15:00:00', utc=True)
        )
        self.assertEqual(
            self.smrf.date_time[-1],
            to_datetime('1998-01-14 19:00:00', utc=True)
        )
        self.assertEqual(
            self.smrf.date_time[0].tzname(),
            str(pytz.UTC)
        )
        self.assertEqual(
            type(self.smrf.date_time),
            list
        )

    def test_assert_time_steps(self):
        self.assertEqual(self.smrf.time_steps, 5)


class TestModelFrameworkMST(SMRFTestCase):
    """
    Test timezone handling for MST.
    """
    TIMEZONE = pytz.timezone('MST')

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        base_config = cls.base_config_copy()
        base_config.cfg['time']['time_zone'] = 'MST'
        cls.smrf = SMRF(base_config)

    def test_timezone_error(self):
        base_config = self.base_config_copy()
        base_config.cfg['time']['time_zone'] = 'unkown'
        with self.assertRaises(Exception):
            SMRF(base_config)

    def test_start_date(self):
        self.assertEqual(
            self.smrf.start_date,
            to_datetime(self.smrf.config['time']['start_date']).tz_localize(
                self.TIMEZONE
            )
        )

    def test_end_date(self):
        self.assertEqual(
            self.smrf.end_date,
            to_datetime(self.smrf.config['time']['end_date']).tz_localize(
                self.TIMEZONE
            )
        )

    def test_time_zone(self):
        self.assertEqual(self.smrf.time_zone, self.TIMEZONE)

    def test_date_time(self):
        self.assertEqual(
            self.smrf.date_time[0],
            to_datetime('1998-01-14 15:00:00').tz_localize(self.TIMEZONE)
        )
        self.assertEqual(
            self.smrf.date_time[-1],
            to_datetime('1998-01-14 19:00:00').tz_localize(self.TIMEZONE)
        )
        self.assertEqual(
            self.smrf.date_time[0].tz.zone,
            self.TIMEZONE.zone
        )

class TestModelFrameworkLakes(SMRFTestCaseLakes):
    """
    Test initialization with a HRRR based run
    """
    def setUp(self):
        self.smrf = SMRF(self.base_config)
        self.smrf.topo = TOPO_MOCK

    def test_init(self):
        self.assertTrue(self.smrf.gridded)
        self.assertEqual(True, self.smrf.load_hrrr)
        self.assertEqual([], self.smrf.config["gridded"]["hrrr_gdal_variables"])

    def test_distribute_default_gridded(self):
        self.smrf.create_distribution()

        for distribution_class in [
            distribute.AirTemperature,
            distribute.VaporPressure,
            distribute.Wind,
            distribute.Precipitation,
            distribute.Albedo,
            distribute.Thermal,
            distribute.Solar,
            distribute.CloudFactor,
            distribute.SoilTemperature
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

        self.assertEquals(set(output_variables), smrf.output_variables)
        self.assertEquals({}, smrf.distribute)

        smrf.create_distribution()

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
        output_variables = ["hrrr_solar"]
        config = self.base_config_copy()
        config.cfg["output"]["variables"] = output_variables
        config = cast_all_variables(config, config.mcfg)

        smrf = SMRF(config)
        smrf.topo = TOPO_MOCK

        self.assertEquals(set(output_variables), smrf.output_variables)
        self.assertEquals({}, smrf.distribute)

        smrf.create_distribution()

        self.assertIsInstance(
            smrf.distribute[distribute.SolarHRRR.DISTRIBUTION_KEY],
            distribute.SolarHRRR,
        )
        # Special case handling for HRRR Solar
        self.assertTrue(distribute.Solar.DISTRIBUTION_KEY in smrf.output_variables)
        self.assertFalse(distribute.SolarHRRR.INI_VARIABLE in smrf.output_variables)
        # Dependencies
        self.assertIsInstance(
            smrf.distribute[distribute.Precipitation.DISTRIBUTION_KEY],
            distribute.Precipitation,
        )
        self.assertIsInstance(
            smrf.distribute[distribute.Albedo.DISTRIBUTION_KEY],
            distribute.Albedo,
        )
