import unittest
import numpy as np

from smrf.envphys import storms

TIME_STEP = (60 * 60)  # In seconds
TOPO = (2, 2)


class TestTimeSinceStormInitial(unittest.TestCase):
    # Cases with current storm and precip all zero
    #

    def test_no_new_precipitation(self):
        run_storm_days(
            np.full(TOPO, TIME_STEP),
            np.zeros(TOPO),
        )

    def test_precipitation_below_mass_threshold(self):
        run_storm_days(
            np.full(TOPO, TIME_STEP),
            np.zeros(TOPO),
            precip=np.full(TOPO, 0.5)
        )

    def test_precipitation_below_percent_snow(self):
        run_storm_days(
            np.full(TOPO, TIME_STEP),
            np.zeros(TOPO),
            precip=np.full(TOPO, 1.),
            percent_snow=np.full(TOPO, 0.2)
        )

    def test_precipitation_full_storm_at_thresholds(self):
        precip = np.full(TOPO, 1.)
        run_storm_days(
            np.full(TOPO, 0),
            precip,
            precip=precip,
            percent_snow=np.full(TOPO, 0.5)
        )

    def test_precipitation_full_domain_above_thresholds(self):
        precip = np.full(TOPO, 3.)
        run_storm_days(
            np.full(TOPO, 0),
            precip,
            precip=precip,
            percent_snow=np.full(TOPO, 1.)
        )

    def test_precipitation_part_domain_above_thresholds(self):
        precip = np.array([[0., 1.], [0., 2.]])
        run_storm_days(
            np.array([[TIME_STEP, 0], [TIME_STEP, 0]]),
            precip,
            precip=precip,
            percent_snow=np.full(TOPO, 1.)
        )


class TestTimeSinceStormContinued(unittest.TestCase):
    # Cases with current storm or precip have incoming values
    #
    INITIAL_STORM_DAYS = np.full(TOPO, (2 * TIME_STEP))
    INITIAL_STORM_PRECIP = np.full(TOPO, 5.)

    def test_no_new_precipitation(self):
        storm_days_expected = self.INITIAL_STORM_DAYS + TIME_STEP
        run_storm_days(
            storm_days_expected,
            np.zeros(TOPO),
            storm_days_in=self.INITIAL_STORM_DAYS,
            storm_precip_in=np.zeros(TOPO),
        )

    def test_below_threshold_precipitation(self):
        storm_days_expected = self.INITIAL_STORM_DAYS + TIME_STEP
        run_storm_days(
            storm_days_expected,
            np.zeros(TOPO),
            precip=np.full(TOPO, 0.2),
            percent_snow=np.full(TOPO, 0.5),
            storm_days_in=self.INITIAL_STORM_DAYS,
            storm_precip_in=np.zeros(TOPO),
        )

    def test_new_precipitation_storm_start(self):
        storm_precip_expected = np.full(TOPO, 2.)
        run_storm_days(
            np.zeros(TOPO),
            storm_precip_expected,
            precip=storm_precip_expected,
            percent_snow=np.full(TOPO, 0.5),
            storm_days_in=self.INITIAL_STORM_DAYS,
            storm_precip_in=np.zeros(TOPO),
        )

    def test_new_precipitation_storm_continued(self):
        new_precip = np.full(TOPO, 2.)
        run_storm_days(
            np.zeros(TOPO),
            self.INITIAL_STORM_PRECIP + new_precip,
            precip=new_precip,
            percent_snow=np.full(TOPO, 0.5),
            storm_days_in=np.zeros(TOPO),
            storm_precip_in=self.INITIAL_STORM_PRECIP,
        )

    def test_part_domain_new_storm_above_mass_threshold(self):
        storm_days_expected = self.INITIAL_STORM_DAYS + TIME_STEP
        storm_days_expected[1, 1] = 0
        new_precip = np.array([[0, 0], [0, 2.]])
        run_storm_days(
            storm_days_expected,
            new_precip,
            precip=new_precip,
            percent_snow=np.array([[0, 0], [0, 0.8]]),
            storm_days_in=self.INITIAL_STORM_DAYS,
            storm_precip_in=np.zeros(TOPO),
        )

    def test_part_domain_new_storm_below_mass_threshold(self):
        run_storm_days(
            self.INITIAL_STORM_DAYS + TIME_STEP,
            np.zeros(TOPO),
            precip=np.array([[0.1, 0], [0, 0.5]]),
            percent_snow=np.array([[0.3, 0], [0, 0.3]]),
            storm_days_in=self.INITIAL_STORM_DAYS,
            storm_precip_in=np.zeros(TOPO),
        )

    def test_part_domain_new_storm_below_percent_threshold(self):
        run_storm_days(
            self.INITIAL_STORM_DAYS + TIME_STEP,
            np.zeros(TOPO),
            precip=np.array([[0, 0], [0, 2.]]),
            percent_snow=np.array([[0, 0], [0, .2]]),
            storm_days_in=self.INITIAL_STORM_DAYS,
            storm_precip_in=np.zeros(TOPO),
        )


def run_storm_days(
    storm_days_expected,
    storm_precip_expected,
    precip=None,
    percent_snow=None,
    storm_days_in=None,
    storm_precip_in=None,
):
    """
    Args:
        storm_days_expected: Expected storm day return
        storm_precip_expected: Expected storm precip return
        precip: Incoming precipitation
        percent_snow: Incoming percent of snow for precipitation
        storm_days_in: Incoming values for storm days
        storm_precip_in: Incoming values for storm precipitation
    """
    if precip is None:
        precip = np.zeros(TOPO)
    if percent_snow is None:
        percent_snow = np.zeros(TOPO)
    if storm_days_in is None:
        storm_days_in = np.zeros(TOPO)
    if storm_precip_in is None:
        storm_precip_in = np.zeros(TOPO)

    storm_days, storm_precip = storms.time_since_storm(
        precip, percent_snow,
        storm_days_in, storm_precip_in,
        TIME_STEP
    )

    np.testing.assert_equal(
        storm_days,
        storm_days_expected,
        err_msg='Unexpected values for storm days'
    )
    np.testing.assert_equal(
        storm_precip,
        storm_precip_expected,
        err_msg='Unexpected values for storm precipitation'
    )
