import os

from smrf.tests.smrf_test_case import SMRFTestCase


class SMRFTestCaseLakes(SMRFTestCase):
    """
    Runs the short simulation over Lakes.
    """
    BBOX = [-119.13778957, 37.4541464, -118.85206348, 37.73084705]

    basin_dir = SMRFTestCase.test_dir.joinpath('basins', 'Lakes')
    config_file = os.path.join(basin_dir, SMRFTestCase.BASE_INI_FILE_NAME)
    gold_dir = basin_dir.joinpath('gold_hrrr')
    input_dir = basin_dir.joinpath('input')
