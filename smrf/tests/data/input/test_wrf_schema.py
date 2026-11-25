import unittest

from smrf.data.input.wrf_schema import FIELDS, WRF, WUS_D3, detect_input_type


class TestDetectInputType(unittest.TestCase):
    """
    Tests for naming the WRF input type from the variables in a file.
    """

    def test_wus_d3(self):
        names = tuple(FIELDS[WUS_D3].values()) + ("Times",)

        self.assertEqual(detect_input_type(names), WUS_D3)

    def test_wrf(self):
        names = tuple(FIELDS[WRF].values()) + ("Times",)

        self.assertEqual(detect_input_type(names), WRF)

    def test_both_sets_is_wus_d3(self):
        # WUS-D3 is checked first, so a file with both sets is read as WUS-D3
        names = tuple(FIELDS[WUS_D3].values()) + tuple(FIELDS[WRF].values())

        self.assertEqual(detect_input_type(names), WUS_D3)

    def test_partial_wus_d3_is_wrf(self):
        # Without LWDNB the file is not WUS-D3, so the WRF fields decide
        names = tuple(v for v in FIELDS[WUS_D3].values() if v != "LWDNB")
        names += tuple(FIELDS[WRF].values())

        self.assertEqual(detect_input_type(names), WRF)

    def test_no_matching_variables_raises(self):
        with self.assertRaisesRegex(ValueError, "matches no WRF input type"):
            detect_input_type(["Times", "XLAT", "HGT"])

    def test_neither_raises_and_lists_missing_fields(self):
        with self.assertRaises(ValueError) as context:
            detect_input_type(["Times", "T2"])

        message = str(context.exception)
        self.assertIn("matches no WRF input type", message)
        self.assertIn("Closest is wrf", message)
        self.assertIn("DWPT", message)
