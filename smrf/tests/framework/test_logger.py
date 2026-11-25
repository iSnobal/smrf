import logging
import os
import tempfile
import unittest

from smrf.framework import logger

# Logger names used only by these tests
NAMES = ("smrf_logger_test_a", "smrf_logger_test_b")


class TestSetupLogging(unittest.TestCase):
    """
    Tests for sending named loggers to the console or a log file.
    """

    def tearDown(self):
        for name in (*NAMES, logger.SMRF_LOGGER):
            named_logger = logging.getLogger(name)
            for handler in named_logger.handlers[:]:
                named_logger.removeHandler(handler)
                handler.close()

    def test_smrf_by_default(self):
        logger.setup_logging({})

        handlers = logging.getLogger(logger.SMRF_LOGGER).handlers
        self.assertEqual(len(handlers), 1)
        self.assertNotIsInstance(handlers[0], logging.FileHandler)

    def test_shared_handler(self):
        logger.setup_logging({}, NAMES)

        first, second = (logging.getLogger(name).handlers for name in NAMES)
        self.assertEqual(len(first), 1)
        self.assertIs(first[0], second[0])

    def test_log_file(self):
        with tempfile.TemporaryDirectory() as folder:
            log_file = os.path.join(folder, "logs", "log.out")
            logger.setup_logging({"log_file": log_file, "log_level": "debug"}, NAMES)
            for name in NAMES:
                logging.getLogger(f"{name}.child").debug("message")

            with open(log_file) as log:
                text = log.read()

        for name in NAMES:
            self.assertIn(f"DEBUG:{name}.child:message", text)

    def test_repeated_setup(self):
        logger.setup_logging({}, NAMES)
        logger.setup_logging({}, NAMES)

        for name in NAMES:
            self.assertEqual(len(logging.getLogger(name).handlers), 1)
