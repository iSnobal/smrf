import logging
import os
import sys

import coloredlogs


FMT = '%(levelname)s:%(name)s:%(message)s'

# Logger that SMRF sets up when it is run on its own
SMRF_LOGGER = 'smrf'


def setup_logging(config: dict, logger_names: tuple[str, ...] = (SMRF_LOGGER,)) -> None:
    """
    Send the given loggers to log_file when it is set, otherwise to the console.
    Only these loggers are changed. Their records still propagate to the root logger.

    Args:
        config: the system section of the user config
        logger_names: names of the loggers to set up, which share one handler
    """
    log_file = config.get('log_file', None)
    if log_file is None:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(coloredlogs.ColoredFormatter(fmt=FMT))
    else:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handler = logging.FileHandler(log_file, mode='a')
        handler.setFormatter(logging.Formatter(FMT))

    for name in logger_names:
        named_logger = logging.getLogger(name)
        for previous in named_logger.handlers[:]:
            named_logger.removeHandler(previous)
            previous.close()
        named_logger.addHandler(handler)
        named_logger.setLevel(config.get('log_level', 'info').upper())
