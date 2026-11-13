import logging
import os
import sys

import coloredlogs


FMT = '%(levelname)s:%(name)s:%(message)s'


def setup_logging(config: dict) -> None:
    """
    Send the smrf logger to log_file when it is set, otherwise to the console.
    Only the smrf logger is changed. Its records still propagate to the root logger.

    Args:
        config: the system section of the user config
    """
    smrf_logger = logging.getLogger('smrf')
    for handler in smrf_logger.handlers[:]:
        smrf_logger.removeHandler(handler)
        handler.close()

    log_file = config.get('log_file', None)
    if log_file is None:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(coloredlogs.ColoredFormatter(fmt=FMT))
    else:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handler = logging.FileHandler(log_file, mode='a')
        handler.setFormatter(logging.Formatter(FMT))

    smrf_logger.addHandler(handler)
    smrf_logger.setLevel(config.get('log_level', 'info').upper())
