# Re-exported because inicheck reads these names from the smrf package
from .inicheck_setup import (
    __config_changelog__,
    __config_checkers__,
    __config_header__,
    __config_titles__,
    __core_config__,
    __recipes__,
)

# Import utils so smrf.utils.checkers is loaded for inicheck, which looks it up
# in sys.modules and does not import it
from . import utils
