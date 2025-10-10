from .image_data import ImageData
from smrf.utils import utils


class ta(ImageData):
    """
    The :mod:`~smrf.distribute.air_temp.ta` class allows for variable specific
    distributions that go beyond the base class.

    Air temperature is a relatively simple variable to distribute as it does
    not rely on any other variables, but has many variables that depend on it.
    Air temperature typically has a negative trend with elevation and performs
    best when detrended. However, even with a negative trend, it is possible to
    have instances where the trend does not apply, for example a temperature
    inversion or cold air pooling.  These types of conditions will have
    unintended consequences on variables that use the distributed air
    temperature.

    Args:
        taConfig: The [air_temp] section of the configuration file

    Attributes:
        config: configuration from [air_temp] section
        air_temp: numpy array of the air temperature
        stations: stations to be used in alphabetical order
    """

    variable = 'air_temp'

    # these are variables that can be output
    OUTPUT_VARIABLES = {
        'air_temp': {
            'units': 'degree_Celsius',
            'standard_name': 'air_temperature',
            'long_name': 'Air temperature'
        }
    }

    def __init__(self, taConfig):
        # extend the base class
        super().__init__(self.variable)
        # check and assign the configuration
        self.getConfig(taConfig)
        self._logger.debug('Created distribute.air_temp')

    def initialize(self, topo, data, date_time=None):
        """
        Initialize the distribution, solely calls
        :mod:`smrf.distribute.ImageData._initialize`.

        Args:
            topo: :mod:`smrf.data.loadTopo.Topo` instance
            data:
            date_time:
        """

        self._logger.debug('Initializing distribute.air_temp')
        self.date_time = date_time
        self._initialize(topo, data.metadata)

    def distribute(self, data):
        """
        Distribute air temperature given a Panda's dataframe for a single time
        step. Calls :mod:`smrf.distribute.image_data.image_data._distribute`.

        Args:
            data: Pandas dataframe for a single time step from air_temp

        """

        self._logger.debug('{} Distributing air_temp'.format(data.name))

        self._distribute(data)
        self.air_temp = utils.set_min_max(self.air_temp, self.min, self.max)
