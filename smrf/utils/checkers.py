from inicheck.checkers import CheckType


class CheckStation(CheckType):
    """
    Custom check for ensuring our stations are always capitalized
    """

    def __init__(self, **kwargs):
        super(CheckStation, self).__init__(**kwargs)

    def type_func(self, value):
        """
        Attempt to convert all the values to upper case.

        Args:
            value: A single string in config entry representing a station name
        Returns:
            value: A single station name all upper case
        """

        return value.upper()


class CheckRawString(CheckType):
    """
    Custom `inicheck` checker that will not change the input string
    """

    def __init__(self, **kwargs):
        super(CheckRawString, self).__init__(**kwargs)

    def type_func(self, value):
        """
        Do not change the passed value at all
        Args:
            value: A single string
        Returns:
            value: A single string unchanged
        """

        return value
