import os

from inicheck.checkers import CheckCriticalFilename, CheckType


class CheckStation(CheckType):
    """
    Custom check for ensuring our stations are always capitalized
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
        super().__init__(**kwargs)

    def type_func(self, value):
        """
        Do not change the passed value at all
        Args:
            value: A single string
        Returns:
            value: A single string unchanged
        """

        return value


class CheckCriticalFileOrDirectory(CheckCriticalFilename):
    """
    Custom `inicheck` checker when needing a file or a folder.
    Example: WRF input
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.message = "Path is neither a file nor a directory."

    def is_valid(self, value):
        """
        Check that the value exists as a file or as a directory.

        Args:
            value: path to a file or directory
        Returns:
            tuple: whether the value is valid, and the message if it is not
        """
        path = self.make_abs_from_cfg(value)
        if os.path.isfile(path) or os.path.isdir(path):
            return True, None
        return False, self.message
