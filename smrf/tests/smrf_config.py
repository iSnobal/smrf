class SMRFConfig():
    """
    Mixin of test helper methods
    """
    @staticmethod
    def _copy_config(config: dict):
        """
        Method to copy a 2-level deep dictionary of dictionaries

        Args:
            config: Dictionary to copy

        Returns:
            Deep copy of given dictionary
        """
        return {k: v.copy() for k, v in config.items()}
