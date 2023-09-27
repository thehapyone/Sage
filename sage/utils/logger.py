import logging

class CustomFormatter(logging.Formatter):

    grey = "\x1b[42;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[41;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: bold_red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class CustomLogger(logging.Logger):
    """
    A custom logger that uses the custom formatter
    """

    def __init__(self, name: str = __name__, level: int = logging.DEBUG) -> None:
        """
        Args:
            name (str, optional): A custom name for the logger. Defaults to __name__.

            level (int, optional): The level of logging. Defaults to logging.DEBUG.
        """
        super().__init__(name, level)

        # create console handler
        log_handler = logging.StreamHandler()
        log_handler.setLevel(level)

        log_handler.setFormatter(CustomFormatter())

        # customize the logger
        self.name = name
        self.setLevel(level)
        self.addHandler(log_handler)
