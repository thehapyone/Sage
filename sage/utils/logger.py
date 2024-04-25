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
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class CustomLogger(logging.Logger):
    """
    A custom logger that uses the custom formatter
    """

    def __init__(self, name: str = __name__, level: int = logging.INFO) -> None:
        """
        Args:
            name (str, optional): A custom name for the logger. Defaults to __name__.

            level (int, optional): The level of logging. Defaults to logging.INFO.
        """
        super().__init__(name, level)

        # create console handler
        self._log_handler = logging.StreamHandler()
        self._log_handler.setLevel(level)

        self._log_handler.setFormatter(CustomFormatter())

        # customize the logger
        self.name = name
        self.setLevel(level)
        self.addHandler(self._log_handler)

        # Get the root logger and attach handler to it
        root_logger = logging.getLogger()
        root_logger.handlers = self.handlers
        root_logger.setLevel(logging.WARNING)

    def setLevel(self, level) -> None:
        self._log_handler.setLevel(level)
        super().setLevel(level)
