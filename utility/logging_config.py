import logging
from colorlog import ColoredFormatter

def configure_logging(level=logging.INFO):
    """ Configure the logging module to use colored output. """
    
    log_format = (
        "%(log_color)s%(levelname)-8s%(reset)s "
        "%(log_color)s%(message)s%(reset)s"
    )

    formatter = ColoredFormatter(
        log_format,
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG':    'bold_cyan',
            'INFO':     'bold_blue',
            'WARNING':  'bold_yellow',
            'ERROR':    'red',
            'CRITICAL': 'bold_red'
        }
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    # Clear any existing handlers
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(level)
    logger.addHandler(handler)