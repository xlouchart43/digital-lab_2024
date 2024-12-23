# my_library/logger_utils.py
import logging
import os

from .config.constants import LOGGER_LEVEL


def get_logger(name: str, log_filename: str):
    # Create a directory for logs if it doesn't exist
    log_directory = "logs"
    os.makedirs(log_directory, exist_ok=True)

    # Full path for the log file
    log_path = os.path.join(log_directory, log_filename)

    # Set up the logger
    logger = logging.getLogger(name)
    logger.setLevel(LOGGER_LEVEL)

    # Check if logger already has handlers to avoid duplicates
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(LOGGER_LEVEL)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

        # Add handlers to the logger
        logger.addHandler(file_handler)

    return logger


def write_message(message: str, logger: logging.Logger, level: str = "info"):
    message_picto = {"success": "✅", "info": "ℹ️", "warning": "⚠️", "error": "❌"}
    message = f"{message_picto.get(level, "")} {message}"
    if logger is not None:
        if level == "info":
            logger.info(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
        else:
            logger.info(message)
    else:
        print(message)
