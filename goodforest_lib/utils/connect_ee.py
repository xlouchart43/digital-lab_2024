"""
This script aims to connect to Google Earth Engine.
The script will authenticate Earth Engine and initialize the connection.
The script will also clear existing credentials if specified.
The script will be used by other scripts to connect to Google Earth Engine.
"""

from logging import Logger
from os import path
from shutil import rmtree

from ee import Authenticate, Initialize

from ..logger_utils import write_message


def authenticate_gee(project_name: str, logger: Logger = None) -> None:
    """Authenticate Google Earth Engine."""
    try:

        write_message("Authenticating Earth Engine...", logger, "info")
        Authenticate()
        Initialize(project=project_name)
        write_message("Successfully authenticated Earth Engine!", logger, "success")
    except Exception as e:
        error_message = f"An error occurred while authenticating Earth Engine: {e}"
        write_message(error_message, logger, "error")
        raise Exception(error_message)


def establish_connection_ee(
    project_name: str = "gee-lab-2024",
    clear_credentials: bool = False,
    logger: Logger = None,
) -> None:
    """Establish connection to Google Earth Engine."""
    # Clear existing credentials if specified
    credentials_dir = path.join(path.expanduser("~"), ".config", "earthengine")
    if path.exists(credentials_dir) and clear_credentials:
        try:
            rmtree(credentials_dir)
            write_message(
                f"Existing Earth Engine credentials removed: {credentials_dir}",
                logger,
                "info",
            )
        except PermissionError:
            write_message(
                f"Unable to remove credentials due to permission error. Please manually delete the directory: {credentials_dir}",
                logger,
                "warning",
            )
        except Exception as e:
            write_message(
                f"An error occurred while removing existing credentials: {e}",
                logger,
                "error",
            )

    # Authenticate Earth Engine
    authenticate_gee(project_name, logger)
