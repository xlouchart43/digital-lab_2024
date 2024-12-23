import os
from logging import Logger

import geopandas as gpd

from ..logger_utils import write_message


def get_files_path_from_folder(
    folder_path: str, extension: str = None, recursive: bool = False
) -> list:
    """Return a list of path to files in a folder."""
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"{folder_path} is not a valid directory.")

    paths_to_files = []
    extension = format_extension(extension)

    if recursive:
        for root, _, files in os.walk(folder_path):
            for file in files:
                if extension is None or file.endswith(extension):
                    paths_to_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(folder_path):
            if extension is None or file.endswith(extension):
                paths_to_files.append(os.path.join(folder_path, file))

    return sorted(paths_to_files)


def get_destination_file_path(
    destination_folder_name: str,
    filename: str,
    extension: str,
    suffix: str = None,
) -> str:
    """
    Return the path to the destination file.
    If the destination folder does not exist, it will be created.
    If the filename is specified in the destination folder argument, it will be used.
    Otherwise, the filename will be used with the specified suffix and extension.
    """
    extension = format_extension(extension)
    # Check if the destination is a folder or a file
    if os.path.isfile(destination_folder_name):
        if destination_folder_name.endswith(extension):
            return destination_folder_name
        else:
            raise ValueError(
                f"The destination file must have the extension {extension}."
            )
    else:
        if filename and extension:
            return os.path.join(
                destination_folder_name,
                (
                    f"{filename}-{suffix}{extension}"
                    if suffix
                    else f"{filename}{extension}"
                ),
            )
        else:
            raise ValueError("The filename and extension must be specified.")


def get_recursive_destination_file_path(
    destination_folder: str,
    file_path: str,
    extension: str,
    suffix: str = None,
):
    """Return the path to the destination file with the same structure as the source folder."""
    temp_destination_folder = get_recursive_destination_folder_path_from_source(
        destination_folder, os.path.dirname(file_path)
    )
    destination_path = get_destination_file_path(
        temp_destination_folder,
        os.path.splitext(os.path.basename(file_path))[0],
        extension,
        suffix,
    )
    return destination_path


def get_recursive_destination_folder_path_from_source(
    destination_folder: str, source_folder: str
) -> str:
    """This function add to the destination path the relative path from the source path
    to keep the structure of the folder."""
    # Normalize the paths to handle any inconsistencies
    destination_path = os.path.normpath(destination_folder)
    source_path = os.path.normpath(source_folder)

    # Split both paths into their components
    destination_parts = destination_path.split(os.sep)
    source_parts = source_path.split(os.sep)

    # Find the first difference
    min_length = min(len(destination_parts), len(source_parts))
    divergence_index = min_length  # Default to the full length if no divergence

    for i in range(min_length):
        if destination_parts[i] != source_parts[i]:
            divergence_index = i
            break

    # Construct the final path
    final_path_parts = (
        destination_parts[:divergence_index] + source_parts[divergence_index:]
    )

    final_path = os.path.join(*final_path_parts)

    return final_path


def format_extension(extension: str) -> str:
    """Return a formatted extension starting with a dot."""
    if extension.startswith("."):
        return extension
    else:
        return f".{extension}"


def load_shp(shp_path: str, logger: Logger = None) -> gpd.GeoDataFrame:
    """Load a shapefile."""
    # Load the shapefile
    try:
        gdf = gpd.read_file(shp_path)
        write_message(f"Shapefile loaded successfully.", logger=logger, level="success")
        return gdf
    except Exception as e:
        write_message(
            f"An error occurred while loading the shapefile: {e}",
            logger=logger,
            level="error",
        )
        return None
