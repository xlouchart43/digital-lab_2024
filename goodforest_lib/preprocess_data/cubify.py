import h5py
import numpy as np
from tqdm import tqdm

from ..utils.file_operations import (
    get_destination_file_path,
    get_files_path_from_folder,
)
from ..utils.models import Sentinel2Dataset


def extract_cubes(file_paths: list, cube_size: int) -> list:
    """Extract cubes from images.

    Args:
        file_paths (list): List of file paths.
        cube_size (int): Cube size.

    Returns:
        list: List of extracted cubes.
    """
    processed_data = []
    dataset = Sentinel2Dataset(file_paths)

    for i in tqdm(range(len(file_paths)), "Extracting cubes from images..."):
        image, _ = dataset[i]
        cubes = dataset.get_cubes(image, cube_size)
        processed_data.append(cubes)

    return processed_data


def main(
    source_folder: str,
    destination_folder: str,
    cube_size: int,
    recursive: bool,
    filename: str,
) -> None:
    """Main function to extract cubes from images.

    Args:
        source_folder (str): Source folder.
        destination_folder (str): Destination folder.
        cube_size (int): Cube size.
        recursive (bool): Recursive search.
        filename (str): Filename.
    """
    file_paths = get_files_path_from_folder(
        source_folder, extension=".npy", recursive=recursive
    )

    processed_data = extract_cubes(file_paths, cube_size)
    cubes = np.array([cube for data in processed_data for cube in data], dtype=np.uint8)

    print(f"Total number of cubes: {len(cubes)}")

    destination_folder = get_destination_file_path(destination_folder, filename, ".h5")

    with h5py.File(destination_folder, "w") as f:
        f.create_dataset("cubes", data=cubes)
