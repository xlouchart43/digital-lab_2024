"""
This script aims to filter the patches that are not suitable for training from a pickle file
The patches comes from a pickle files that contains the patches and their labels.
The dimension of the pickle file content is (n, CUBE_SIZE, CUBE_SIZE) where n is the number of patches.
25 is the number of bands and CUBE_SIZE is the dimension of the patch.
The NB_S2_BANDS firsts channels are the Sentinel-2 bands and the NB_VEGETATION_INDICES following are the Vegetation Indices, the last one is the label.
"""

import os
import pickle

import h5py
import numpy as np
from tqdm import tqdm

from ..utils.file_operations import get_destination_file_path
from ..utils.models.S2_cubes import get_cube_quality


def load_pickle(pickle_path: str) -> np.ndarray:
    """
    Load a pickle file as a np.array

    Args:
        pickle_path (str): Path to the pickle file.

    Returns:
        np.ndarray: The content of the pickle file.
    """
    try:
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
        print(f"✅ Pickle file loaded successfully with shape: {data.shape}")
        return data
    except FileNotFoundError as e:
        print(f"❌ The pickle file does not exist: {e}")
        return None
    except Exception as e:
        print(f"❌ An error occurred while loading the pickle file: {e}")
        return None


def prepare_cubes(data: np.ndarray, threshold: float) -> list:
    """
    Prepare the patches for training by filtering the patches that are not suitable for training.

    Args:
        data (np.ndarray): The patches.
        threshold (float): The threshold to filter the patches.

    Returns:
        list: The patches that are suitable for training.
    """
    cubes = np.empty(data.shape, dtype=np.uint8)
    i = 0
    for patch in tqdm(data, desc="Processing patches"):
        if get_cube_quality(patch, threshold):
            cubes[i] = patch
            i += 1
    cubes = cubes[:i]
    return cubes


def save_pickle(data: np.ndarray, pickle_path: str) -> None:
    """Save the a np.array to a pickle file with type UInt8.

    Args:
        data (np.ndarray): The data to save.
        pickle_path (str): The path to save the pickle file.
    """
    data = data.astype(np.uint8)
    try:
        with open(pickle_path, "wb") as f:
            pickle.dump(data, f)
        print(f"✅ Pickle file saved successfully with shape: {data.shape}")
    except Exception as e:
        print(f"❌ An error occurred while saving the pickle file: {e}")


def main(
    source_file: str, destination_path: str, threshold: float, delete_original: bool
) -> None:
    """
    Main function to filter the patches that are not suitable for training from a pickle file.

    Args:
        source_file (str): Path to the pickle file.
        destination_path (str): Path to save the filtered patches.
        threshold (float): The threshold to filter the patches.
        delete_original (bool): Delete the original pickle file.
    """
    if not os.path.isfile(destination_path):
        destination_path = get_destination_file_path(
            destination_path, "good_cubes", ".h5"
        )
    elif os.path.splitext(destination_path)[1] != ".h5":
        raise ValueError(
            f"The destination path {destination_path} must be a pickle file."
        )

    if source_file == destination_path:
        destination_path = destination_path.replace(".h5", "_filtered.h5")

    try:
        with h5py.File(source_file, "r") as f:
            data = f["cubes"][:]
        print(f"✅ H5 file loaded successfully with shape: {data.shape}")
    except FileNotFoundError as e:
        print(f"❌ The H5 file does not exist: {e}")
        return
    except Exception as e:
        print(f"❌ An error occurred while loading the H5 file: {e}")
        return

    cubes = prepare_cubes(data, threshold)
    print(f"✅ {len(cubes)} patches are suitable for training.")

    with h5py.File(destination_path, "w") as f:
        f.create_dataset("cubes", data=cubes)

    if delete_original and os.path.normpath(source_file) != os.path.normpath(
        destination_path
    ):
        os.remove(source_file)
        print(f"❌ The former pickle file {source_file} has been deleted.")
