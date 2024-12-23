import os
import pickle
from json import loads
import pandas as pd
import numpy as np
from geopandas import GeoDataFrame
from tqdm import tqdm

from ..config.constants import BC_PEST_TREES_COMBINATION, BC_SEVERITY_MAPPING_LIST
from ..utils.file_operations import (
    get_destination_file_path,
    get_files_path_from_folder,
    get_recursive_destination_file_path,
)
from ..utils.models import Sentinel2Dataset


def map_multi_classes(x: pd.Series, nb_steps_severity: int = 4) -> int:
    """Map the multi-classes to a single class.

    Args:
        x (pd.Series): A row of a DataFrame.
        nb_steps_severity (int): Number of steps for the severity.

    Returns:
        int: The mapped class.
    """
    bc_severity_mapping = BC_SEVERITY_MAPPING_LIST[nb_steps_severity]
    if x["PEST_SEVERITY_CODE"] in ["G", "H"]:
        return 0
    if (
        x["TREE_SPECIES_CODE"] not in BC_PEST_TREES_COMBINATION
        or x["PEST_SPECIES_CODE"]
        not in BC_PEST_TREES_COMBINATION[x["TREE_SPECIES_CODE"]]
    ):
        return 2 + bc_severity_mapping[x["PEST_SEVERITY_CODE"]]
    else:
        return (
            2
            + nb_steps_severity
            + BC_PEST_TREES_COMBINATION[x["TREE_SPECIES_CODE"]][x["PEST_SPECIES_CODE"]]
            * nb_steps_severity
            + bc_severity_mapping[x["PEST_SEVERITY_CODE"]]
        )


def match_labels(
    num_classes: int, gdf: GeoDataFrame, polygons: GeoDataFrame
) -> GeoDataFrame:
    """Match the labels to the polygons.

    Args:
        num_classes (int): Number of classes.
        gdf (GeoDataFrame): Global GeoDataFrame.
        polygons (GeoDataFrame): Polygons GeoDataFrame.

    Returns:
        GeoDataFrame: The GeoDataFrame with the labels."""
    if num_classes in [2, 3, 4, 5]:
        labels = gdf.loc[:, ["PEST_SEVERITY_CODE"]]
        labels["labels"] = labels.apply(
            lambda x: (
                BC_SEVERITY_MAPPING_LIST[num_classes][x["PEST_SEVERITY_CODE"]] + 2
            ),
            axis=1,
        )
        polygons["labels"] = labels["labels"].astype("uint8")
    elif num_classes == 120:
        labels = gdf.loc[
            :, ["PEST_SPECIES_CODE", "PEST_SEVERITY_CODE", "TREE_SPECIES_CODE"]
        ]
        labels["labels"] = labels.apply(
            lambda x: map_multi_classes(x, nb_steps_severity=2),
            axis=1,
        )
        polygons["labels"] = labels["labels"].astype("uint8")
    elif num_classes == 255:
        labels = gdf.loc[
            :, ["PEST_SPECIES_CODE", "PEST_SEVERITY_CODE", "TREE_SPECIES_CODE"]
        ]
        labels["labels"] = labels.apply(
            lambda x: map_multi_classes(x, nb_steps_severity=4),
            axis=1,
        )
        polygons["labels"] = labels["labels"].astype("uint8")
    else:
        raise NotImplementedError("Only 2 to 5 and 120 and 255 classes are supported.")
    return polygons


def get_annotated_gdf_from_BC(gdf_path: str, year: int, insects: list) -> GeoDataFrame:
    """Return the annotated GeoDataFrame for the specified year and insects
    from British Columbia dataset.

    Args:
        gdf_path (str): Path to the GeoDataFrame.
        year (int): Year.
        insects (list): List of insects.

    Returns:
        GeoDataFrame: The annotated GeoDataFrame."""
    with open(gdf_path, "rb") as f:
        gdf = pickle.load(f)

    gdf = gdf[(gdf["CAPTURE_YEAR"] == year)]
    if insects:
        print("Filtering insects...")
        gdf = gdf[(gdf["PEST_SPECIES_CODE"].isin(insects))]

    return gdf


def annotate_images(file_paths: list, gdf: GeoDataFrame, num_classes: int) -> dict:
    """
    Annotate the images with the GeoDataFrame.

    Args:
        file_paths (list): List of file paths.
        gdf (GeoDataFrame): Global GeoDataFrame.
        num_classes (int): Number of classes."""
    annotated_data = dict()
    dataset = Sentinel2Dataset(file_paths)
    for i, file_path in tqdm(enumerate(file_paths), "Annotating images..."):
        image, profile = dataset[i]

        polygons = gdf["geometry"].to_frame()
        polygons = polygons.to_crs(profile["crs"])
        polygons = match_labels(num_classes, gdf, polygons)
        image_with_labels = dataset.get_labels(image, profile, polygons)

        annotated_data[file_path] = image_with_labels

    return annotated_data


def main(
    source_folder: str,
    destination_folder: str,
    gdf_path: str,
    num_classes: int,
    year: int,
    insects: list,
    recursive: bool,
    suffix: str,
) -> None:
    """Main function to annotate images.

    Args:
        source_folder (str): Source folder.
        destination_folder (str): Destination folder.
        gdf_path (str): Path to the global GeoDataFrame.
        num_classes (int): Number of classes.
        year (int): Year.
        insects (list): List of insects.
        recursive (bool): Recursive search.
        suffix (str): Suffix.
    """
    if os.path.isfile(destination_folder):
        destination_folder = os.path.dirname(destination_folder)

    file_paths = get_files_path_from_folder(source_folder, ".tif", recursive)

    gdf = get_annotated_gdf_from_BC(
        gdf_path, year, insects if num_classes not in [120, 255] else None
    )

    annotated_images = annotate_images(file_paths, gdf, num_classes)

    for file_path, image in tqdm(annotated_images.items(), "Saving images..."):
        if recursive:
            destination_path = get_recursive_destination_file_path(
                destination_folder, file_path, ".npy", suffix
            )
        else:
            destination_path = get_destination_file_path(
                destination_folder,
                file_path,
                ".npy",
                suffix,
            )

        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        np.save(destination_path, image)
