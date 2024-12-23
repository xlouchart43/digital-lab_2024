"""
This script aims to load Sentinel-2 data from a given shapefile.
The shapefile contains polygons or points that represent the area of interest.
"""

import os
from datetime import datetime
from logging import Logger

from ee import Image
from tqdm import tqdm

from ..logger_utils import get_logger, write_message
from ..utils.connect_ee import establish_connection_ee
from ..utils.ee_api import (
    compute_vegetation_indices,
    convert_bands_to_uint8,
    get_image_collection_with_min_cloud_cover,
    save_img,
)
from ..utils.manage_geometry import extract_aoi_from_shp
from ..utils.models.ee_requests_params import TrainingParams


def load_S2_images_from_aois(
    aoi_list: list, training_params: TrainingParams, logger: Logger = None
) -> None:
    """Load the Sentinel-2 images from the area of interest list.
    The images are saved in the destination folder.

    Args:
        aoi_list (list): The list of areas of interest.
        training_params (TrainingParams): The parameters for the training.
    """
    try:
        for i, full_aoi in tqdm(enumerate(aoi_list)):
            if full_aoi is None:
                continue

            name, aoi = full_aoi
            tile_id = None
            if not "aoi" in name:
                tile_id = name

            s2_images = get_image_collection_with_min_cloud_cover(
                aoi, training_params, 1, 1, tile_id, logger=logger
            )

            s2_size = s2_images.size()
            s2_list = s2_images.toList(s2_size)
            write_message(
                f"Images found in the Sentinel-2 collection: {s2_size.getInfo()}",
                logger,
                "success",
            )

            aoi_to_save = aoi

            final_images = []
            final_dates = []
            # Select the bands and clip the image
            for j in range(s2_size.getInfo()):
                cur_image = Image(s2_list.get(j))

                date = cur_image.get("system:time_start").getInfo()
                date_str = datetime.fromtimestamp(date / 1000).strftime("%Y-%m-%d")
                final_dates.append(date_str)

                cur_image_with_vegetation_indices = compute_vegetation_indices(
                    cur_image
                )
                final_image = convert_bands_to_uint8(
                    cur_image_with_vegetation_indices, aoi_to_save
                )
                final_images.append(final_image.uint8())

            # Save the images
            for j, final_image in enumerate(final_images):
                save_img(
                    final_image,
                    training_params.destination_folder,
                    f"image_tile{i}_{final_dates[j]}",
                    aoi_to_save,
                    training_params,
                    logger=logger,
                )
    except Exception as e:
        write_message(
            f"An error occurred while loading the Sentinel-2 images: {e}",
            logger,
            "error",
        )


def main(
    shp_file_path: str,
    aoi_indices: list[int],
    start_date: str,
    end_date: str,
    ee_project_name: str,
    saving_location: str,
    destination_folder: str,
    gcs_bucket_name: str,
    clear_credentials: bool,
    custom_buffer: int,
    max_cloud_cover: int,
    step_cloud_cover: int,
) -> None:
    """Main function to load the Sentinel-2 images from the area of interest.

    Args:
        shp_file_path (str): The path to the shapefile.
        aoi_indices (list[int]): The indices of the areas of interest to process.
        start_date (str): The start date for the Sentinel-2 images.
        end_date (str): The end date for the Sentinel-2 images.
        ee_project_name (str): The Earth Engine project name.
        saving_location (str): The saving location.
        destination_folder (str): The destination folder.
        gcs_bucket_name (str): The Google Cloud Storage bucket name.
        clear_credentials (bool): Whether to clear the credentials.
        custom_buffer (int): The custom buffer for the area of interest.
        max_cloud_cover (int): The maximum cloud cover.
        step_cloud_cover (int): The step cloud cover.
    """
    # Initialize the logger with a name using current date
    logger = get_logger(
        __name__, f"load_S2_train-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    )

    training_params = TrainingParams(
        ee_project_name=ee_project_name,
        saving_location=saving_location,
        gcs_bucket_name=gcs_bucket_name,
        start_date=start_date,
        end_date=end_date,
        destination_folder=destination_folder,
        max_cloud_cover=max_cloud_cover,
        step_cloud_cover=step_cloud_cover,
        logger=logger,
    )

    # Check the arguments
    if not os.path.exists(shp_file_path):
        error_message = f"The shapefile does not exist: {shp_file_path}"
        write_message(error_message, logger, "error")
        raise FileNotFoundError(error_message)

    # Launch the main function
    establish_connection_ee(
        project_name=ee_project_name,
        clear_credentials=clear_credentials,
        logger=logger,
    )

    # Extract the area of interest from the shapefile
    aoi_list = extract_aoi_from_shp(
        shp_file=shp_file_path, custom_buffer=custom_buffer, logger=logger
    )

    if aoi_indices:
        aoi_list = [
            aoi_list[i] if i in aoi_indices else None for i in range(len(aoi_list))
        ]

    # Load the Sentinel-2 images
    load_S2_images_from_aois(aoi_list, training_params, logger=logger)
