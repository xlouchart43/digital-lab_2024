"""
This script aims to load Sentinel-2 data from a given shapefile.
The shapefile contains polygons or points that represent the area of interest.
"""

import os
from datetime import datetime

import numpy as np
import rasterio
from ee import Geometry as eeGeometry
from ee import Image

from ..config.constants import (
    BUFFER_PREDICTION,
    CLEAR_CREDENTIALS,
    EE_PROJECT_NAME,
    GCS_BUCKET_NAME,
    PERIOD_S2_IMAGES,
    SAVING_LOCATION,
)
from ..utils.connect_ee import establish_connection_ee
from ..utils.ee_api import (
    compute_vegetation_indices,
    convert_bands_to_uint8,
    get_latest_S2_images,
    save_img,
)
from ..utils.file_operations import get_files_path_from_folder
from ..utils.manage_geometry import extract_aoi_from_shp
from ..utils.models import PredictionParams


def process_S2_image(
    img: Image,
    cloud_mask: Image,
    aoi_index: int,
    aoi: eeGeometry,
    prediction_params: PredictionParams,
    shp_parent_folder: str,
) -> None:
    """Process the Sentinel-2 image and save it to the disk.
    The image is saved in the folder shp_parent_folder/aoi_{aoi_index}/tile_id.

    Args:
        img (Image): The Sentinel-2 image.
        cloud_mask (Image): The cloud mask.
        aoi_index (int): The index of the area of interest.
        aoi (eeGeometry): The area of interest.
        prediction_params (PredictionParams): The parameters for the prediction.
        shp_parent_folder (str): The parent folder of the shapefile.
    """

    # Get useful information
    tile_id = img.get("MGRS_TILE").getInfo()
    date = img.get("system:time_start").getInfo()
    date_str = datetime.fromtimestamp(date / 1000).strftime("%Y-%m-%d")
    print(f"Processing image {date_str} for tile {tile_id}.")
    img = compute_vegetation_indices(img)
    img = convert_bands_to_uint8(img, aoi)

    # Save the image
    destination_folder = os.path.join(shp_parent_folder, f"aoi_{aoi_index}", tile_id)
    filename = f"{date_str}"
    save_img(img, destination_folder, filename, aoi, prediction_params)

    # Save the cloud mask
    cloud_mask_filename = f"{filename}_cloud-mask"
    save_img(
        cloud_mask, destination_folder, cloud_mask_filename, aoi, prediction_params
    )

    # Convert the cloud_mask tiff to a numpy array
    with rasterio.open(
        os.path.join(destination_folder, f"{cloud_mask_filename}.tif")
    ) as src:
        cloud_mask_array = src.read()[0].astype(np.uint8)
    np.save(
        os.path.join(destination_folder, f"{cloud_mask_filename}.npy"), cloud_mask_array
    )
    # Remove the tiff file
    os.remove(os.path.join(destination_folder, f"{cloud_mask_filename}.tif"))


def load_S2_image_from_aoi(
    aoi_index: int,
    aoi: eeGeometry,
    prediction_params: PredictionParams,
    shp_parent_folder: str,
) -> None:
    """Load the latest Sentinel-2 images from the area of interest.

    Args:
        aoi_index (int): The index of the area of interest.
        aoi (eeGeometry): The area of interest.
        prediction_params (PredictionParams): The parameters for the prediction.
        shp_parent_folder (str): The parent folder of the shapefile.
    """
    s2_img, s2_cloud_masks = get_latest_S2_images(
        aoi, prediction_params, prediction_params.filter_black_images
    )
    for img, cloud_mask in zip(s2_img, s2_cloud_masks):
        process_S2_image(
            img, cloud_mask, aoi_index, aoi, prediction_params, shp_parent_folder
        )


def process_shapefile(shp_file: str, prediction_params: PredictionParams) -> None:
    """Acquire the area of interest from the shapefile.

    Args:
        shp_file (str): The path to the shapefile.
        prediction_params (PredictionParams): The parameters for the prediction.
    """
    aoi_list = extract_aoi_from_shp(shp_file, prediction_params.buffer)
    aoi_list = [aoi_list[i][1] for i in range(len(aoi_list))]

    for aoi_index, aoi in enumerate(aoi_list):
        shp_parent_folder = os.path.dirname(shp_file)
        load_S2_image_from_aoi(aoi_index, aoi, prediction_params, shp_parent_folder)


def main(
    source_folder: str,
    before_date: str,
    max_date_diff_in_days: int = PERIOD_S2_IMAGES,
    ee_project_name: str = EE_PROJECT_NAME,
    saving_location: str = SAVING_LOCATION,
    gcs_bucket_name: str = GCS_BUCKET_NAME,
    clear_credentials: bool = CLEAR_CREDENTIALS,
    buffer: float = BUFFER_PREDICTION,
    filter_black_images: bool = True,
) -> None:
    """Main function to load Sentinel-2 images from a shapefile.
    The function processes all the shapefiles in the source_folder.

    Args:
        source_folder (str): The folder containing the shapefiles.
        before_date (str): The date before which the images are selected.
        max_date_diff_in_days (int, optional): The maximum date difference in days. Defaults to PERIOD_S2_IMAGES.
        ee_project_name (str, optional): The Earth Engine project name. Defaults to EE_PROJECT_NAME.
        saving_location (str, optional): The location where the images are saved. Defaults to SAVING_LOCATION.
        gcs_bucket_name (str, optional): The name of the Google Cloud Storage bucket. Defaults to GCS_BUCKET_NAME.
        clear_credentials (bool, optional): Whether to clear the credentials. Defaults to CLEAR_CREDENTIALS.
        buffer (float, optional): The buffer around the area of interest. Defaults to BUFFER_PREDICTION.
        filter_black_images (bool, optional): Whether to filter out black images. Defaults to True.
    """
    prediction_params = PredictionParams(
        source_folder,
        before_date,
        max_date_diff_in_days,
        ee_project_name,
        saving_location,
        gcs_bucket_name,
        buffer,
        filter_black_images,
    )

    # Launch the main function
    establish_connection_ee(ee_project_name, clear_credentials)

    # Compute predictions for all the shapefiles in the folder
    shapefiles_to_process = get_files_path_from_folder(
        source_folder, ".shp", recursive=False
    )

    for shp_file in shapefiles_to_process:
        process_shapefile(shp_file, prediction_params)
