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
    FORDEAD_PERIOD_S2_IMAGES,
    SAVING_LOCATION,
)
from ..utils.connect_ee import establish_connection_ee
from ..utils.ee_api import (
    get_S2_images_fordead,
    save_img,
)
from ..utils.file_operations import get_files_path_from_folder
from ..utils.manage_geometry import extract_aoi_from_shp
from ..utils.models import FordeadParams
from ..config.sentinel2_bands import BAND_NAMES


def process_S2_image(
    img: Image,
    aoi: eeGeometry,
    fordead_params: FordeadParams,
) -> None:
    # Get useful information
    tile_id = img.get("MGRS_TILE").getInfo()
    date = img.get("system:time_start").getInfo()
    date_str = datetime.fromtimestamp(date / 1000).strftime("%Y-%m-%d")
    print(f"Processing image {date_str} for tile {tile_id}.")

    destination_folder = os.path.join(
        fordead_params.destination_folder, f"{date_str}_{tile_id}"
    )

    # Save the image's bands

    for band in BAND_NAMES:
        img_band = img.select(band)
        filename = f"SENTINEL2A_{tile_id}_{date_str}_{band}"
        save_img(
            img_band, destination_folder, filename, aoi, fordead_params, dtype="int16"
        )


def load_S2_image_from_aoi(
    aoi: eeGeometry,
    fordead_params: FordeadParams,
) -> None:
    s2_img = get_S2_images_fordead(
        aoi,
        fordead_params,
        fordead_params.filter_black_images,
        fordead_params.threshold,
    )
    for img in s2_img:
        process_S2_image(
            img,
            aoi,
            fordead_params,
        )


def process_shapefile(shp_file: str, fordead_params: FordeadParams) -> None:
    """Acquire the area of interest from the shapefile."""
    aoi_list = extract_aoi_from_shp(shp_file, fordead_params.buffer)
    aoi_list = [aoi_list[i][1] for i in range(len(aoi_list))]

    for aoi in aoi_list:
        load_S2_image_from_aoi(
            aoi,
            fordead_params,
        )


def main(
    source_folder: str,
    destination_folder: str,
    before_date: str,
    date_diff_in_days: int = FORDEAD_PERIOD_S2_IMAGES,
    ee_project_name: str = EE_PROJECT_NAME,
    saving_location: str = SAVING_LOCATION,
    gcs_bucket_name: str = GCS_BUCKET_NAME,
    clear_credentials: bool = CLEAR_CREDENTIALS,
    buffer: float = BUFFER_PREDICTION,
    filter_black_images: bool = True,
    threshold: float = 0.4,
):
    fordead_params = FordeadParams(
        source_folder,
        destination_folder,
        before_date,
        date_diff_in_days,
        ee_project_name,
        saving_location,
        gcs_bucket_name,
        buffer,
        filter_black_images,
        threshold,
    )

    # Launch the main function
    establish_connection_ee(ee_project_name, clear_credentials)

    # Compute predictions for all the shapefiles in the folder
    shapefiles_to_process = get_files_path_from_folder(
        source_folder, ".shp", recursive=True
    )

    for shp_file in shapefiles_to_process:
        process_shapefile(shp_file, fordead_params)
