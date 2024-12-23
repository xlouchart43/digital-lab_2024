import os
from datetime import datetime, timedelta
from logging import Logger

import ee
from ee import Filter, Geometry, Image, ImageCollection, batch
from geemap import download_ee_image

from ..config.constants import CUBE_SIZE
from ..config.sentinel2_bands import BANDS
from ..config.vegetation_indices import VEGETATION_INDICES
from ..logger_utils import write_message
from .models import EERequestParams, FordeadParams, PredictionParams, TrainingParams


def vegetation_bare_soil_mask(image: Image) -> Image:
    """Create a vegetation mask from the Sentinel-2 image."""
    scl = image.select("SCL")
    mask = scl.eq(4).Or(scl.eq(5))
    return image.updateMask(mask)


def get_cloud_mask(image: Image) -> Image:
    """Return a 1 channel 2D image with 1 where there is cloud and 0 elsewhere."""
    scl = image.select("SCL")
    mask = scl.eq(3).Or(scl.eq(7)).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10))
    return mask.rename("cloud_mask")


def fetch_image_collection(
    aoi: Geometry,
    tile_id: str,
    start_date: str,
    end_date: str,
    max_cloud_cover: int,
    nb_max_images: int,
) -> ImageCollection:
    img_collection = (
        ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_cover))
        .limit(nb_max_images)
        .map(vegetation_bare_soil_mask)
    )
    if tile_id:
        img_collection = img_collection.filterMetadata("MGRS_TILE", "equals", tile_id)
    return img_collection


def get_image_collection_with_min_cloud_cover(
    aoi: Geometry,
    training_params: TrainingParams,
    nb_min_images: int,
    nb_max_images: int,
    tile_id: str = None,
    logger: Logger = None,
) -> ImageCollection:
    if nb_min_images > nb_max_images:
        error_message = "The number of minimum images should be less than or equal to the number of maximum images."
        write_message(error_message, logger, "error")
        raise ValueError(error_message)
    s2_images = None
    for cloud_cover in range(
        0, training_params.max_cloud_cover + 1, training_params.step_cloud_cover
    ):
        s2_images = fetch_image_collection(
            aoi,
            tile_id,
            training_params.start_date,
            training_params.end_date,
            cloud_cover,
            nb_max_images,
        )
        if (
            s2_images.size().getInfo() >= nb_min_images
            or cloud_cover >= training_params.max_cloud_cover
        ):
            write_message(
                f"Cloud cover: {cloud_cover} - Images found: {s2_images.size().getInfo()}",
                logger,
                "success",
            )
            return s2_images
        else:
            write_message(
                f"Not enough images found with a cloud cover below {cloud_cover}%",
                logger,
                "warning",
            )
    return s2_images


def get_latest_S2_images(
    aoi: Geometry, prediction_params: PredictionParams, filter_black_images: bool = True
) -> tuple[list[Image], list[Image]]:

    nb_max_images = 10

    after_date = (
        datetime.strptime(prediction_params.before_date, "%Y-%m-%d")
        - timedelta(days=prediction_params.max_date_diff_in_days)
    ).strftime("%Y-%m-%d")

    s2_collection = (
        ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(after_date, prediction_params.before_date)
        .sort("system:time_start", False)
        .limit(nb_max_images)
    )

    images_list = []
    cloud_masks_list = []
    s2_list = s2_collection.toList(nb_max_images)
    for i in range(s2_list.size().getInfo()):
        image = Image(s2_list.get(i)).clip(aoi)
        if image.get("MGRS_TILE").getInfo() not in prediction_params.visited_tiles:
            # Display number of pixels
            cloud_mask = get_cloud_mask(image)
            cloud_masks_list.append(cloud_mask)

            image_cleaned = vegetation_bare_soil_mask(image)
            if filter_black_images:
                total_pixels = (
                    image.select(BANDS[0].name)
                    .reduceRegion(
                        reducer=ee.Reducer.count(), geometry=image.geometry(), scale=30
                    )
                    .values()
                    .get(0)
                    .getInfo()
                )
                print(f"Total pixels: {total_pixels}")
                if not filter_image(image_cleaned, total_pixels, threshold=0.8):
                    print(f"Image {i} is too dark, skipping...")
                    continue
            images_list.append(image_cleaned)

            prediction_params.add_tile_to_visited(image.get("MGRS_TILE").getInfo())

    return images_list, cloud_masks_list


def filter_image(image: Image, total_pixels: int, threshold: float) -> bool:
    """Pixel with to much black pixels are not considered."""
    # Count only black pixels on all bands
    not_black_pixels = (
        image.select(BANDS[0].name)
        .reduceRegion(reducer=ee.Reducer.count(), geometry=image.geometry(), scale=30)
        .values()
        .get(0)
        .getInfo()
    )
    print(f"Not black pixels: {not_black_pixels}")
    print(f"Ratio: {1 - not_black_pixels / total_pixels:.2f}%")
    return 1 - not_black_pixels / total_pixels <= threshold


def convert_bands_to_uint8(image: Image, aoi_to_save: Geometry) -> Image:
    final_image = image.clip(aoi_to_save)
    # Instead of making a linear compression, from 0-10000 to 0-255 on all bands
    # we use the percentiles of each band to apply a specific linear compression on each band
    for band in BANDS:
        final_image = final_image.addBands(
            final_image.select(band.name)
            .expression(band.normalize_expression_ee())
            .rename(f"{band.name}_norm")
        )
    # Select the normalized bands and the vegetation indices
    final_image = final_image.select(
        [f"{band.name}_norm" for band in BANDS]
        + [index.name for index in VEGETATION_INDICES]
    )
    return final_image


def compute_vegetation_indices(image: Image) -> Image:
    """Compute vegetation indices from the Sentinel-2 image with bands in range 0-1e4."""
    for vegetation_index in VEGETATION_INDICES:
        image = image.addBands(
            image.expression(vegetation_index.computation_expression_ee()).rename(
                vegetation_index.name
            )
        )
    return image


def save_image_GCS(
    image: Image,
    gcs_bucket_name: str,
    destination_folder: str,
    filename: str,
    aoi: Geometry,
    max_size: int = 16 * CUBE_SIZE,
    logger: Logger = None,
) -> None:
    """Save an image from Earth Engine on Google Cloud Storage.
    max_size parameter is the maximum size of the image in pixels (if the image size is bigger, it's cut in fewer pieces).
    """
    try:
        task = batch.Export.image.toCloudStorage(
            image=image,
            description=filename,
            bucket=gcs_bucket_name,
            fileNamePrefix=os.path.join(destination_folder, filename),
            region=aoi,
            scale=10,
            crs="EPSG:4326",
            fileFormat="Geotif",
            shardSize=CUBE_SIZE,  # Size of tiles
            fileDimensions=max_size,  # Size of the resulting image
            maxPixels=2e8,
            formatOptions={"cloudOptimized": True},
        )
        task.start()
        write_message(
            f"Image {filename} saved on Google Cloud Storage: {os.path.join(gcs_bucket_name, destination_folder, filename)}",
            logger,
            "success",
        )
    except Exception as e:
        write_message(
            f"An error occurred while saving the image on Google Cloud Storage: {e}",
            logger,
            "error",
        )


def save_image_locally(
    image: Image,
    destination_folder: str,
    filename: str,
    aoi: Geometry,
    extension: str = "tif",
    dtype: str = "uint8",
    logger: Logger = None,
) -> None:
    """Save an image locally."""
    try:
        os.makedirs(destination_folder, exist_ok=True)
        image_path = os.path.join(destination_folder, f"{filename}.{extension}")
        download_ee_image(
            image, image_path, scale=10, region=aoi, dtype=dtype, crs="EPSG:4326"
        )
    except Exception as e:
        write_message(
            f"An error occurred while saving the image locally: {e}", logger, "error"
        )


def save_img(
    img: Image,
    destination_folder: str,
    filename: str,
    aoi: Geometry,
    ee_request_params: EERequestParams,
    dtype: str = "uint8",
    logger: Logger = None,
) -> None:
    if ee_request_params.saving_location == "gcs":
        save_image_GCS(
            img,
            ee_request_params.gcs_bucket_name,
            destination_folder,
            filename,
            aoi,
            logger=logger,
        )
    elif ee_request_params.saving_location == "local":
        save_image_locally(
            img,
            destination_folder,
            filename,
            aoi,
            dtype=dtype,
            logger=logger,
        )
    else:
        error_message = "The saving location must be either 'local' or 'gcs'."
        write_message(error_message, logger, "error")
        raise ValueError(error_message)


def get_S2_images_fordead(
    aoi: Geometry,
    fordead_params: FordeadParams,
    filter_black_images: bool = True,
    threshold: float = 0.4,
) -> tuple[list[Image], list[Image]]:

    after_date = (
        datetime.strptime(fordead_params.before_date, "%Y-%m-%d")
        + timedelta(days=fordead_params.date_diff_in_days)
    ).strftime("%Y-%m-%d")

    s2_collection = (
        ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(fordead_params.before_date, after_date)
        .sort("system:time_start", True)
    )
    size = s2_collection.size().getInfo()
    images_false_list = s2_collection.toList(size)
    images_list = []
    for i in range(size):
        image = Image(images_false_list.get(i)).clip(aoi)
        image_cleaned = vegetation_bare_soil_mask(image)
        if filter_black_images:
            total_pixels = (
                image.select(BANDS[0].name)
                .reduceRegion(
                    reducer=ee.Reducer.count(), geometry=image.geometry(), scale=30
                )
                .values()
                .get(0)
                .getInfo()
            )
            if not filter_image(image_cleaned, total_pixels, threshold=threshold):
                print(f"Image {i} is too dark, skipping...")
                continue
        images_list.append(image_cleaned)
    return images_list
