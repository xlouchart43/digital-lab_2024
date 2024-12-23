import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import rasterio

from ...config.constants import NB_S2_BANDS
from ...utils.compute_additional_bands import append_band_values
from ...utils.file_operations import get_files_path_from_folder


def fetch_data_with_dates(
    source_folder: str,
    bands: list[int],
    recursive: bool = False,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray[datetime]]:
    """
    Fetch the data from the source folder.
    The files in the folder are expected to be in the format: "YYYY-MM-DD[*].tif"

    Input:
        - source_folder: the path to the folder containing the images
        - bands: the list of bands to fetch
        - recursive: whether to fetch the images recursively
        - normalize: whether to normalize the images between 0 and 1
    Output:
        - images: the images images
        - dates: the dates of the images
    """
    file_paths = get_files_path_from_folder(
        source_folder, extension=".tif", recursive=recursive
    )

    dates = list(map(lambda x: x.split(os.path.sep)[-1][:10], file_paths))
    dates = np.array(list(map(lambda x: datetime.strptime(x, "%Y-%m-%d"), dates)))

    images = []

    for file in file_paths:
        with rasterio.open(file) as src:
            image = src.read()
            image = append_band_values(image, bands)
            mask_black_px = np.where(np.all(image[:NB_S2_BANDS] == 0, axis=0))
            image[:, mask_black_px[0], mask_black_px[1]] = np.nan
            if normalize:
                image = image / 255
            images.append(image)

    try:
        images = np.array(images)
    except ValueError:
        raise ValueError("The images have different shapes")

    return images, dates


def get_historical_data_with_dates(
    images: np.ndarray,
    dates: np.ndarray[datetime],
    bands: list[int] = None,
    after_date: str = None,
    before_date: str = None,
) -> tuple[np.ndarray, np.ndarray[datetime]]:
    """
    Get the historical data from the images.

    Input:
        - images: the images with shape (n_images, n_bands, height, width)
        - dates: the dates of the images
        - bands: the list of bands to fetch
        - after_date: the date after which to fetch the data
        - before_date: the date before which to fetch the data
    Output:
        - historical_images: the historical images
        - historical_dates: the dates of the historical images
    """
    if after_date is not None:
        after_date = datetime.strptime(after_date, "%Y-%m-%d")
    if before_date is not None:
        before_date = datetime.strptime(before_date, "%Y-%m-%d")

    if bands is not None and (min(bands) < 0 or max(bands) >= images.shape[1]):
        raise ValueError("The bands are out of range")
    if bands is None:
        bands = list(range(images.shape[1]))

    historical_images = []
    historical_dates = []

    for i, date in enumerate(dates):
        if (after_date is None or date >= after_date) and (
            before_date is None or date <= before_date
        ):
            historical_dates.append(date)
            historical_images.append(images[i, bands])

    return np.array(historical_images), np.array(historical_dates)
