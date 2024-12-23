from os.path import exists

import geopandas as gpd
import numpy as np
import rasterio
from tqdm import tqdm

from ..utils.file_operations import get_files_path_from_folder
from .plot_S2_images import ImageGridS2, ImageS2


class TifImageFromS2(ImageS2):
    """Class to describe a tif image from Google Earth Engine
    extracted from Sentinel2."""

    def __init__(self, tif_path: str) -> None:
        self.tif_path = tif_path
        image, self.profile = self.read_tif()
        super().__init__(image)

    def read_tif(self) -> tuple:
        with rasterio.open(self.tif_path) as src:
            image = src.read()
            profile = src.profile
        return image, profile


class TifImageGrid(ImageGridS2):
    def __init__(self, tif_paths: list) -> None:
        self.tif_paths = tif_paths
        super().__init__(self.load_images(tif_paths))

    def load_images(self, tif_paths: list) -> list:
        images = []
        for tif_path in tqdm(tif_paths, "Importing images..."):
            image = TifImageFromS2(tif_path)
            images.append(image)
        return images


def main(
    source: str, multiple: bool, compression: int, colors: str, recursive: bool
) -> None:
    print(f"source: {source}")
    print(f"multiple: {multiple}")
    print(f"compression: {compression}")
    print(f"colors: {colors}")
    print(f"recursive: {recursive}")
    if multiple and not exists(source):
        raise NotADirectoryError(f"The source folder {source} does not exist.")

    if not multiple and not source.endswith(".tif"):
        raise FileNotFoundError(f"The source file {source} is not a tif file.")

    # Load the image(s)
    if multiple:
        image_paths = get_files_path_from_folder(
            source, extension=".tif", recursive=recursive
        )
        if len(image_paths) == 0:
            raise FileNotFoundError(f"No tif files found in folder {source}.")
        else:
            print(image_paths)
            image_grid = TifImageGrid(image_paths)
            if colors == "rgb":
                image_grid.display_grid_rgb(compression)
            elif colors == "nir":
                image_grid.display_grid_nir(compression)
            else:
                raise ValueError(f"Unknown colors {colors}.")
    else:
        print(f"Loading image {source}...")
        tif_image = TifImageFromS2(source)
        if colors == "rgb":
            tif_image.display_rgb_img(compression)
        elif colors == "nir":
            tif_image.display_nir_img(compression)
        else:
            raise ValueError(f"Unknown colors {colors}.")
