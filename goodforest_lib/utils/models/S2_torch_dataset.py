import os

import numpy as np
import rasterio
import torch
from geopandas import GeoDataFrame
from rasterio.features import rasterize
from torch.utils.data import Dataset

from ...config import NB_CHANNELS_WITHOUT_LABELS


class Sentinel2Dataset(Dataset):
    """
    Class to describe a dataset of Sentinel2 images containing:
    - 11 bands of the image from Sentinel 2
    - 13 vegetation indices
    - 1 mask with the labels
    """

    def __init__(self, file_paths: list) -> None:
        self.file_paths = file_paths

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> tuple:
        if os.path.splitext(self.file_paths[idx])[1] == ".tif":
            with rasterio.open(self.file_paths[idx]) as src:
                image = src.read()
                profile = src.profile

            image = torch.from_numpy(image).to(torch.uint8)
            if image.dim() == 2:
                image = image.unsqueeze(0)
            elif image.dim() == 3 and image.shape[0] != NB_CHANNELS_WITHOUT_LABELS:
                image = image.permute(2, 0, 1)
            print(image.shape)
            return image, profile
        elif os.path.splitext(self.file_paths[idx])[1] == ".npy":
            image = np.load(self.file_paths[idx])
            print(image.shape)
            return torch.from_numpy(image).to(torch.uint8), None
        else:
            raise ValueError(
                f"File format {os.path.splitext(self.file_paths[idx])[1]} not supported."
            )

    @staticmethod
    def get_labels(
        image: torch.tensor, profile: dict, polygons: GeoDataFrame
    ) -> torch.tensor:
        c, h, w = image.shape
        features = [(row["geometry"], row["labels"]) for _, row in polygons.iterrows()]
        raster = rasterize(
            features,
            out_shape=(profile["height"], profile["width"]),
            transform=profile["transform"],
            fill=0,
            all_touched=True,
            dtype=rasterio.uint8,
        )
        np_image = image.numpy()
        all_black = np.all(np_image[:11] == 0, axis=0)
        mask = np.where(all_black, 0, np.where(raster != 0, raster, 1))
        new_image = torch.zeros((c + 1, h, w), dtype=torch.uint8)
        new_image[:c, :, :] = image
        new_image[c, :, :] = torch.from_numpy(mask).to(torch.uint8)
        return new_image

    @staticmethod
    def get_cubes(image: np.array, cube_size: int) -> np.ndarray:
        _, h, w = image.shape
        return np.array(
            [
                image[:, i : i + cube_size, j : j + cube_size].numpy()
                for i in range(0, h, cube_size)
                for j in range(0, w, cube_size)
                if image[:, i : i + cube_size, j : j + cube_size].shape[1:]
                == (cube_size, cube_size)
            ],
            dtype=np.uint8,
        )
