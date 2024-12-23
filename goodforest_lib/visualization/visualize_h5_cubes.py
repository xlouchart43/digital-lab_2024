from os.path import exists

import h5py
import numpy as np

from .plot_S2_images import ImageGridS2


class H5CubesFromS2:
    def __init__(
        self, h5_path: str, nb_per_grid: int = 9, compression: int = None
    ) -> None:
        self.h5_path = h5_path
        self.data = self.load_h5()
        self.nb_cubes = self.data.shape[0]
        self.nb_per_grid = nb_per_grid
        self.img_grids = self.create_img_grids()
        self.compression = compression

    def load_h5(self) -> np.ndarray:
        if not exists(self.h5_path):
            raise FileNotFoundError(f"The pickle file {self.h5} does not exist.")

        with h5py.File(self.h5_path, "r") as h5_file:
            data = h5_file["cubes"][:]
        return data

    def create_img_grids(self) -> list:
        img_grids = []
        for i in range(0, self.nb_cubes, self.nb_per_grid):
            img_grids.append(
                ImageGridS2([cube for cube in self.data[i : i + self.nb_per_grid]])
            )
        if self.nb_cubes % self.nb_per_grid != 0:
            img_grids.append(
                ImageGridS2(
                    [cube for cube in self.data[-(self.nb_cubes % self.nb_per_grid) :]]
                )
            )
        return img_grids

    def display_grids_rgb(self) -> None:
        for i, img_grid in enumerate(self.img_grids):
            img_grid.display_grid_rgb(
                self.compression, f"Grid {i}/{len(self.img_grids)}"
            )

    def display_grids_nir(self) -> None:
        for i, img in enumerate(self.img_grids):
            img.display_grid_nir(self.compression, f"Grid {i}/{len(self.img_grids)}")


def main(source_file: str, nb_per_grid: int, compression: int) -> None:
    cubes = H5CubesFromS2(source_file, nb_per_grid, compression)
    cubes.display_grids_rgb()
    cubes.display_grids_nir()
