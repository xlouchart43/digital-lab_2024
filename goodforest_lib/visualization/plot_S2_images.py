import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class ImageS2:
    def __init__(self, image: np.array) -> None:
        self.image = image

    def get_rgb_array(self, resize: int = None) -> np.ndarray:
        """Get the RGB bands of the image.
        If the image is larger than the specified width, it will be resized."""
        rgb = np.stack([self.image[3], self.image[2], self.image[1]], axis=0)
        rgb = np.moveaxis(rgb, 0, -1)
        if resize is not None:
            rgb = self.resize_img(rgb, resize)
        rgb = self.enhance_colors(rgb)
        return rgb

    def get_nir_array(self, resize: int = None) -> np.ndarray:
        """Get the NIR band of the image.
        If the image is larger than the specified width, it will be resized."""
        nir = self.image[7]
        nir = np.stack([nir, np.zeros_like(nir), np.zeros_like(nir)], axis=0)
        nir = np.moveaxis(nir, 0, -1)
        if resize is not None:
            nir = self.resize_img(nir, resize)
        nir = self.enhance_colors(nir)
        return nir

    def display_rgb_img(self, resize: int = None) -> None:
        rgb = self.get_rgb_array(resize)
        self.display_image(rgb)

    def display_nir_img(self, resize: int = None) -> None:
        nir = self.get_nir_array(resize)
        self.display_image(nir)

    @staticmethod
    def enhance_colors(image: np.ndarray) -> np.ndarray:
        """Stretch the values of the image to improve contrast."""
        p99, p1 = np.percentile(image.flatten(), (99, 1))
        image = (image - p1) / (p99 - p1 + 1e-6)
        image = np.clip(image, 0, 1)
        return image

    @staticmethod
    def resize_img(image: np.ndarray, max_width: int) -> np.ndarray:
        if image.shape[2] > max_width:
            scale = max_width / image.shape[0]
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        return image

    @staticmethod
    def display_image(image: np.ndarray) -> None:
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.show()


class ImageGridS2:
    def __init__(self, images: list[ImageS2], title: str = "Grille d'images") -> None:
        self.images = []
        self.title = title
        for image in tqdm(images, "Importing images..."):
            self.images.append(image)

    def display_grid_rgb(self, compression: int, title: str = "") -> None:
        images = []
        for image in tqdm(self.images, "Getting RGB images..."):
            array_img = image
            rgb = array_img.get_rgb_array(compression)
            images.append(rgb)
        self.display_img_grid(images, title)

    def display_grid_nir(self, compression: int, title: str = "") -> None:
        images = []
        for image in tqdm(self.images, "Getting NIR images..."):
            array_img = image
            nir = array_img.get_nir_array(compression)
            images.append(nir)
        self.display_img_grid(images, title)

    @staticmethod
    def display_img_grid(images: list, title: str) -> None:
        n = len(images)
        n_cols = int(np.ceil(np.sqrt(n)))
        n_rows = int(np.ceil(n / n_cols))
        fig, axs = plt.subplots(
            n_rows, n_cols, figsize=(min(10, 20 // n_rows), min(8, 30 // n_cols))
        )
        fig.suptitle(title)
        axs_flat = axs.flatten() if n > 1 else [axs]
        for i, ax in enumerate(axs_flat):
            if i < n:
                img = images[i]
                ax.set_title(f"Image {i}")
                ax.imshow(img)
            ax.axis("off")
        fig.tight_layout()
        plt.show()
