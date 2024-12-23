import numpy as np

from ..config import BANDS


def compute_cr_swir(image: np.ndarray):
    """
    Compute the CR_SWIR band from the input image
    """
    channel_CR_SWIR = (
        BANDS[9].denormalize(image[9])
        / (
            BANDS[8].denormalize(image[8])
            + (1613 - 864)
            * (BANDS[10].denormalize(image[10]) - BANDS[8].denormalize(image[8]))
            / (2200 - 864)
        )
    ) * 90  # To limit the range to 0-255
    return channel_CR_SWIR


def append_band_values(image: np.ndarray, bands: list[str]) -> np.ndarray:
    """
    Append the computed bands to the input image
    """
    for band in bands:
        if band == "CR_SWIR":
            new_channel = compute_cr_swir(image)
        else:
            raise NotImplementedError(f"Band {band} is not implemented")
        new_channel = np.expand_dims(new_channel, axis=0)
        image = np.concatenate((image, new_channel), axis=0)
    return image
