import numpy as np

from ...config.constants import CUBE_SIZE, NB_S2_BANDS


def get_cube_quality(data: np.ndarray, threshold: float) -> None:
    """
    Estimate the quality of the cube.
    The dimension of the cube is (NB_CHANNELS_WITH_LABELS, CUBE_SIZE, CUBE_SIZE).
    """
    # Compute ratio of black pixels
    black_pixels = np.sum(data[:NB_S2_BANDS] == 0, axis=0)
    black_pixel_ratio = np.sum(black_pixels == NB_S2_BANDS) / (CUBE_SIZE * CUBE_SIZE)
    return black_pixel_ratio <= threshold
