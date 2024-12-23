import numpy as np


def get_overlapped_swir_zones(
    images: np.ndarray, zone_size: int
) -> tuple[
    dict[tuple[int, int], np.ndarray], dict[tuple[int, int], list[tuple[int, int]]]
]:
    """
    Get the overlapped zones of an array of images with their starting coordinates, where new zones
    start at the middle of the initial zones, ensuring no pixel in the initial zones is left uncovered.

    Args:
        images (np.ndarray): The array of images of shape (num_images, channels, height, width).
        zone_size (int): The size of the zones.

    Returns:
        tuple[dict[tuple[int, int], np.ndarray], dict[tuple[int, int], list[tuple[int, int]]]:
            A tuple containing a dictionary with the overlapped zones and their starting coordinates,
            and a dictionary with the coordinates of the pixels and the zones they belong to.
    """
    zones_with_positions = {}
    height, width = images.shape[2], images.shape[3]

    pixel_dict = {}
    for i in range(0, height, zone_size):
        for j in range(0, width, zone_size):
            zone = images[:, -1, i : i + zone_size, j : j + zone_size]
            zones_with_positions[(i, j)] = zone
            for x in range(i, min(i + zone_size, height)):
                for y in range(j, min(j + zone_size, width)):
                    pixel_dict[(x, y)] = pixel_dict.get((x, y), []) + [(i, j)]

    for i in range(zone_size // 2, height, zone_size):
        for j in range(zone_size // 2, width, zone_size):
            zone = images[:, -1, i : i + zone_size, j : j + zone_size]
            zones_with_positions[(i, j)] = zone
            for x in range(i, min(i + zone_size, height)):
                for y in range(j, min(j + zone_size, width)):
                    pixel_dict[(x, y)] = pixel_dict.get((x, y), []) + [(i, j)]

    return zones_with_positions, pixel_dict
