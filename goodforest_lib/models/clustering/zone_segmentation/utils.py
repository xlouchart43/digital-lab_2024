def get_original_pixel(
    idx: int, zone_idx: tuple[int, int], zone_size: int, image_size: tuple[int, int]
) -> tuple[int, int]:
    """
    Get the original pixel coordinates from the index of the pixel in the flattened zone.

    Args:
        idx (int): Index of the pixel in the flattened zone.
        zone_idx (tuple(int,int)): The starting coordinates of the zone.
        zone_size (int): The size of the zone.
        image_size (tuple(int,int)): The size of the image.

    Returns:
        tuple(int,int): The original pixel coordinates (row, col).
    """
    zone_width = min(zone_size, image_size[1] - zone_idx[1])
    return zone_idx[0] + idx // zone_width, zone_idx[1] + idx % zone_width
