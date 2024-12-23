import json
from datetime import datetime, timedelta

import numpy as np
from tqdm import tqdm

from ..fetch_data import get_historical_data_with_dates
from .clustering_process import davies_bouldin_method, kmeans_with_euc
from .prepare_data import get_overlapped_swir_zones
from .utils import get_original_pixel

ZONE_SIZE = 10


def get_clusters(
    images: np.ndarray,
    dates: np.ndarray,
    bands: list[int],
    before_date: str,
    cluster_period_duration_in_days: str = 3 * 30,
) -> tuple[list[list[set[tuple[int, int]]]], list[datetime]]:
    """
    Get the clusters from the historical data

    Parameters
    ----------
    images : np.ndarray
        The images to get the clusters from
    dates : np.ndarray
        The dates of the historical data
    bands : list[int]
        The bands to fetch
    before_date : str
        The date before which to fetch the data
    cluster_period_duration_in_days : str
        The duration of the period to cluster


    Returns
    -------
    list[list[set[tuple[int, int]]]]
        The clusters of the historical data
    list[datetime]
        The dates of the clustering
    """
    after_date = datetime.strptime(before_date, "%Y-%m-%d") - timedelta(
        days=int(cluster_period_duration_in_days)
    )
    after_date = after_date.strftime("%Y-%m-%d")
    historical_data, dates_clustering = get_historical_data_with_dates(
        images, dates, bands=bands, after_date=after_date, before_date=before_date
    )
    historical_data = np.nan_to_num(historical_data)

    zones_with_swir, pixel_dict = get_overlapped_swir_zones(historical_data, ZONE_SIZE)

    global_clusters, global_centroids, global_labels = clustering_zones_step(
        zones_with_swir
    )

    partitions = get_outlier_input(
        global_clusters, pixel_dict, 2, images[0].shape[-2:], ZONE_SIZE
    )

    return partitions, dates_clustering


def get_outlier_input(
    global_clusters: dict[int, dict[int, list[int]]],
    pixel_dict: dict[int, list[int]],
    num_grid: int,
    image_size: tuple[int, int],
    zone_size=ZONE_SIZE,
) -> list[list[set[tuple[int, int]]]]:
    """
    Get outlier detection input from the clusters and pixel dictionary of the historical data and the image size

    Parameters
    ----------
    global_clusters : dict[int, dict[int, list[int]]]
        The clusters of the historical data
    pixel_dict : dict[int, list[int]]
        The pixel dictionary of the historical data
    num_grid : int
        The number of grids to divide the data into
    image_size : tuple[int, int]
        The size of the image
    zone_size : int
        The size of the zone


    Returns
    -------
    list[list[set[tuple[int, int]]]]
        The outlier detection input from the clusters and pixel dictionary of the historical data
    """
    clusters = {
        zone: {
            key: [get_original_pixel(idx, zone, zone_size, image_size) for idx in value]
            for key, value in zone_data.items()
        }
        for zone, zone_data in global_clusters.items()
    }
    grid = [[] for _ in range(num_grid)]
    for i in range(num_grid):
        values = pixel_dict.values()
        for value in values:
            if i < len(value) and value[i] not in grid[i]:
                grid[i].append(value[i])
    res = [[] for _ in range(num_grid)]
    for i in range(num_grid):
        for zone in grid[i]:
            res[i].extend(list(clusters[zone].values()))
    for i in range(num_grid):
        res[i] = [set(x) for x in res[i]]
    return res


def clustering_zones_step(
    zones_with_swir: list[tuple[np.ndarray, tuple[int, int]]]
) -> tuple[
    dict[int, dict[int, list[int]]], dict[int, np.ndarray], dict[int, list[int]]
]:
    """
    Perform clustering on the healthy CR-SWIR.

    Args:
        zones_with_swir (List[Tuple[np.ndarray, Tuple[int, int]]]): List of tuples, where each tuple contains the zone
                                                                   (as a numpy array) and its starting (row, col) coordinates.
    Returns:
        Tuple[Dict[int, Dict[int, List[int]]], Dict[int, np.ndarray], Dict[int, List[int]]]: A dictionary of clusters for each zone, a dictionary of centroids for each zone.
    """
    global_clusters = {}
    global_centroids = {}
    global_labels = {}
    for i in tqdm(zones_with_swir.keys()):
        zone = zones_with_swir[i]
        zone_time_series = zone.transpose(1, 2, 0).reshape(-1, zone.shape[0])
        k = davies_bouldin_method(zone_time_series, 10)
        clusters, labels = kmeans_with_euc(zone_time_series, k)
        global_centroids[i] = clusters
        global_labels[i] = labels
        clusters = {}
        for idx, label in enumerate(labels):
            clusters[label] = clusters.get(label, []) + [idx]
        global_clusters[i] = clusters
    return global_clusters, global_centroids, global_labels
