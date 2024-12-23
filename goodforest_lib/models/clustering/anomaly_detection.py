from datetime import datetime, timedelta

import numpy as np

from ...config.constants import ZONE_CLUSTER_SIZE
from .fetch_data import get_historical_data_with_dates


def predict_dieback(
    images: np.ndarray,
    dates: np.ndarray,
    bands: list[int],
    before_date: str,
    clusters: list[list[set[tuple[int, int]]]],
    detection_period_duration_in_days: str = 3 * 30,
) -> tuple[list[list[set[tuple[int, int]]]], list[datetime]]:
    after_date = datetime.strptime(before_date, "%Y-%m-%d") - timedelta(
        days=int(detection_period_duration_in_days)
    )
    after_date = after_date.strftime("%Y-%m-%d")
    historical_data, dates_detection = get_historical_data_with_dates(
        images, dates, bands=bands, after_date=after_date, before_date=before_date
    )
    historical_data = historical_data.transpose(2, 3, 0, 1)

    anomalies, anomalies_degrees = anomaly_detection(
        historical_data=historical_data,
        global_clusters=clusters,
        threshold_factor=1,
        threshold_count=3,
        zone_size=ZONE_CLUSTER_SIZE,
    )
    return anomalies, anomalies_degrees, dates_detection


def anomaly_detection(
    historical_data: np.ndarray,
    global_clusters: list[list[set[tuple[int, int]]]],
    threshold_factor: float,
    threshold_count: int,
    zone_size: int = ZONE_CLUSTER_SIZE,
) -> tuple[list, dict]:
    """
    Detect anomalies in given zones based on SWIR values and clustering.

    Args:
        zone_size (int): The size of each zone.
        detection_swir (dict): SWIR values for each zone in the detection period.
        global_clusters (list[list[set]]): Clusters for each grid.
        threshold_factor (float): Factor to adjust anomaly detection threshold.
        threshold_count (int): Count threshold to flag an anomaly.

    Returns:
        tuple: A tuple containing the list of anomalies and a dictionary
               with 'detected' and 'flagged' anomalies.
    """

    anomalies_degree = {}

    anomalies = detect_zone_anomalies_with_max_std(
        zone_size, historical_data, global_clusters, threshold_factor, threshold_count
    )

    pixel_counts = {}
    for pixel, _, _ in anomalies:
        if pixel in pixel_counts:
            pixel_counts[pixel] += 1
        else:
            pixel_counts[pixel] = 1

    # List of detected anomalies (appear only once)
    anomalies_degree["detected"] = [
        pixel for pixel, count in pixel_counts.items() if count == 1
    ]

    # List of flagged anomalies (appear more than once)
    anomalies_degree["flagged"] = [
        pixel for pixel, count in pixel_counts.items() if count >= 2
    ]

    return anomalies, anomalies_degree


def detect_zone_anomalies_with_max_std(
    zone_size: int,
    detection_swir: dict,
    global_clusters: dict,
    threshold_factor: float,
    threshold_count: int,
) -> list[tuple[tuple[int, int], np.ndarray]]:
    """
    Detect anomalies in a zone by monitoring deviations from updated cluster centroids using
    max-std-based adaptive thresholds.

    Args:
        zone_size (int): The size of the zone.
        detection_swir (dict): Dictionary containing SWIR time series data for zones for the detection period.
        global_clusters (list[list[set]]): Clusters for each grid.
        threshold_factor (float): Factor to adjust the sensitivity of the adaptive threshold.
        threshold_count (int): The number of times a pixel must be flagged to be considered an anomaly.

    Returns:
        List[Tuple[Tuple[int, int], np.ndarray]]: List of detected anomalies and their per-date deviation distances.
    """
    anomalies = []

    deviations = outlier_detect(global_clusters, detection_swir, threshold_factor)

    for pixel, deviation_per_date_per_channel, threshold in deviations:
        consecutive_count = 0
        for deviation in deviation_per_date_per_channel:
            if np.any(deviation > threshold):
                consecutive_count += 1
                if consecutive_count == threshold_count:
                    anomalies.append((pixel, deviation_per_date_per_channel, threshold))
                    break
            else:
                consecutive_count = 0

    return anomalies


def outlier_detect(
    clusters: list[list[set]],
    historical_data: np.ndarray,
    threshold_factor: float,
) -> list[tuple[tuple[int, int], np.ndarray, float]]:
    """
    Detect outliers in the clusters using the max std threshold method.

    Args:
        clusters (list[list[set]]): Clusters for each grid.
        detection_swir (dict): Dictionary containing SWIR time series data for zones for the detection period.
        zone_size (int): The size of the zone.
        threshold_factor (float): Factor to adjust the sensitivity of the adaptive threshold.

    Returns:
        list[Tuple[Tuple[int, int], np.ndarray, float]]: List of detected outliers and their per-date deviation distances.
    """
    deviations = []
    for grid in clusters:
        for cluster in grid:
            nodes = [[i, j] for i, j in cluster]
            nodes_i, nodes_j = zip(*nodes)
            cluster_swir = historical_data[nodes_i, nodes_j, :, :]
            cluster_centroid = np.nanmedian(cluster_swir, axis=0)
            for i, j in cluster:
                pixel_ = historical_data[i, j, :, :]
                deviation_detected, deviation_per_date, adaptive_threshold = (
                    detect_deviation_with_max_std(
                        pixel_, cluster_swir, cluster_centroid, threshold_factor
                    )
                )
                if deviation_detected:
                    deviations.append(((i, j), deviation_per_date, adaptive_threshold))
    return deviations


def compute_max_std_threshold(
    cluster_time_series: np.ndarray, threshold_factor: float
) -> float:
    """
    Compute a single adaptive threshold based on the maximum standard deviation of the SWIR values across all dates.

    Args:
        cluster_time_series (np.ndarray): SWIR time series of all pixels in the cluster (shape: n_pixels, n_timesteps, n_channels).
        threshold_factor (float): A factor that controls the sensitivity of the threshold.

    Returns:
        float: The computed maximum threshold value based on the max std.
    """

    variability_per_date = np.nanstd(cluster_time_series, axis=0)
    max_std = np.max(variability_per_date, axis=0)
    min_threshold = np.ones_like(max_std) * 0.05
    adaptive_threshold = np.maximum(min_threshold, max_std * threshold_factor)
    return adaptive_threshold


def detect_deviation_with_max_std(
    pixel_data: np.ndarray,
    cluster_time_series: np.ndarray,
    updated_centroid: np.ndarray,
    threshold_factor: float,
) -> tuple[bool, np.ndarray]:
    """
    Detect if a pixel deviates from the updated centroid of its cluster using a max-std-based adaptive threshold.

    Args:
        pixel_swir (np.ndarray): The SWIR time series of the pixel (shape: n_timesteps, n_channels).
        cluster_time_series (np.ndarray): The SWIR time series of all pixels in the cluster (shape: n_pixels, n_timesteps).
        updated_centroid (np.ndarray): The updated centroid (excluding the pixel's time series) (shape: n_timesteps).
        threshold_factor (float): Factor to adjust the sensitivity of the adaptive threshold.

    Returns:
        Tuple[bool, np.ndarray, float]: A tuple containing a boolean indicating if a deviation was detected, the deviation per date, and the adaptive threshold.
    """

    deviation_per_date = pixel_data - updated_centroid
    adaptive_threshold = compute_max_std_threshold(
        cluster_time_series, threshold_factor
    )

    deviation_detected = np.any(deviation_per_date > adaptive_threshold)

    return deviation_detected, deviation_per_date, adaptive_threshold
