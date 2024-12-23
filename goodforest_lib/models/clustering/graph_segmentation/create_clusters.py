from datetime import datetime, timedelta

import numpy as np

from ..fetch_data import get_historical_data_with_dates
from .create_graph import build_pixel_graph
from .louvain_clustering import get_louvain_partition, regroup_smallest_clusters


def get_clusters(
    images: np.ndarray,
    dates: np.ndarray,
    resolution: float,
    bands: list[int],
    before_date: str,
    cluster_period_duration_in_days: str = 3 * 30,
    min_cluster_size: int = 8,
) -> tuple[list[list[set[tuple[int, int]]]], list[datetime]]:
    """
    Get the clusters from the historical data

    Parameters
    ----------
    images : np.ndarray
        The images to get the clusters from
    dates : np.ndarray
        The dates of the historical data
    resolution : float
        The resolution of the partition
    bands : list[int]
        The bands to fetch
    before_date : str
        The date before which to fetch the data
    cluster_period_duration_in_days : str
        The duration of the period to cluster
    min_cluster_size : int
        The minimum size of the clusters

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
    historical_data = historical_data.transpose(2, 3, 0, 1)

    graph = build_pixel_graph(historical_data)
    print(
        f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges"
    )
    partition = get_louvain_partition(graph, resolution=resolution)
    print(
        f"Partition has {len(partition)} clusters, characterizing {np.sum([len(c) for c in partition])} pixels"
    )

    partition = regroup_smallest_clusters(
        historical_data, graph, partition, min_size=min_cluster_size
    )

    return [partition], dates_clustering
