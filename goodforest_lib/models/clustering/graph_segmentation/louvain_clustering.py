import warnings

import networkx as nx
import numpy as np

from ....config.constants import RANDOM_SEED
from ..utils import get_labels_from_partition


def get_louvain_partition(
    graph: nx.Graph, resolution: float = 1000
) -> list[set[tuple[int, int]]]:
    """
    Compute the Louvain partition of a graph

    Parameters
    ----------
    graph : nx.Graph
        The graph to partition
    resolution : float
        The resolution of the partition

    Returns
    -------
    list[set[tuple[int, int]]]
        The partition of the graph
    """
    partition = nx.community.louvain_communities(
        graph, resolution=resolution, seed=RANDOM_SEED
    )
    return partition


def compute_centroids(
    historical_data: np.ndarray, partition: list[set[tuple[int, int]]]
) -> np.ndarray:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    centroids = np.empty((len(partition), *historical_data.shape[2:]), dtype=np.float32)
    for i, cluster in enumerate(partition):
        nodes = list(cluster)
        nodes_y, nodes_x = zip(*nodes)
        cluster_data = historical_data[nodes_y, nodes_x, :, :]
        centroids[i] = np.nanmedian(cluster_data, axis=0)
    return centroids


def regroup_smallest_clusters(
    historical_data: np.ndarray,
    graph: nx.Graph,
    partition: list[set[tuple[int, int]]],
    min_size: int = 8,
) -> list[set[tuple[int, int]]]:
    """
    Regroup the smallest clusters of a partition

    Parameters
    ----------
    historical_data : np.ndarray
        The historical data
    graph : nx.Graph
        The graph
    partition : list[set[tuple[int, int]]]
        The partition to regroup
    min_size : int, optional
        The minimum size of a cluster, by default 8

    Returns
    -------
    list[set[tuple[int, int]]]
        The new partition
    """
    small_clusters = [
        [idx, cluster]
        for idx, cluster in enumerate(partition)
        if len(cluster) < min_size
    ]
    nodes = np.array(graph.nodes)
    labels = get_labels_from_partition(partition)
    centroids = compute_centroids(historical_data, partition)

    new_partition = partition.copy()
    clusters_to_delete = []

    for cluster_idx, cluster in small_clusters:
        # Check if it has been merged in the meantime
        if len(cluster) >= min_size:
            continue

        available_clusters = set()
        for i, j in cluster:
            neighbors = [
                (i - 1, j - 1),
                (i - 1, j),
                (i - 1, j + 1),
                (i, j - 1),
                (i, j + 1),
                (i + 1, j - 1),
                (i + 1, j),
                (i + 1, j + 1),
            ]
            for n_i, n_j in neighbors:
                if graph.has_edge((i, j), (n_i, n_j)):
                    available_clusters.add(int(labels[n_i][n_j]))
        available_clusters -= {-1, cluster_idx}

        # Compute norm2 distance between clusters centroids without nan values
        available_clusters = np.array(list(available_clusters), dtype=int)
        available_centroids = centroids[available_clusters]
        cluster_centroid = centroids[cluster_idx]
        cluster_nan_mask = np.any(np.isnan(cluster_centroid), axis=1)
        available_centroids_nan_mask = np.any(np.isnan(available_centroids), axis=2)

        distances_centroids = np.empty((available_centroids.shape[0],))
        for i, available_centroid in enumerate(available_centroids):
            nan_mask = np.logical_or(cluster_nan_mask, available_centroids_nan_mask[i])
            available_centroid = available_centroid[~nan_mask]
            cluster_centroid_temp = cluster_centroid[~nan_mask]
            distances_centroids[i] = (
                np.linalg.norm(available_centroid - cluster_centroid_temp, ord=2)
                / np.sum(~nan_mask)
                if len(available_centroid) > 0
                else np.inf
            )

        if len(distances_centroids) == 0:
            print(
                f"No available clusters for cluster {cluster_idx} with size {len(cluster)}"
            )
            continue

        # Find closest cluster
        closest_cluster_idx = available_clusters[np.argmin(distances_centroids)]
        if closest_cluster_idx in list(map(lambda x: x[0], small_clusters)):
            # If the closest cluster is also a small cluster, we merge the two and store it in the future small cluster to analyze
            other_small_idx = list(map(lambda x: x[0], small_clusters)).index(
                closest_cluster_idx
            )
            small_clusters[other_small_idx][1] = small_clusters[other_small_idx][
                1
            ].union(cluster)

        new_partition[closest_cluster_idx] = new_partition[closest_cluster_idx].union(
            cluster
        )
        nodes = [[i, j] for i, j in cluster]
        nodes_i, nodes_j = zip(*nodes)
        labels[nodes_i, nodes_j] = closest_cluster_idx
        clusters_to_delete.append(cluster_idx)

    new_partition = [
        cluster
        for idx, cluster in enumerate(new_partition)
        if not idx in clusters_to_delete
    ]

    return new_partition
