import warnings

import numpy as np
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import davies_bouldin_score


def davies_bouldin_method(time_series: list[np.ndarray], max_clusters: int) -> int:
    """
    Compute the Davies-Bouldin score for different numbers of clusters and visualize the results.

    Args:
        time_series (list[np.ndarray]): list of time series.
        max_clusters (int): Maximum number of clusters to consider.

    Returns:
        int: The best number of clusters according to the Davies-Bouldin method.
    """
    best_k = 2
    best_score = float("inf")
    n_samples = len(time_series)
    max_clusters = min(max_clusters, n_samples)
    for i in range(2, max_clusters + 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            kmeans = KMeans(
                n_clusters=i, random_state=42, init="k-means++", max_iter=300, n_init=1
            )
            labels = kmeans.fit_predict(time_series)
            n_labels = len(set(labels))
            if n_labels < 2:
                continue
            db_score = davies_bouldin_score(time_series, labels)
            if db_score < best_score:
                best_k = i
                best_score = db_score

    # If needed uncomment the next lines to visualize the Davies-Bouldin score

    # plt.scatter(k, db_scores[k-2], color='red', label=f'Best k = {k+2}')
    # plt.title('Davies-Bouldin Method')
    # plt.xlabel('Number of Clusters')
    # plt.ylabel('Davies-Bouldin Score')
    # plt.show()

    return best_k


def kmeans_with_euc(
    time_series: list[np.ndarray], n_clusters: int
) -> tuple[np.ndarray, list[int]]:
    """
    Perform K-means clustering on time series data using the euclidean distance.

    Args:
        time_series (list[np.ndarray]): list of time series.
        n_clusters (int): Number of clusters.

    Returns:
        tuple[np.ndarray, list[int]]: Cluster centers and the cluster labels for each time series.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            init="k-means++",
            max_iter=300,
            n_init=1,
        )
        labels = kmeans.fit_predict(time_series)  # change to distance_matrix if needed
    centroids = kmeans.cluster_centers_

    for cluster_label in np.unique(labels):
        cluster_indices = np.where(labels == cluster_label)[0]
        if len(cluster_indices) == 1:
            single_element_idx = cluster_indices[0]

            single_element_distances = np.array(
                [
                    np.linalg.norm(centroids[centroid_idx] - centroids[cluster_label])
                    for centroid_idx in range(len(centroids))
                    if centroid_idx != cluster_label
                ]
            )

            nearest_cluster = np.argmin(single_element_distances)
            labels[single_element_idx] = nearest_cluster

    return centroids, labels
