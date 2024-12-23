import numpy as np


def estimate_image_size_from_partition(partition: list[set[int]]) -> tuple[int, int]:
    max_i = max([max(nodes_set, key=lambda x: x[0])[0] for nodes_set in partition])
    max_j = max([max(nodes_set, key=lambda x: x[1])[1] for nodes_set in partition])
    return max_i + 1, max_j + 1


def get_labels_from_partition(
    partition: list[set[int]], image_size: tuple[int, int] = None
) -> np.ndarray:
    """
    Build the image with the labels of the clusters

    Parameters
    ----------
    partition : list[set[int]]
        The partition of the graph
    image_size : tuple[int, int], optional
        The size of the image, by default None, in which case it is estimated from the partition

    Returns
    -------
    np.ndarray
        The image with the labels of the clusters
    """
    if image_size is None:
        image_size = estimate_image_size_from_partition(partition)

    print(image_size)

    labels = -np.ones(image_size, dtype=int)

    for cluster_index, nodes_set in enumerate(partition):
        nodes_list = np.array(list(nodes_set))
        nodes_list_i, nodes_list_j = nodes_list[:, 0], nodes_list[:, 1]
        labels[nodes_list_i, nodes_list_j] = cluster_index

    return labels
