import networkx as nx
import numpy as np


def f_correction(x: np.float32) -> np.float32:
    f = lambda v: np.pow(v, 2)
    return np.clip(f(x), 0, 1)


def get_weights_from_arrays(array1: np.ndarray, array2: np.ndarray) -> np.ndarray:
    if array1.shape != array2.shape:
        raise ValueError("Arrays must have the same shape")

    delta = np.abs(array1 - array2).reshape(*array1.shape[:2], -1)
    delta = np.where(np.isnan(delta), 0, delta)
    delta = np.linalg.norm(delta, axis=2)

    stacked_arrays = np.concatenate([array1, array2], axis=3)
    nb_union_nan = np.sum(np.all(~np.isnan(stacked_arrays), axis=3), axis=2)
    delta_norm = delta / (nb_union_nan + 1e-6)

    delta_norm = f_correction(delta_norm)

    # Correct those with nb__union_nan = 0 to -1 instead of 0
    delta_norm = np.where(nb_union_nan == 0, -1, delta_norm)

    return delta_norm


def build_pixel_graph(historical_data: np.ndarray) -> nx.Graph:
    """
    Build a graph from the historical data

    Parameters
    ----------
    historical_data : np.ndarray
        The historical data to build the graph from

    Returns
    -------
    nx.Graph
        The graph built from the historical data
    """
    height, width = historical_data.shape[:2]
    graph = nx.Graph()

    vertical_edges_weights = get_weights_from_arrays(
        historical_data[:-1, :], historical_data[1:, :]
    )
    horizontal_edges_weights = get_weights_from_arrays(
        historical_data[:, :-1], historical_data[:, 1:]
    )
    diagonal_edges_weights = get_weights_from_arrays(
        historical_data[:-1, :-1], historical_data[1:, 1:]
    )
    diagonal_edges_weights_2 = get_weights_from_arrays(
        historical_data[1:, :-1], historical_data[:-1, 1:]
    )

    for i in range(height):
        for j in range(width):
            # Add vertical edges
            if i < height - 1:
                weight = vertical_edges_weights[i, j]
                if weight != -1:
                    graph.add_edge((i, j), (i + 1, j), weight=weight)
            # Add horizontal edges
            if j < width - 1:
                weight = horizontal_edges_weights[i, j]
                if weight != -1:
                    graph.add_edge((i, j), (i, j + 1), weight=weight)
            # Add diagonal edges
            if i < height - 1 and j < width - 1:
                weight = diagonal_edges_weights[i, j]
                if weight != -1:
                    graph.add_edge((i, j), (i + 1, j + 1), weight=weight)
            if i > 0 and j < width - 1:
                weight = diagonal_edges_weights_2[i - 1, j]
                if weight != -1:
                    graph.add_edge((i, j), (i - 1, j + 1), weight=weight)

    return graph
