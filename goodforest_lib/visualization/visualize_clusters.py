from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom

from ..models.clustering.utils import get_labels_from_partition

COLORS = np.array(
    [
        [150, 0, 0],
        [0, 150, 0],
        [0, 0, 150],
        [150, 150, 0],
        [0, 150, 150],
        [150, 0, 150],
        [0, 0, 0],
    ]
)


def visualize_clusters(
    clusters: list[list[set[tuple[int, int]]]],
    background_image: np.ndarray,
    scale_factor: int,
    windows_view: tuple[int, int, int, int] = None,
) -> None:
    """
    Visualize the clusters on the background image

    Parameters
    ----------
    clusters : list[list[set[tuple[int, int]]]]
        The clusters to visualize
    background_image : np.ndarray
        The background image
    zoom : int
        The zoom to apply to the image
    windows_view : tuple[int, int, int, int]
        The window to display the image (top-left corner and bottom-right corner)

    Returns
    -------
    None
    """
    if len(clusters) > len(COLORS):
        raise ValueError(
            f"Cannot visualize more than {len(COLORS)} clusters with different colors"
        )
    if windows_view is None:
        windows_view = 0, 0, background_image.shape[0], background_image.shape[1]

    h, w = background_image.shape[:2]
    rgb_big = zoom(background_image, (scale_factor, scale_factor, 1), order=3)

    border_layer = np.zeros((len(clusters), *rgb_big.shape))

    for grid_idx, (grid, color) in enumerate(zip(clusters, COLORS)):
        labels = get_labels_from_partition(grid)
        for i in range(h):
            for j in range(w):
                if labels[i, j] < 0:
                    continue
                if i > 0 and labels[i, j] != labels[i - 1, j]:
                    border_layer[
                        grid_idx,
                        scale_factor * i,
                        scale_factor * j : scale_factor * (j + 1),
                    ] = color
                if i < h - 1 and labels[i, j] != labels[i + 1, j]:
                    border_layer[
                        grid_idx,
                        scale_factor * (i + 1),
                        scale_factor * j : scale_factor * (j + 1),
                    ] = color
                if j > 0 and labels[i, j] != labels[i, j - 1]:
                    border_layer[
                        grid_idx,
                        scale_factor * i : scale_factor * (i + 1),
                        scale_factor * j,
                    ] = color
                if j < w - 1 and labels[i, j] != labels[i, j + 1]:
                    border_layer[
                        grid_idx,
                        scale_factor * i : scale_factor * (i + 1),
                        scale_factor * (j + 1),
                    ] = color
    border_layer = np.mean(border_layer, axis=0)
    border_mask = np.any(border_layer > 0, axis=-1)
    rgb_big[border_mask] = 0.3 * rgb_big[border_mask] + 0.7 * border_layer[border_mask]
    rgb_big = np.clip(rgb_big, 0, 1)
    plt.figure(figsize=(12, 8))
    plt.title("Clusters")

    y1, x1, y2, x2 = windows_view
    w, h = x2 - x1, y2 - y1

    plt.imshow(
        rgb_big[
            y1 * scale_factor : y2 * scale_factor, x1 * scale_factor : x2 * scale_factor
        ]
    )
    plt.xticks(
        np.arange(0, w * scale_factor, scale_factor)[:: w // 15],
        np.arange(x1, x2)[:: w // 15],
    )
    plt.yticks(
        np.arange(0, h * scale_factor, scale_factor)[:: h // 10],
        np.arange(y1, y2)[:: h // 10],
    )
    # Add a legend for each border color
    for i in range(len(clusters)):
        plt.plot([], [], color=tuple(COLORS[i] / 255), label=f"Grid {i+1}")
    plt.legend()
    plt.show()


def display_cluster_behavior(
    historical_data: np.ndarray,
    dates: np.ndarray,
    clusters: list[list[set[tuple[int, int]]]],
    date_clustering: list[datetime],
    pixel: tuple[int, int],
    channels: list[int],
):
    if len(channels) > len(COLORS):
        raise ValueError(
            "Too many channels to display for the number of available colors"
        )

    nb_layers = len(clusters)

    # Get the pixels in the cluster
    clusters_of_pixel = []
    for cluster in clusters:
        for nodes in cluster:
            if pixel in nodes:
                nodes_i, nodes_j = zip(*list(nodes))
                clusters_of_pixel.append([nodes_i, nodes_j])
                break

    # Extract the time series of the cluster
    clusters_data = []
    for nodes_i, nodes_j in clusters_of_pixel:
        cluster_data = []
        for i, j in zip(nodes_i, nodes_j):
            cluster_data.append(historical_data[i, j, :, :])
        cluster_data = np.array(cluster_data)
        clusters_data.append(cluster_data)

    # Plot the time series
    fig, axs = plt.subplots(nb_layers, 1)
    fig.set_size_inches(15, 5 * nb_layers)

    for cluster_data, ax in zip(clusters_data, np.array(axs).flatten()):
        # Display years in background
        colors = ["green", "lightgreen"]
        years = list(set(map(lambda x: x.year, dates)))
        for i, year in enumerate(years):
            date_begin = datetime(year, 1, 1)
            date_end = datetime(year, 12, 31)
            if i == 0:
                date_begin = dates[0]
            if i == len(years) - 1:
                date_end = dates[-1]
            ax.axvspan(date_begin, date_end, color=colors[i % 2], alpha=0.1)

        for channel, color in zip(channels, COLORS):
            color = tuple(color / 255)
            channel_data = cluster_data[:, :, channel]
            for i in range(len(channel_data)):
                not_nan_mask = ~np.isnan(channel_data[i])
                ax.plot(
                    dates[not_nan_mask],
                    channel_data[i][not_nan_mask],
                    color=color,
                    alpha=0.02,
                )

            channel_data = np.array(channel_data)
            median_channel = np.nanmedian(channel_data, axis=0)

            not_nan_mask = ~np.isnan(median_channel)

            ax.plot(
                dates[not_nan_mask],
                median_channel[not_nan_mask],
                color=color,
                label=f"Median channel {channel}",
                lw=1.2,
            )
            ax.set_xlabel("Date")
            ax.set_ylabel("Channels value")
            ax.legend()

    fig.suptitle("Time series of cluster")
    plt.show()


def display_outlying_px_behaviour(
    historical_data: np.ndarray,
    dates_clustering: list[datetime],
    dates_detection: list[datetime],
    partition: list[list[set[tuple[int, int]]]],
    thresholds: np.ndarray,
    pixel: tuple[int, int],
):
    dates = np.hstack([dates_clustering, dates_detection])

    nb_channels = historical_data.shape[3]
    if nb_channels > len(COLORS):
        raise ValueError(
            "Too many channels to display for the number of available colors"
        )

    nb_layers = len(partition)

    # Get the pixels in the cluster
    clusters_of_pixel = []
    for grid in partition:
        for cluster in grid:
            if pixel in cluster:
                nodes_i, nodes_j = zip(*list(cluster))
                clusters_of_pixel.append([nodes_i, nodes_j])
                break

    # Extract the time serie of the pixel
    pixel_data = historical_data[pixel[0], pixel[1], :, :]

    # Extract the time series of the cluster
    clusters_data = []
    for nodes_i, nodes_j in clusters_of_pixel:
        cluster_data = historical_data[nodes_i, nodes_j, :, :]
        clusters_data.append(cluster_data)

    # Plot the time series
    fig, axs = plt.subplots(nb_layers, 1)
    fig.set_size_inches(15, 5 * nb_layers)

    for cluster_data, ax in zip(clusters_data, np.array(axs).flatten()):
        # Plot rectangle to draw the line between the clustering and the detection
        sep_dates = (
            dates_clustering[0],
            dates_clustering[-1]
            + timedelta(days=(dates_detection[0] - dates_clustering[-1]).days),
            dates_detection[-1],
        )
        ax.axvspan(sep_dates[0], sep_dates[1], color="green", alpha=0.1)
        ax.axvspan(sep_dates[1], sep_dates[2], color="blue", alpha=0.1)

        for channel, color in zip(list(range(nb_channels)), COLORS):
            color = tuple(color / 255)
            channel_data = cluster_data[:, :, channel]

            # Display pixel data
            not_nan_mask = ~np.isnan(pixel_data[:, channel])
            ax.scatter(
                dates[not_nan_mask],
                pixel_data[:, channel][not_nan_mask],
                color=color,
                alpha=1,
                marker="+",
            )

            median_channel = np.nanmedian(channel_data, axis=0)

            not_nan_mask = ~np.isnan(median_channel)

            ax.plot(
                dates[not_nan_mask],
                median_channel[not_nan_mask],
                color=color,
                label=f"Median channel {channel}",
                lw=1.2,
            )
            ax.fill_between(
                dates[not_nan_mask],
                median_channel[not_nan_mask] - thresholds[channel],
                median_channel[not_nan_mask] + thresholds[channel],
                color=color,
                alpha=0.2,
                label=f"Threshold channel {channel}",
            )
            ax.set_xlabel("Date")
            ax.set_ylabel("Channels value")
            ax.legend()

    fig.suptitle("Time series of cluster")
    plt.show()
