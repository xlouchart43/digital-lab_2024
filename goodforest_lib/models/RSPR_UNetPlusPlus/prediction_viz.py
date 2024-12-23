import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from .model_no_normalization import RSPRUNetPlusPlus
from .predict import Predict

from ...config.constants import DICT_CLS_COLOR_5_CLASSES


def stretch_contrast(
    image: np.ndarray, low_percentile: int = 2, high_percentile: int = 98
) -> np.ndarray:
    """
    Stretch the contrast of the image.

    Args:
        image (np.ndarray): 3D array of the image of shape (H, W, C).
        low_percentile (int): Low percentile for stretching.
        high_percentile (int): High percentile for stretching.

    Returns:
        np.ndarray: Stretched image.
    """
    low = np.percentile(image, low_percentile)
    high = np.percentile(image, high_percentile)
    stretched = np.clip((image - low) / (high - low), 0, 1)
    return stretched


def visualize_prediction(
    file_path: str,
    prediction: np.ndarray,
    output_path: str,
    dict_cls_color: dict = DICT_CLS_COLOR_5_CLASSES,
) -> None:
    """
    Visualize the prediction heatmap using matplotlib.

    Args:
        file_path (str): Path to the input raster file.
        prediction (np.ndarray): 2D array of predicted class labels of shape (H, W).
        output_path (str): Path to save the heatmap.
    """
    with rasterio.open(file_path) as src:
        original_image = src.read()
    black_pixels = np.all(original_image[:11] == 0, axis=0)

    pred = prediction.copy()
    pred = pred.reshape(pred.shape[1:])
    pred[black_pixels] = 5

    list_colors = list(dict_cls_color.values())
    cmap = ListedColormap(list_colors)
    patches = [Patch(color=color, label=cls) for cls, color in dict_cls_color.items()]

    plt.figure(figsize=(10, 8))
    plt.imshow(pred, cmap=cmap)
    plt.title("Predicted classes", fontsize=15)
    plt.legend(handles=patches, loc="upper right")
    plt.axis("off")
    plt.savefig(output_path)
    plt.show()


def visualize_best_prob_heatmap(
    file_path: str, best_prob: np.ndarray, output_path: str
) -> None:
    """
    Visualize the prediction heatmap using matplotlib.

    Args:
        file_path (str): Path to the input raster file.
        prediction (np.ndarray): 2D array of predicted class labels of shape (H, W).
        output_path (str): Path to save the heatmap.
    """
    with rasterio.open(file_path) as src:
        original_image = src.read()
    black_pixels = np.all(original_image[:11] == 0, axis=0)

    best_prob_copy = best_prob.copy()
    best_prob_copy = best_prob_copy.reshape(best_prob_copy.shape[1:])
    best_prob_copy[black_pixels] = 1

    cmap = plt.get_cmap(
        "viridis"
    )  # You can choose other colormaps like 'plasma', 'jet', etc.

    plt.figure(figsize=(10, 8))
    plt.imshow(best_prob_copy, cmap=cmap)
    plt.colorbar(label="Best Probability")
    plt.title("Best Probability Heatmap", fontsize=15)
    plt.axis("off")
    plt.savefig(output_path)
    plt.show()


def visualize_nb_pred_heatmap(
    file_path: str, nb_pred: np.ndarray, output_path: str
) -> None:
    """
    Visualize the prediction heatmap using matplotlib.

    Args:
        file_path (str): Path to the input raster file.
        prediction (np.ndarray): 2D array of predicted class labels of shape (H, W).
        output_path (str): Path to save the heatmap.
    """
    with rasterio.open(file_path) as src:
        original_image = src.read()
    black_pixels = np.all(original_image[:11] == 0, axis=0)

    nb_pred_copy = nb_pred.copy()
    nb_pred_copy[black_pixels] = 0

    cmap = plt.get_cmap(
        "viridis"
    )  # You can choose other colormaps like 'plasma', 'jet', etc.

    plt.figure(figsize=(10, 8))
    plt.imshow(nb_pred_copy, cmap=cmap)
    plt.colorbar(label="Prediction Class")
    plt.title("Number of Predictions Heatmap")
    plt.axis("off")
    plt.savefig(output_path)
    plt.show()


def model_load(
    num_classes: int, checkpoint_path: str = "", model_path: str = ""
) -> torch.nn.Module:
    """
    Load the model from the checkpoint or model path.

    Args:
        num_classes (int): Number of classes.
        checkpoint_path (str): Path to the model checkpoint.
        model_path (str): Path to the model.

    Returns:
        torch.nn.Module: Trained model
    """
    model = RSPRUNetPlusPlus(num_classes=num_classes)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state_dict"])
    elif model_path:
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    return model


def predictor(model: torch.nn.Module) -> Predict:
    """
    Create a predictor object.

    Args:
        model (torch.nn.Module): Trained model.

    Returns:
        Predict: Predictor object.
    """
    return Predict(model)
