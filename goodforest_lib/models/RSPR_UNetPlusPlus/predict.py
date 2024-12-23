from typing import Dict, Tuple

import geopandas as gpd
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from shapely.geometry import shape

from ...config.constants import CUBE_SIZE, OVERLAP_PREDICTION_SIZE, BORDER_EFFECT_PADDING


class Predict:
    def __init__(self, model: torch.nn.Module, num_classes: int, device: torch.device = None) -> None:
        """
        Initialize the predictor object.

        Args:
            model (torch.nn.Module): Trained model.
            device (torch.device): Device to run the model on.
        """
        self.model = model
        self.num_classes = num_classes
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        self.model.eval()


    def postprocess_output(self, output: torch.Tensor) -> np.ndarray:
        """
        Post-process the model output into classes.

        Args:
            output (torch.Tensor): Model output tensor.

        Returns:
            np.ndarray: Predicted classes."""
        output = F.softmax(output, dim=1)
        best_prob_tensor, pred_tensor = torch.max(output, dim=1)
        pred_array = pred_tensor.cpu().numpy()
        best_prob_array = best_prob_tensor.cpu().numpy()
        return pred_array, best_prob_array

    def get_logits(self, image: np.ndarray) -> np.ndarray:
        """
        Predict on an image.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Predictions.
        """
        image_test = torch.from_numpy(image).to(torch.float32)
        image_test = image_test.unsqueeze(0)
        print("image shape: ", image_test.shape)
        self.model.eval()
        with torch.no_grad():
            image_test = image_test.to(self.device)
            outputs = self.model(image_test)

        return outputs
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Predict on an image.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Predictions.
        """
        outputs = self.get_logits(image)

        pred_array, best_prob_array = self.postprocess_output(outputs)
        return pred_array, best_prob_array

    def get_logits_cubes(self, cubes: list[np.ndarray]) -> np.ndarray:
        """
        Returns class probabilities on cubes.

        Args:
            cubes (np.ndarray): Input cubes.

        Returns:
            np.ndarray: Class probabilities per pixel.
        """
        n_cubes = len(cubes)
        _, h, w = cubes[0].shape
        probs = np.empty((n_cubes, self.num_classes, h, w))

        for i in range(n_cubes):
            cube = cubes[i]
            cube = torch.from_numpy(cube).to(torch.float32)
            cube = cube.unsqueeze(0)
            self.model.eval()
            with torch.no_grad():
                cube = cube.to(self.device)
                prob = self.model(cube)
            probs[i] = prob[0]
        probs = torch.from_numpy(probs)
        probs = F.softmax(probs, dim=1)
        
        return probs
    
    def predict_cubes(self, cubes: np.ndarray) -> np.ndarray:
        """
        Predict on cubes.

        Args:
            cubes (np.ndarray): Input cubes.

        Returns:
            np.ndarray: Predictions.
        """
        probs = self.get_logits_cubes(cubes)
        pred_array, _ = self.postprocess_output(probs)

        return pred_array

    def most_frequent_positive_np(self, arr):
        # Replace non-positive values with NaN to ignore them in calculations
        arr = np.where(arr >= 0, arr, np.nan)

        # Find the mode (most frequent value) using np.bincount for integer-like values.
        def get_most_frequent(arr_slice):
            values, counts = np.unique(
                arr_slice[~np.isnan(arr_slice)], return_counts=True
            )
            if len(values) == 0:
                return np.nan  # Return NaN if there are no valid positive numbers
            max_count = np.max(counts)
            max_freq_values = values[counts == max_count]
            return np.max(max_freq_values)  # In case of tie, return the largest number

        # Apply the function along the last axis
        result = np.apply_along_axis(get_most_frequent, -1, arr)

        return result

    def divide_image_into_cubes(
        self,
        image: np.ndarray,
        cube_size: int = CUBE_SIZE,
        overlap_size: int = OVERLAP_PREDICTION_SIZE,
    ):
        _, H, W  = image.shape
        step_h, step_w = 0, 0
        step_size = cube_size - overlap_size

        cubes = []
        pos_cubes = []

        while step_h < H - cube_size:
            while step_w < W - cube_size: 
                cube = image[:, step_h:step_h+cube_size, step_w:step_w+cube_size]
                cubes.append(cube)
                pos_cubes.append((step_h, step_w))
                step_w += step_size
            cube = image[:, step_h:step_h+cube_size, -cube_size:]
            pos_cubes.append((step_h, -cube_size))
            step_w = 0
            step_h += step_size

        while step_w < W - cube_size:
            cube = image[:, -cube_size:, step_w:step_w+cube_size]
            cubes.append(cube)
            pos_cubes.append((-cube_size, step_w))
            step_w += step_size
        cube = image[:, -cube_size:, -cube_size:] 
        cubes.append(cube)
        pos_cubes.append((-cube_size, -cube_size))
        
        cubes = np.array(cubes)

        return cubes, pos_cubes

    def prep_extended_cubes(
        self,
        prob_cubes,
        im_height,
        im_width,
        pos_cubes,
        cube_size = CUBE_SIZE,
        border_effect_padding = BORDER_EFFECT_PADDING,
    ) -> np.ndarray:
        """
        Documenter
        Corriger la prise en compte des cubes sur le bord
        """
        n_cubes, n_channels, _, _ = prob_cubes.shape
        list_extended_cubes = []
        for i in range(n_cubes):
            extended_cube = np.zeros((n_channels, im_height, im_width))
            h_corner, w_corner = pos_cubes[i]

            left_padding = border_effect_padding
            right_padding = cube_size - border_effect_padding
            extended_cube[
                :, 
                h_corner+left_padding:h_corner+right_padding, 
                w_corner+left_padding:w_corner+right_padding
                ] = prob_cubes[
                    i,
                    :,
                    left_padding:right_padding, 
                    left_padding:right_padding]
            list_extended_cubes.append(extended_cube)

        return list_extended_cubes
    
    def merge_prob_cubes(
        self,
        prob_cubes,
        im_height,
        im_width,
        pos_cubes,
        cube_size = CUBE_SIZE,
        border_effect_padding = BORDER_EFFECT_PADDING,
    ):
        """
        Documenter
        Ajout des masques ?"""
        list_extended_cubes = self.prep_extended_cubes(prob_cubes, im_height, im_width, pos_cubes, cube_size, border_effect_padding)

        sum_prob_array = np.sum(list_extended_cubes, axis=0)
        nb_pred_array = np.round(np.sum(sum_prob_array, axis=0))

        prob_array = np.divide(sum_prob_array, nb_pred_array)

        return prob_array, nb_pred_array

    def predict_image_file(
        self, file_path: str
    ) -> Tuple[np.ndarray, dict]:
        """
        Predict on a file.

        Args:
            file_path (str): Path to the file.

        Returns:
            np.ndarray: Predictions.
            dict: Profile of the input raster.
        """
        with rasterio.open(file_path) as src:
            image = src.read()
            profile = src.profile
        limit_im_size = 1.5 * CUBE_SIZE #faire une constante globale avec le overlap et modifier cette ligne
        _, h, w = image.shape
        if h < limit_im_size and w < limit_im_size:
            pred_array, best_prob_array = self.predict(image)
            nb_pred_array = np.ones_like(pred_array).squeeze(0)
        else:
            cubes, pos_cubes = self.divide_image_into_cubes(image)
            print('Predicting ...')
            prob_cubes = self.get_logits_cubes(cubes)
            print('All cubes predicted')
            prob_array, nb_pred_array = self.merge_prob_cubes(prob_cubes, h, w, pos_cubes)
            print('Predictions merged')
            prob_array = torch.from_numpy(prob_array)
            prob_array = prob_array.unsqueeze(0)
            pred_array, best_prob_array = self.postprocess_output(prob_array)

        return pred_array, best_prob_array, nb_pred_array, profile

    def save_prediction(
        self, prediction: np.ndarray, profile: dict, output_path: str
    ) -> None:
        """
        Save the prediction as a raster.

        Args:
            prediction (np.ndarray): Predicted classes.
            profile (dict): Profile of the input raster.
            output_path (str): Path to save the prediction.
        """
        profile.update(dtype=rasterio.uint8, count=1)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(prediction.astype(rasterio.uint8), 1)

    def extract_shp(
        self, prediction: np.ndarray, profile: dict, output_path: str
    ) -> gpd.GeoDataFrame:
        """
        Extract the shapefile from the prediction.

        Args:
            prediction (np.ndarray): Predicted classes.
            profile (dict): Profile of the input raster.
            output_path (str): Path to save the shapefile.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame of the shapefile.
        """
        mask = prediction != 255
        shapes_generator = rasterio.features.shapes(
            mask.astype(np.int16), mask=mask, transform=profile["transform"]
        )
        geometries = [shape(geom) for geom, val in shapes_generator]
        gdf = gpd.GeoDataFrame({"geometry": geometries}, crs=profile["crs"])

        # Save the GeoDataFrame as a shapefile
        gdf.to_file(output_path)

        return gdf
