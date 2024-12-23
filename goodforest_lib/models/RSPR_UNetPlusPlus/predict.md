# Predict Module

## Description
The `Predict` class provides functionality to make predictions using a trained model on images, cubes, or raster files. It also offers methods for saving predictions as rasters and extracting shapefiles from the predicted outputs. The class is designed to operate on PyTorch models, supporting CPU and GPU computation.

## Methodology

1. **Initialization**:
   - The class is initialized with a PyTorch model, the number of predicted classes and an optional device (CPU or GPU). The model is set to evaluation mode during initialization, ensuring that it is ready for inference.

2. **Image Prediction**:
   - The `predict` method takes as input an image as a NumPy array. It's first run through the model using the`get_logits` method and then postprocessed using the `postprocess_output` method, by applying `softmax` to obtain probabilities and then `max` to get both the max probability and the predicted class associated to it.

4. **Cube Prediction**:
   - The `predict_cubes` method is designed for making predictions on 3D image cubes. It follows the same steps as image prediction but operates on cubes, ensuring flexibility for 3D datasets.

5. **File-Based Prediction**:
   - The `predict_image_file` method allows prediction on a raster image file. It reads the image and its profile (metadata) using the `rasterio` library, makes predictions, and returns both the predicted classes and the original raster profile.

6. **Saving Predictions**:
   - The `save_prediction` method saves the predicted classes as a raster file, updating the original raster profile to fit the prediction format.

7. **Shapefile Extraction**:
   - The `extract_shp` method extracts a shapefile from the predicted raster classes. It uses the `rasterio.features.shapes` function to generate geometries from the predicted classes and saves these geometries as a shapefile.

## Architecture

### Key Components:
- **Model Inference**: The class utilizes a PyTorch model for predictions, processing the input images or cubes through the model to obtain logits.
- **Postprocessing**: The model output is transformed into class predictions using `softmax` and `argmax`.
- **Raster File Support**: `rasterio` is used for reading raster files and handling geospatial data profiles.
- **Shapefile Extraction**: The predictions can be converted into shapefiles using `shapely` and `geopandas` for further geospatial analysis.

### Example Usage

```python
import torch
from models import RSPRUNetPlusPlus 
from predict import Predict  


model = RSPRUNetPlusPlus(num_classes)  
predictor = Predict(model=model, device=torch.device("cuda"))

prediction, profile = predictor.predict_image_file("path/to/image.tif")

# Save prediction
predictor.save_prediction(prediction, profile, "path/to/output.tif")

# Extract shapefile from prediction
gdf = predictor.extract_shp(prediction, profile, "path/to/output.shp")
