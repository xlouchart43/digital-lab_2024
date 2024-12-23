# Preprocess Data

## Description

This module aims to preprocess the data after they've been downloaded from the `load-data` module. The data is preprocessed to prepare it for the training of the model. The module provides three subcommands: `labelize`, `cubify`, and `filter`.

`labelize` allows you to labelize the data thanks to an annotated dataset to prepare a training set. `cubify` allows you to tile the data into cubes to prepare the data for the model. `filter` allows you to filter the cubes with few vegetation to improve the model's performance in the training phase.

This labelization tasks aims to generate data that will be inputs for the deep-learning model. The data should have the following formatting at the end : `(NB_OF_CUBES, NB_CHANNELS, CUBE_HEIGHT, CUBE_WIDTH)`. The channels are the concatenation of the sentinel2 channels and the vegetation indices. After labelization in training phase, a final channel is added with type `np.uint8`, allowing to define a class per value.

## Methodology

### Labelize

The `labelize` subcommand allows you to labelize the data thanks to an annotated dataset. The annotated dataset is a pickle file that contains annotated polygons. For now, the annotated dataset allowed is the one provided by British Columbia's Ministry of Forests, Lands, Natural Resource Operations, and Rural Development, available [here](https://catalogue.data.gov.bc.ca/dataset/pest-infestation-polygons). For other dataset, you'll need to dupplicate this module to create a new one, adapted to your dataset format. To make sure it works in the deeplearning model, consider adding the class value as a `np.uint8` as the last channel of the cubes.

### Cubify

The `cubify` subcommand allows you to tile the data into cubes. The cubes are created by tiling the data into small squares of a fixed size `CUBE_SIZE`$= 256$. The cubes are then stored in an h5 file.

### Filter

The `filter` subcommand allows you to filter the cubes with few vegetation. The cubes are filtered based on the percentage of black pixels in the cube. If the percentage of black pixels is higher than a threshold, the cube is removed from the dataset. By default, the threshold is set to `MAX_THRESHOLD_BLACK_PIXELS`$= 80\%$.