# Config

This directory contains configuration files for the GoodForest library.

You can freely modify these files to adapt the library to your needs.

## Specific files

The `sentinel2_bands.py` file contains the bands used in the Sentinel-2 satellite imagery. In order to apply the compression on scale 0-255, emprirical measures have been made to establish an affine transformation from scale 0-10000 to 0-255 specifically for each band. The `sentinel2_bands.py` file contains the affine transformation for each band and the way how to reverse it. It can indeed be useful to compute new vegetation indices or to apply a specific transformation on the bands.

The `vegetation_indices.py` file contains the vegetation indices that can be computed from the Sentinel-2 satellite imagery. The vegetation indices are computed from the bands of the satellite imagery. The `vegetation_indices.py` file contains the formula to compute each vegetation index.