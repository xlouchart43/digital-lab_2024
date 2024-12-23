import os
import pickle

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.plot import show


def read_sentinel2(tif_path):
    with rasterio.open(tif_path) as src:
        image = src.read()
        profile = src.profile
    print(image.shape)
    return image, profile


def read_and_enhance_sentinel2(tif_path):
    with rasterio.open(tif_path) as src:
        image = src.read()
        profile = src.profile

    # Assuming bands 4, 3, 2 are R, G, B
    rgb = np.stack([image[3], image[2], image[1]], axis=0)

    # Enhance image visibility
    p2, p98 = np.percentile(rgb, (2, 98))
    rgb_stretched = (rgb - p2) / (p98 - p2)
    rgb_stretched = np.clip(rgb_stretched, 0, 1)

    return rgb_stretched, profile


def rasterize_polygons(shp_path, raster_profile):
    with open(shp_path, "rb") as f:
        gdf = pickle.load(f)
    gdf = gdf["geometry"].to_frame()
    gdf = gdf.to_crs(raster_profile["crs"])

    raster = rasterize(
        [(geometry, 1) for geometry in gdf.geometry],
        out_shape=(raster_profile["height"], raster_profile["width"]),
        transform=raster_profile["transform"],
        fill=0,
        all_touched=True,
        dtype=rasterio.uint8,
    )
    return raster


def create_visualization(sentinel_image, polygon_raster):
    # Create a 4-channel image (RGB + Alpha)
    rgba = np.zeros((4, sentinel_image.shape[1], sentinel_image.shape[2]))
    rgba[:3, :, :] = sentinel_image

    # Set alpha channel to semi-transparent where polygons exist
    rgba[3, :, :] = np.where(polygon_raster == 1, 1, 1)

    # # Create red mask for polygons
    red_mask = np.zeros_like(rgba)
    red_mask[0, :, :] = np.where(polygon_raster == 1, 1, 0)  # Red channel
    red_mask[3, :, :] = np.where(polygon_raster == 1, 0.5, 0)  # Alpha channel

    fig, ax = plt.subplots(figsize=(9, 9))

    # Display the Sentinel-2 image
    ax.imshow(np.transpose(rgba, (1, 2, 0)))

    # Overlay the red polygons
    ax.imshow(np.transpose(red_mask, (1, 2, 0)))

    ax.set_title("Sentinel-2 RGB Composite with Polygon Overlay")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def enhance_band(band):
    p2, p98 = np.percentile(band, (2, 98))
    return np.clip((band - p2) / (p98 - p2), 0, 1)


def create_visualization_bands(sentinel_image, polygon_raster):
    fig, axs = plt.subplots(3, 4, figsize=(9, 9))
    axs = axs.ravel()

    band_names = [
        "Coastal aerosol",
        "Blue",
        "Green",
        "Red",
        "Vegetation Red Edge 1",
        "Vegetation Red Edge 2",
        "Vegetation Red Edge 3",
        "NIR",
        "Narrow NIR",
        "Water vapour",
        "SWIR 1",
    ]

    for i in range(11):
        enhanced_band = enhance_band(sentinel_image[i])

        # Create an RGB image where the band is displayed in grayscale
        rgb = np.stack([enhanced_band, enhanced_band, enhanced_band])

        # Add alpha channel
        rgba = np.zeros((4, rgb.shape[1], rgb.shape[2]))
        rgba[:3, :, :] = rgb
        rgba[3, :, :] = 1

        # # Create red mask for polygons
        red_mask = np.zeros_like(rgba)
        red_mask[0, :, :] = np.where(polygon_raster == 1, 1, 0)  # Red channel
        red_mask[3, :, :] = np.where(polygon_raster == 1, 0.5, 0)  # Alpha channel

        axs[i].imshow(np.transpose(rgba, (1, 2, 0)), cmap="gray")
        axs[i].imshow(np.transpose(red_mask, (1, 2, 0)))
        axs[i].set_title(f"Band {i+1}: {band_names[i]}")
        axs[i].axis("off")

    # Remove the last (empty) subplot
    fig.delaxes(axs[-1])

    plt.tight_layout()
    plt.suptitle(
        "Sentinel-2 11-Band Visualization with Polygon Overlay", fontsize=16, y=1.02
    )
    plt.show()


def get_data(gdf_path: str, year: int, insects: list):

    with open(gdf_path, "rb") as f:
        gdf = pickle.load(f)
    gdf = gdf[(gdf["CAPTURE_YEAR"] == year) & (gdf["PEST_SPECIES_CODE"].isin(insects))]
    path = f'{"_".join(insects)}_{year}.pkl'
    if not os.path.exists(path):
        with open(path, "wb") as f:
            pickle.dump(gdf, f)
    return path


# Main execution
if __name__ == "__main__":
    sentinel2_path = "data/raw/Sentinel2_Export_0.tif"
    polygon_path = get_data("pest_infestation_poly.pkl", 2022, ["IBB"])
    polygo_raster = None
    bands = False
    if bands:
        sentinel_image, profile = read_sentinel2(sentinel2_path)
        polygon_raster = rasterize_polygons(polygon_path, profile)
        create_visualization_bands(sentinel_image, polygon_raster)
    else:
        sentinel_image, profile = read_and_enhance_sentinel2(sentinel2_path)
        polygon_raster = rasterize_polygons(polygon_path, profile)
        create_visualization(sentinel_image, polygon_raster)
