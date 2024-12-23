import argparse
import os

from ..config.constants import (
    BUFFER_PREDICTION,
    EE_PROJECT_NAME,
    GCS_BUCKET_NAME,
    MAX_THRESHOLD_CLOUDY_PIXELS,
    PERIOD_S2_IMAGES,
    FORDEAD_PERIOD_S2_IMAGES,
    SAVING_LOCATION,
    STEP_CLOUD_COVER,
    TILE_OVERLAP_KM,
    TILE_SIZE_KM,
)


def parser_load_S2_data_training(subparsers: argparse._SubParsersAction) -> None:
    """Add the parser for the S2-train command.

    Args:
        subparsers (argparse._SubParsersAction): The subparsers.
    """
    parser = subparsers.add_parser(
        "S2-train",
        help="Load Sentinel2 data thanks to Google Earth Engine for training.",
    )
    parser.add_argument(
        "-s",
        "--shp-file-path",
        type=str,
        help="Shapefile of the geometry to fetch (either Points or Polygons).",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date for the Sentinel-2 data (of type YYYY-MM-DD).",
        required=True,
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date for the Sentinel-2 data (of type YYYY-MM-DD).",
        required=True,
    )
    parser.add_argument(
        "-e",
        "--ee-project-name",
        type=str,
        default=EE_PROJECT_NAME,
        help=f"Google Earth Engine project name. (default: {EE_PROJECT_NAME}).",
    )
    parser.add_argument(
        "--saving-location",
        type=str,
        default=SAVING_LOCATION,
        choices=["local", "gcs"],
        help=f"Where to save the images. (default: {SAVING_LOCATION}).",
    )
    parser.add_argument(
        "-d",
        "--destination-folder",
        type=str,
        required=True,
        help="Folder where to save the images (for both local and GCS saving).",
    )
    parser.add_argument(
        "-g",
        "--gcs-bucket-name",
        type=str,
        default=GCS_BUCKET_NAME,
        help=f"Google Cloud Storage bucket name. (default: {GCS_BUCKET_NAME}).",
    )
    parser.add_argument(
        "-c",
        "--clear-credentials",
        action="store_true",
        help="Clear existing Earth Engine credentials.",
    )
    parser.add_argument(
        "--custom-buffer",
        type=int,
        default=int(1000 * (TILE_SIZE_KM - TILE_OVERLAP_KM) / 2),
        help=f"Custom buffer to add around the shapefile geometry (default: {int(1000 * (TILE_SIZE_KM - TILE_OVERLAP_KM) / 2)}.",
    )
    parser.add_argument(
        "--max-cloud-cover",
        type=int,
        default=MAX_THRESHOLD_CLOUDY_PIXELS,
        help=f"Maximum cloud cover percentage for the images (default: {MAX_THRESHOLD_CLOUDY_PIXELS}%%).",
    )
    parser.add_argument(
        "--step-cloud-cover",
        type=int,
        default=STEP_CLOUD_COVER,
        help=f"Step for the cloud cover percentage (default: {STEP_CLOUD_COVER}%%).",
    )
    parser.add_argument(
        "--aoi-indices",
        type=int,
        nargs="+",
        help="Indices of the AOIs to fetch (default: all).",
    )


def parser_load_S2_data_prediction(subparsers: argparse._SubParsersAction) -> None:
    """Add the parser for the S2-predict command.

    Args:
        subparsers (argparse._SubParsersAction): The subparsers.
    """
    parser = subparsers.add_parser(
        "S2-predict",
        help="Load Sentinel2 data thanks to Google Earth Engine for prediction.",
    )
    parser.add_argument(
        "-s",
        "--source_folder",
        required=True,
        type=str,
        help="Folder containing the shapefiles that describe the area of interest.",
    )
    parser.add_argument(
        "--before-date",
        type=str,
        help="Date before which to fetch the Sentinel-2 data (of type YYYY-MM-DD).",
        required=True,
    )
    parser.add_argument(
        "--max-date-diff-in-days",
        type=int,
        default=PERIOD_S2_IMAGES,
        help=f"Maximum number of days between the date of the image and the before-date (default: {PERIOD_S2_IMAGES}).",
    )
    parser.add_argument(
        "-e",
        "--ee-project-name",
        type=str,
        default=EE_PROJECT_NAME,
        help=f"Google Earth Engine project name (default: {EE_PROJECT_NAME}).",
    )
    parser.add_argument(
        "--saving-location",
        type=str,
        default=SAVING_LOCATION,
        choices=["local", "gcs"],
        help=f"Where to save the images (default: {SAVING_LOCATION}).",
    )
    parser.add_argument(
        "-g",
        "--gcs-bucket-name",
        type=str,
        default=GCS_BUCKET_NAME,
        help=f"Google Cloud Storage bucket name (default: {GCS_BUCKET_NAME}).",
    )
    parser.add_argument(
        "-c",
        "--clear-credentials",
        action="store_true",
        help="Clear existing Earth Engine credentials.",
    )
    parser.add_argument(
        "-b",
        "--buffer",
        type=int,
        default=BUFFER_PREDICTION,
        help=f"Buffer to add around the shapefile geometry (default: {BUFFER_PREDICTION}).",
    )
    parser.add_argument(
        "-f",
        "--filter-black-images",
        action="store_true",
        help="Filter images with too much black pixels.",
    )


def parser_load_S2_data_fordead(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "S2-fordead",
        help="Load Sentinel2 data thanks to Google Earth Engine for fordead notebook demo.",
    )
    parser.add_argument(
        "-s",
        "--source_folder",
        required=True,
        type=str,
        help="Folder containing the shapefiles that describe the area of interest.",
    )
    parser.add_argument(
        "-d",
        "--destination-folder",
        type=str,
        required=True,
        help="Folder where to save the images (for both local and GCS saving).",
    )
    parser.add_argument(
        "--before-date",
        type=str,
        help="Date before which to fetch the Sentinel-2 data (of type YYYY-MM-DD).",
        required=True,
    )
    parser.add_argument(
        "--date-diff-in-days",
        type=int,
        default=FORDEAD_PERIOD_S2_IMAGES,
        help=f"Number of days between the date of the first image acquisition and the last date of acquisition (default: {FORDEAD_PERIOD_S2_IMAGES}).",
    )
    parser.add_argument(
        "-e",
        "--ee-project-name",
        type=str,
        default=EE_PROJECT_NAME,
        help=f"Google Earth Engine project name (default: {EE_PROJECT_NAME}).",
    )
    parser.add_argument(
        "--saving-location",
        type=str,
        default=SAVING_LOCATION,
        choices=["local", "gcs"],
        help=f"Where to save the images (default: {SAVING_LOCATION}).",
    )
    parser.add_argument(
        "-g",
        "--gcs-bucket-name",
        type=str,
        default=GCS_BUCKET_NAME,
        help=f"Google Cloud Storage bucket name (default: {GCS_BUCKET_NAME}).",
    )
    parser.add_argument(
        "-c",
        "--clear-credentials",
        action="store_true",
        help="Clear existing Earth Engine credentials.",
    )
    parser.add_argument(
        "-b",
        "--buffer",
        type=int,
        default=BUFFER_PREDICTION,
        help=f"Buffer to add around the shapefile geometry (default: {BUFFER_PREDICTION}).",
    )
    parser.add_argument(
        "-f",
        "--filter-black-images",
        action="store_true",
        help="Filter images with too much black pixels.",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.4,
        help="Threshold for the black pixels (default: 0.4).",
    )
