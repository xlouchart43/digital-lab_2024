import argparse

from ..config import (
    CUBE_SIZE,
    DEFAULT_INSECTS,
    FILENAME_CUBES_FILE,
    MAX_THRESHOLD_BLACK_PIXELS,
    SUFFIX_ANNOTATED,
)


def parser_labelize_images_from_shp(subparsers: argparse._SubParsersAction) -> None:
    """
    Add the parser for the labelize command.

    Args:
        subparsers (argparse._SubParsersAction): Subparsers.
    """
    parser = subparsers.add_parser(
        "labelize",
        help="Labelize a tif images from an annotated shapefile to prepare the training and save it as a npy file.",
    )
    parser.add_argument(
        "-s",
        "--source-folder",
        type=str,
        required=True,
        help="Folder containing the images to labelize.",
    )
    parser.add_argument(
        "-d",
        "--destination-folder",
        type=str,
        help="Destination folder where to save the resulting labeled images (default is the source folder).",
    )
    parser.add_argument(
        "-g",
        "--gdf-path",
        type=str,
        required=True,
        help="Path to the shapefile containing the annotations.",
    )
    parser.add_argument(
        "-y",
        "--year",
        type=int,
        required=True,
        help="Year of the annotations to extract.",
    )
    parser.add_argument(
        "-n",
        "--num-classes",
        type=int,
        required=True,
        help="Number of classes to labelize.",
    )
    parser.add_argument(
        "-i",
        "--insects",
        type=str,
        nargs="+",
        required=True,
        default=DEFAULT_INSECTS,
        help=f"Insects to extract from the annotations (default: {', '.join(DEFAULT_INSECTS)}).",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        dest="recursive",
        default=True,
        action="store_true",
        help="Enable recursive search for images in subdirectories (default: enable recursiver search).",
    )
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Disable recursive search for images in subdirectories.",
    )
    parser.add_argument(
        "-suf",
        "--suffix",
        type=str,
        default=SUFFIX_ANNOTATED,
        help=f"Suffix to add to the name of the saved files (default: {SUFFIX_ANNOTATED}).",
    )


def parser_cubify_images(subparsers: argparse._SubParsersAction) -> None:
    """
    Add the parser for the cubify command.

    Args:
        subparsers (argparse._SubParsersAction): Subparsers.
    """
    parser = subparsers.add_parser(
        "cubify", help="Cubify tif images to prepare the training."
    )
    parser.add_argument(
        "-s",
        "--source-folder",
        type=str,
        required=True,
        help="Folder containing the images to cubify.",
    )
    parser.add_argument(
        "-d",
        "--destination-folder",
        type=str,
        help="Destination folder where to save the resulting cubes (default is the source folder).",
    )
    parser.add_argument(
        "-c",
        "--cube-size",
        type=int,
        default=CUBE_SIZE,
        help=f"Size of the cubes to extract from the images (default: {CUBE_SIZE}).",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        dest="recursive",
        default=True,
        action="store_true",
        help="Enable recursive search for images in subdirectories (default: enable recursiver search).",
    )
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Disable recursive search for images in subdirectories.",
    )
    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        default=FILENAME_CUBES_FILE,
        help=f"Name of the file to save the cubes (default: {FILENAME_CUBES_FILE}).",
    )


def parser_filter_cubes(subparsers: argparse._SubParsersAction) -> None:
    """
    Add the parser for the filter command.

    Args:
        subparsers (argparse._SubParsersAction): Subparsers.
    """
    parser = subparsers.add_parser(
        "filter",
        help="Filter cubes from .h5 file to remove the ones without enough vegetation and save it as .h5 file.",
    )
    parser.add_argument(
        "-s",
        "--source-file",
        type=str,
        required=True,
        help="Path to the .h5 file containing the cubes to filter.",
    )
    parser.add_argument(
        "-d",
        "--destination-path",
        type=str,
        help="Destination path where to save the resulting filtered cubes (default is the source file parent folder).",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=MAX_THRESHOLD_BLACK_PIXELS,
        help=f"Maximum threshold of black pixels (other than vegetation) to filter the cubes (in range 0-1) (default: {MAX_THRESHOLD_BLACK_PIXELS}).",
    )
    parser.add_argument(
        "--delete-original",
        default=False,
        action="store_true",
        help="Whether to delete the former .h5 file after the computation (default: False).",
    )
