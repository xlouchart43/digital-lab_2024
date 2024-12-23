from argparse import _SubParsersAction

from ..config.constants import DEFAULT_COLORS_VISUALIZATION, DEFAULT_COMPRESSION


def parser_visualize_tif(subparsers: _SubParsersAction) -> None:
    visualize_parser = subparsers.add_parser("tif", help="Visualize tif images")
    visualize_parser.add_argument(
        "-m", "--multiple", action="store_true", help="Display multiple images"
    )
    visualize_parser.add_argument(
        "-s", "--source", type=str, required=True, help="Path to the tif file or folder"
    )
    visualize_parser.add_argument(
        "-c",
        "--compression",
        type=int,
        default=DEFAULT_COMPRESSION,
        help="Compress images to the specified width (ex: 512). By default, "
        + f"{'the width is resized to ' + str(DEFAULT_COMPRESSION) + 'px' if DEFAULT_COMPRESSION is not None else 'no compression is applied'}.",
    )
    visualize_parser.add_argument(
        "--colors",
        type=str,
        default=DEFAULT_COLORS_VISUALIZATION,
        choices=["rgb", "nir"],
        help="Define the type of image to display. By default, the color {DEFAULT_COLORS_VISUALIZATION} is used.",
    )
    visualize_parser.add_argument(
        "-r",
        "--recursive",
        dest="recursive",
        action="store_true",
        default=True,
        help="Enable recursive search for tif images in the source folder. (default)",
    )
    visualize_parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Disable recursive search for tif images in the source folder.",
    )


def parser_visualize_h5_cubes(subparsers: _SubParsersAction) -> None:
    visualize_parser = subparsers.add_parser("h5-cubes", help="Visualize pkl cubes")
    visualize_parser.add_argument(
        "-s",
        "--source-file",
        type=str,
        required=True,
        help="Path to the .h5 file containing the cubes to visualize.",
    )
    visualize_parser.add_argument(
        "-n",
        "--nb-per-grid",
        type=int,
        default=9,
        help="Number of cubes per grid.",
    )
    visualize_parser.add_argument(
        "-c",
        "--compression",
        type=int,
        default=DEFAULT_COMPRESSION,
        help="Compress images to the specified width (ex: 512). By default, "
        + f"{'the width is resized to ' + str(DEFAULT_COMPRESSION) + 'px' if DEFAULT_COMPRESSION is not None else 'no compression is applied'}.",
    )
