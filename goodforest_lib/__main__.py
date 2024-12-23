import argparse

from .load_data import (
    main_load_S2_for_prediction,
    main_load_S2_for_training,
    main_load_S2_for_fordead,
)
from .load_data.parser import (
    parser_load_S2_data_prediction,
    parser_load_S2_data_training,
    parser_load_S2_data_fordead,
)
from .preprocess_data import main_cubify_images, main_filter_cubes, main_labelize_images
from .preprocess_data.parser import (
    parser_cubify_images,
    parser_filter_cubes,
    parser_labelize_images_from_shp,
)
from .utils import main_download_from_gcs, main_upload_to_gcs
from .utils.parser import parser_download_from_gcs, parser_upload_to_gcs
from .visualization import main_visualize_h5_cubes, main_visualize_tif
from .visualization.parser import parser_visualize_h5_cubes, parser_visualize_tif


def main():
    parser = argparse.ArgumentParser(description="GoodForest CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Define subparsers for different modules
    add_subparser(
        subparsers,
        "load-data",
        "Load data",
        [
            parser_load_S2_data_training,
            parser_load_S2_data_prediction,
            parser_load_S2_data_fordead,
        ],
    )
    add_subparser(
        subparsers,
        "preprocess-data",
        "Preprocess data",
        [parser_labelize_images_from_shp, parser_cubify_images, parser_filter_cubes],
    )
    add_subparser(
        subparsers,
        "visualize",
        "Visualize data",
        [parser_visualize_tif, parser_visualize_h5_cubes],
    )
    add_subparser(
        subparsers,
        "utils",
        "Utilities",
        [parser_download_from_gcs, parser_upload_to_gcs],
    )

    args = parser.parse_args()

    if args.command == "visualize":
        if args.subcommand == "tif":
            main_visualize_tif(
                source=args.source,
                multiple=args.multiple,
                compression=args.compression,
                colors=args.colors,
                recursive=args.recursive,
            )
        elif args.subcommand == "h5-cubes":
            main_visualize_h5_cubes(
                source_file=args.source_file,
                nb_per_grid=args.nb_per_grid,
                compression=args.compression,
            )

    elif args.command == "load-data":
        if args.subcommand == "S2-train":
            main_load_S2_for_training(
                shp_file_path=args.shp_file_path,
                aoi_indices=args.aoi_indices,
                start_date=args.start_date,
                end_date=args.end_date,
                ee_project_name=args.ee_project_name,
                saving_location=args.saving_location,
                destination_folder=args.destination_folder,
                gcs_bucket_name=args.gcs_bucket_name,
                clear_credentials=args.clear_credentials,
                max_cloud_cover=args.max_cloud_cover,
                step_cloud_cover=args.step_cloud_cover,
                custom_buffer=args.custom_buffer,
            )
        elif args.subcommand == "S2-predict":
            main_load_S2_for_prediction(
                source_folder=args.source_folder,
                before_date=args.before_date,
                max_date_diff_in_days=args.max_date_diff_in_days,
                ee_project_name=args.ee_project_name,
                saving_location=args.saving_location,
                gcs_bucket_name=args.gcs_bucket_name,
                clear_credentials=args.clear_credentials,
                buffer=args.buffer,
                filter_black_images=args.filter_black_images,
            )
        elif args.subcommand == "S2-fordead":
            main_load_S2_for_fordead(
                source_folder=args.source_folder,
                destination_folder=args.destination_folder,
                before_date=args.before_date,
                date_diff_in_days=args.date_diff_in_days,
                ee_project_name=args.ee_project_name,
                saving_location=args.saving_location,
                gcs_bucket_name=args.gcs_bucket_name,
                clear_credentials=args.clear_credentials,
                buffer=args.buffer,
                filter_black_images=args.filter_black_images,
            )
    elif args.command == "preprocess-data":
        if args.subcommand == "labelize":
            main_labelize_images(
                source_folder=args.source_folder,
                destination_folder=(
                    args.destination_folder
                    if args.destination_folder
                    else args.source_folder
                ),
                gdf_path=args.gdf_path,
                num_classes=args.num_classes,
                year=args.year,
                insects=args.insects,
                recursive=args.recursive,
                suffix=args.suffix,
            )
        elif args.subcommand == "cubify":
            main_cubify_images(
                source_folder=args.source_folder,
                destination_folder=(
                    args.destination_folder
                    if args.destination_folder
                    else args.source_folder
                ),
                cube_size=args.cube_size,
                recursive=args.recursive,
                filename=args.filename,
            )
        elif args.subcommand == "filter":
            main_filter_cubes(
                source_file=args.source_file,
                destination_path=(
                    args.destination_path if args.destination_path else args.source_file
                ),
                threshold=args.threshold,
                delete_original=args.delete_original,
            )
    elif args.command == "utils":
        if args.subcommand == "download-from-gcs":
            main_download_from_gcs(
                gcs_project_name=args.gcs_project_name,
                gcs_bucket_name=args.gcs_bucket_name,
                source_path=args.source_path,
                destination_path=args.destination_path,
                extension=args.extension,
            )
        elif args.subcommand == "upload-to-gcs":
            main_upload_to_gcs(
                gcs_project_name=args.gcs_project_name,
                gcs_bucket_name=args.gcs_bucket_name,
                source_path=args.source_path,
                destination_path=args.destination_path,
                extension=args.extension,
            )


def add_subparser(
    main_subparser: argparse._SubParsersAction, name: str, help: str, parsers_list: list
) -> None:
    """Add a subparser for a module."""
    parser = main_subparser.add_parser(name, help=help)
    subparser = parser.add_subparsers(dest="subcommand", required=True)
    for parser in parsers_list:
        parser(subparser)


if __name__ == "__main__":
    main()
