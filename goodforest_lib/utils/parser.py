from argparse import _SubParsersAction

from ..config.constants import EE_PROJECT_NAME, GCS_BUCKET_NAME


def parser_download_from_gcs(subparsers: _SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "download-from-gcs",
        description="Download a file or folder from Google Cloud Storage.",
    )
    parser.add_argument(
        "-p",
        "--gcs-project-name",
        type=str,
        default=EE_PROJECT_NAME,
        help=f"Name of the Google project. (default: {EE_PROJECT_NAME})",
    )
    parser.add_argument(
        "-b",
        "--gcs-bucket-name",
        type=str,
        default=GCS_BUCKET_NAME,
        help=f"Name of the Google bucket. (default: {GCS_BUCKET_NAME})",
    )
    parser.add_argument(
        "-s",
        "--source-path",
        type=str,
        required=True,
        help="Path to the file or folder in the bucket.",
    )
    parser.add_argument(
        "-d",
        "--destination-path",
        type=str,
        required=True,
        help="Path to the destination file or folder.",
    )
    parser.add_argument(
        "-e",
        "--extension",
        type=str,
        nargs="+",
        help="Extensions of the files to download (default is all).",
    )


def parser_upload_to_gcs(subparsers: _SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "upload-to-gcs",
        description="Upload a file or folder to Google Cloud Storage.",
    )
    parser.add_argument(
        "-p",
        "--gcs-project-name",
        type=str,
        default=EE_PROJECT_NAME,
        help=f"Name of the Google project. (default: {EE_PROJECT_NAME})",
    )
    parser.add_argument(
        "-b",
        "--gcs-bucket-name",
        type=str,
        default=GCS_BUCKET_NAME,
        help=f"Name of the Google bucket. (default: {GCS_BUCKET_NAME})",
    )
    parser.add_argument(
        "-s",
        "--source-path",
        type=str,
        required=True,
        help="Path to the file or folder to upload.",
    )
    parser.add_argument(
        "-d",
        "--destination-path",
        type=str,
        required=True,
        help="Path to the destination file or folder in the bucket.",
    )
    parser.add_argument(
        "-e",
        "--extension",
        type=str,
        nargs="+",
        help="Extensions of the files to upload (default is all).",
    )
