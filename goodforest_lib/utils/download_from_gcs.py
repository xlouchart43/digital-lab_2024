"""
Download a file or folder from Google Cloud Storage.
"""

import os

from google.cloud.storage import Bucket

from .connect_gcs import connect_to_bucket


def download_file(bucket: Bucket, source_blob_name: str, destination_file_name: str):
    """Downloads a blob from the bucket."""
    try:
        print(
            f"⚙️ Downloading storage object {source_blob_name} from GCS to local file."
        )
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
    except Exception as e:
        raise Exception(
            f"❌ An error occurred while downloading the storage object: {e}"
        )
    print(f"✅ Downloaded successfully!")


def download_folder(
    bucket: Bucket, source_folder_name, destination_folder_name: str, extensions: list
):
    """Downloads all blobs in a folder from the bucket."""
    try:
        blobs = bucket.list_blobs(prefix=source_folder_name)
    except Exception as e:
        raise Exception(f"❌ An error occurred while listing blobs in the folder: {e}")

    for blob in blobs:
        if extensions and blob.name.split(".")[-1] not in extensions:
            continue

        blob_path = blob.name.split("/")
        print(f"⚙️ Downloading storage object {blob_path[-1]} from GCS to local file.")
        if len(blob_path) > 1:
            temp_destination_folder = "/".join(
                destination_folder_name.split("/") + blob_path[:-1]
            )
            os.makedirs(temp_destination_folder, exist_ok=True)

        destination_file_name = temp_destination_folder + "/" + blob_path[-1]
        try:
            blob.download_to_filename(destination_file_name)
        except Exception as e:
            print(f"❌ An error occurred while downloading the storage object: {e}")
        print(f"✅ Downloaded successfully!")


def main(
    gcs_project_name: str,
    gcs_bucket_name: str,
    source_path: str,
    destination_path: str,
    extension: str,
) -> None:
    bucket = connect_to_bucket(gcs_project_name, gcs_bucket_name)

    # Check if the source is a file or a folder
    if os.path.isfile(source_path):
        download_file(bucket, source_path, destination_path)
    else:
        download_folder(bucket, source_path, destination_path, extension)
