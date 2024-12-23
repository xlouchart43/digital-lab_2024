"""
Upload a file or folder to Google Cloud Storage.
"""

import os

from google.cloud.storage import Bucket

from .connect_gcs import connect_to_bucket


def upload_file(
    bucket: Bucket, source_blob_name: str, destination_file_name: str
) -> None:
    """Uploads a file to the bucket."""
    try:
        print(f"⚙️ Uploading local file {source_blob_name} to GCS storage.")
        blob = bucket.blob(destination_file_name)
        blob.upload_from_filename(source_blob_name)
    except Exception as e:
        raise Exception(f"❌ An error occurred while uploading the local file: {e}")
    print(f"✅ Uploaded successfully!")


def upload_folder(
    bucket: Bucket,
    source_folder_name: str,
    destination_folder_name: str,
    extensions: list,
) -> None:
    """Upload all files in a folder to the bucket."""
    for root, _, files in os.walk(source_folder_name):
        for file in files:
            if extensions and file.split(".")[-1] not in extensions:
                continue

            source_blob_name = os.path.join(root, file)
            destination_blob_name = "/".join(
                list(filter(lambda x: x != "", destination_folder_name.split("/")))
                + list(
                    filter(
                        lambda x: x != "",
                        source_blob_name.split(source_folder_name)[1].split("/"),
                    )
                )
            )
            try:
                print(
                    f"⚙️ Uploading local file {source_blob_name} to GCS storage {destination_blob_name}."
                )
                blob = bucket.blob(destination_blob_name)
                blob.upload_from_filename(source_blob_name)
                print(f"✅ Uploaded successfully!")
            except Exception as e:
                print(f"❌ An error occurred while uploading the local file: {e}")


def main(
    gcs_project_name: str,
    gcs_bucket_name: str,
    source_path: str,
    destination_path: str,
    extension: list,
) -> None:
    bucket = connect_to_bucket(gcs_project_name, gcs_bucket_name)

    # Check if the source is a file or a folder
    if os.path.isfile(source_path):
        upload_file(bucket, source_path, destination_path)
    else:
        upload_folder(bucket, source_path, destination_path, extension)
