"""
Useful functions to connect to Google Cloud Storage.
"""

from os import system

from google.auth.exceptions import DefaultCredentialsError, RefreshError
from google.cloud import storage


def connect_to_bucket(gcs_project_name: str, gcs_bucket_name: str) -> storage.Bucket:
    """Connects to the bucket."""
    try:
        storage_client = storage.Client(project=gcs_project_name)
        bucket = storage_client.bucket(gcs_bucket_name)
    except RefreshError as e:
        system("gcloud auth application-default login")
        return connect_to_bucket(gcs_project_name, gcs_bucket_name)
    except DefaultCredentialsError as e:
        system("gcloud auth application-default login")
        return connect_to_bucket(gcs_project_name, gcs_bucket_name)
    except Exception as e:
        raise Exception(f"❌ An error occurred while connecting to the bucket: {e}")

    print(f"Connected to bucket {gcs_bucket_name}.")
    return bucket
