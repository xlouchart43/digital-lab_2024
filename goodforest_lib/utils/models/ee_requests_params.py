import logging
from os.path import isdir
from re import match


class EERequestParams:
    def __init__(
        self,
        ee_project_name: str,
        saving_location: str,
        gcs_bucket_name: str,
        logger: logging.Logger = None,
    ):
        if saving_location not in ["local", "gcs"]:
            raise ValueError("The saving location must be either 'local' or 'gcs'.")
        self.ee_project_name = ee_project_name
        self.saving_location = saving_location
        self.gcs_bucket_name = gcs_bucket_name
        self.logger = logger


class TrainingParams(EERequestParams):
    def __init__(
        self,
        ee_project_name: str,
        saving_location: str,
        gcs_bucket_name: str,
        start_date: str,
        end_date: str,
        destination_folder: str,
        max_cloud_cover: int,
        step_cloud_cover: int,
        logger: logging.Logger = None,
    ):
        if not (
            match(r"\d{4}-\d{2}-\d{2}", start_date)
            and match(r"\d{4}-\d{2}-\d{2}", end_date)
        ):
            error_message = "The dates must be of the form YYYY-MM-DD."
            logging.error(error_message)
            raise ValueError(error_message)
        if start_date > end_date:
            error_message = "The start date must be before the end date."
            logging.error(error_message)
            raise ValueError(error_message)
        max_cloud_cover = abs(int(max_cloud_cover))
        step_cloud_cover = abs(int(step_cloud_cover))

        self.start_date = start_date
        self.end_date = end_date
        self.destination_folder = destination_folder
        self.max_cloud_cover = max_cloud_cover
        self.step_cloud_cover = step_cloud_cover

        super().__init__(ee_project_name, saving_location, gcs_bucket_name, logger)


class PredictionParams(EERequestParams):
    def __init__(
        self,
        source_folder: str,
        before_date: str,
        max_date_diff_in_days: int,
        ee_project_name: str,
        saving_location: str,
        gcs_bucket_name: str,
        buffer: float,
        filter_black_images: bool,
        logger: logging.Logger = None,
    ):
        # Check the arguments
        if not isdir(source_folder):
            error_message = f"The source folder {source_folder} does not exist."
            logging.error(error_message)
            raise FileNotFoundError(error_message)
        if not (match(r"\d{4}-\d{2}-\d{2}", before_date)):
            error_message = "The date must be of the form YYYY-MM-DD."
            logging.error(error_message)
            raise ValueError(error_message)
        max_date_diff_in_days = abs(int(max_date_diff_in_days))

        self.source_folder = source_folder
        self.before_date = before_date
        self.max_date_diff_in_days = max_date_diff_in_days
        self.buffer = buffer
        self.filter_black_images = filter_black_images
        super().__init__(ee_project_name, saving_location, gcs_bucket_name, logger)

        # Initialize the visited tiles
        self.visited_tiles = []

    def add_tile_to_visited(self, tile_id: str) -> None:
        self.visited_tiles.append(tile_id)


class FordeadParams(EERequestParams):
    def __init__(
        self,
        source_folder: str,
        destination_folder: str,
        before_date: str,
        date_diff_in_days: int,
        ee_project_name: str,
        saving_location: str,
        gcs_bucket_name: str,
        buffer: float,
        filter_black_images: bool,
        threshold: float,
        logger: logging.Logger = None,
    ):
        # Check the arguments
        if not isdir(source_folder):
            error_message = f"The source folder {source_folder} does not exist."
            logging.error(error_message)
            raise FileNotFoundError(error_message)
        if not (match(r"\d{4}-\d{2}-\d{2}", before_date)):
            error_message = "The date must be of the form YYYY-MM-DD."
            logging.error(error_message)
            raise ValueError(error_message)
        date_diff_in_days = abs(int(date_diff_in_days))

        self.destination_folder = destination_folder
        self.source_folder = source_folder
        self.before_date = before_date
        self.date_diff_in_days = date_diff_in_days
        self.buffer = buffer
        self.filter_black_images = filter_black_images
        self.threshold = threshold
        super().__init__(ee_project_name, saving_location, gcs_bucket_name, logger)
