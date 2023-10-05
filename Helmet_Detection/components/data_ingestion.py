
import os
import sys
import gdown
import zipfile
from Helmet_Detection.logging import logger
from Helmet_Detection.exception import HelmetException
from Helmet_Detection.entity.config_entity import DataIngestionConfig
from Helmet_Detection.entity.artifacts_entity import DataIngestionArtifact


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise HelmetException(e, sys)

    def download_data(self):
        try:
            zip_file_name = self.data_ingestion_config.data_file_name
            dataset_url = self.data_ingestion_config.data_download_url
            zip_save_dir = self.data_ingestion_config.data_ingestion_dir
            os.makedirs(zip_save_dir, exist_ok=True)
            zip_file_path = os.path.join(zip_save_dir, zip_file_name)
            logger.info(f"Downloading started :\n"
                        f"Source URL: {dataset_url}\n"
                        f"Target Path: {zip_file_path}")

            file_id = dataset_url.split("/")[-2]
            prefix = "https://drive.google.com/uc?/export=download&id="
            gdown.download(prefix+file_id, zip_file_path)
            logger.info(f"Downloading finished :\n"
                        f"Source URL: {dataset_url}\n"
                        f"Save Path: {zip_file_path}")

            return zip_file_path
        except Exception as e:
            raise HelmetException(e, sys)

    def extract_zip_file(self, zip_file_path: str) -> str:
        try:
            feature_store_path = self.data_ingestion_config.feature_store_file_path
            os.makedirs(feature_store_path)

            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                logger.info(f"Extracting zip file\n"
                            f"Source Path: {zip_file_path}\n"
                            f"Save Path: {feature_store_path}")
                zip_ref.extractall(feature_store_path)

            logger.info(f"Extracted zip file\n"
                        f"Source Path: {zip_file_path}\n"
                        f"Save Path: {feature_store_path}")

            return feature_store_path
        except Exception as e:
            raise HelmetException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        logger.info("Initiated Data Ingestion stage (Reached DataIngestion class)...")
        try:
            zip_file_name = self.data_ingestion_config.data_file_name
            zip_save_dir = self.data_ingestion_config.data_ingestion_dir

            if not os.path.exists(os.path.join(zip_save_dir, zip_file_name)):
                zip_file_path = self.download_data()
            else:
                zip_file_path = os.path.join(zip_save_dir, zip_file_name)
                logger.info(f"Downloaded file already available at {zip_file_path}")

            if not os.path.exists(self.data_ingestion_config.feature_store_file_path):
                feature_store_path = self.extract_zip_file(zip_file_path)
            else:
                feature_store_path = self.data_ingestion_config.feature_store_file_path
                logger.info(f"Extracted file already available at {feature_store_path}")

            data_ingestion_artifact = DataIngestionArtifact(test_file_path=os.path.join(feature_store_path, "test"),
                                                            train_file_path=os.path.join(feature_store_path, "train"),
                                                            valid_file_path=os.path.join(feature_store_path, "valid"))
            logger.info("Finished Data Ingestion stage (Exited DataIngestion class)...")

            return data_ingestion_artifact
        except Exception as e:
            raise HelmetException(e, sys)
