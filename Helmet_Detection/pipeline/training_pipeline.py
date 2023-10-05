
import sys
from Helmet_Detection.logging import logger
from Helmet_Detection.exception import HelmetException
from Helmet_Detection.components.data_ingestion import DataIngestion
from Helmet_Detection.components.data_transformation import DataTransformation
from Helmet_Detection.entity.config_entity import (DataIngestionConfig, DataTransformationConfig)
from Helmet_Detection.entity.artifacts_entity import (DataIngestionArtifact, DataTransformationArtifact)


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logger.info("Initiated Data Ingestion stage (Reached TrainPipeline class)...")
            data_ingestion = DataIngestion(self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logger.info("Finished Data Ingestion stage (Exited TrainPipeline class)...")

            return data_ingestion_artifact
        except Exception as e:
            raise HelmetException(e, sys)

    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataTransformationArtifact:
        try:
            logger.info("Initiated Data Transformation stage (Reached TrainPipeline class)...")
            data_transformation = DataTransformation(data_ingestion_artifact=data_ingestion_artifact,
                                                     data_transformation_config=self.data_transformation_config)
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logger.info("Exited Data Transformation stage (Exited TrainPipeline class)...")

            return data_transformation_artifact
        except Exception as e:
            raise HelmetException(e, sys) from e

    def run_pipeline(self) -> None:
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact)
        except Exception as e:
            raise HelmetException(e, sys)
