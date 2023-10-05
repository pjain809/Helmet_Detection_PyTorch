
import sys
from Helmet_Detection.logging import logger
from Helmet_Detection.exception import HelmetException
from Helmet_Detection.components.data_ingestion import DataIngestion
from Helmet_Detection.components.data_transformation import DataTransformation
from Helmet_Detection.components.model_evaluation import ModelEvaluation
from Helmet_Detection.components.model_trainer import ModelTrainer
from Helmet_Detection.entity.config_entity import (DataIngestionConfig, DataTransformationConfig, ModelTrainerConfig,
                                                   ModelEvaluationConfig)
from Helmet_Detection.entity.artifacts_entity import (DataIngestionArtifact, DataTransformationArtifact,
                                                      ModelTrainerArtifact, ModelEvaluationArtifact)


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()

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

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            logger.info("Initiated Model Trainer stage (Reached TrainPipeline class)...")
            model_trainer = ModelTrainer(data_transformation_artifact=data_transformation_artifact,
                                         model_trainer_config=self.model_trainer_config)
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logger.info("Exited Model Trainer stage (Exited TrainPipeline class)...")

            return model_trainer_artifact
        except Exception as e:
            raise HelmetException(e, sys) from e

    def start_model_evaluation(self,
                               model_trainer_artifact: ModelTrainerArtifact,
                               data_transformation_artifact: DataTransformationArtifact):
        try:
            logger.info("Initiated Model Evaluation stage (Reached TrainPipeline class)...")
            model_evaluation = ModelEvaluation(data_transformation_artifact=data_transformation_artifact,
                                               model_trainer_artifact=model_trainer_artifact,
                                               model_evaluation_config=self.model_evaluation_config)
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
            logger.info("Exited Model Evaluation stage (Exited TrainPipeline class)...")

            return model_evaluation_artifact
        except Exception as e:
            raise HelmetException(e, sys) from e

    def run_pipeline(self) -> None:
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact)
            model_evaluation_artifact = self.start_model_evaluation(data_transformation_artifact=data_transformation_artifact,
                                                                    model_trainer_artifact=model_trainer_artifact)
        except Exception as e:
            raise HelmetException(e, sys)
