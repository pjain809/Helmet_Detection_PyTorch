
import os
from datetime import datetime
from dataclasses import dataclass
from Helmet_Detection.constants import *


@dataclass
class TrainingPipelineConfig:
    artifacts_dir: str = ARTIFACTS_DIR


training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()


@dataclass
class DataIngestionConfig:
    data_file_name: str = DATA_FILE_NAME
    data_download_url: str = DATA_DOWNLOAD_URL
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifacts_dir, DATA_INGESTION_DIR_NAME)
    feature_store_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR)


ingestion_config: DataIngestionConfig = DataIngestionConfig()


@dataclass
class DataTransformationConfig:
    features_dir: str = ingestion_config.feature_store_file_path
    data_transform_dir: str = os.path.join(training_pipeline_config.artifacts_dir, DATA_TRANSFORMATION_DIR_NAME)
    train_transform_artifacts_dir: str = os.path.join(data_transform_dir, DATA_TRANSFORMATION_TRAIN_DIR)
    test_transform_artifacts_dir: str = os.path.join(data_transform_dir, DATA_TRANSFORMATION_TEST_DIR)
    train_transform_object_path: str = os.path.join(train_transform_artifacts_dir, DATA_TRANSFORMATION_TRAIN_FILE_NAME)
    test_transform_object_path: str = os.path.join(test_transform_artifacts_dir, DATA_TRANSFORMATION_TEST_FILE_NAME)
    train_split: str = DATA_TRANSFORMATION_TRAIN_SPLIT
    test_split: str = DATA_TRANSFORMATION_TEST_SPLIT


@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str = os.path.join(training_pipeline_config.artifacts_dir, MODEL_TRAINER_DIR_NAME)
    trained_model_dir: str = os.path.join(model_trainer_dir, TRAINED_MODEL_DIR)
    trained_model_path: str = os.path.join(trained_model_dir, TRAINED_MODEL_NAME)
    BATCH_SIZE: int = TRAINED_BATCH_SIZE
    SHUFFLE: bool = TRAINED_SHUFFLE
    NUM_WORKERS: int = TRAINED_NUM_WORKERS
    EPOCH: int = EPOCH
    DEVICE: str = DEVICE
