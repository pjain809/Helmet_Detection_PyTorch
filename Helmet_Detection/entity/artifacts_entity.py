
from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    test_file_path: str
    train_file_path: str
    valid_file_path: str


@dataclass
class DataTransformationArtifact:
    transformed_test_object: str
    transformed_train_object: str
    number_of_classes: int


@dataclass
class ModelTrainerArtifact:
    trained_model_path: str
