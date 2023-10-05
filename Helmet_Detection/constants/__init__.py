
import torch

ARTIFACTS_DIR: str = "artifacts"

"""
Data Ingestion Related Configuration
"""

DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
RAW_FILE_NAME: str = 'helmet'
DATA_FILE_NAME = "helmet-data.zip"
DATA_DOWNLOAD_URL: str = "https://drive.google.com/file/d/1oYBdYcQKPGPfqj7n4is-10k17vL6Cmlp/view?usp=sharing"
DATA_INGESTION_ARTIFACTS_DIR = "DataIngestionArtifacts"
DATA_INGESTION_TRAIN_DIR: str = "train"
DATA_INGESTION_TEST_DIR: str = "test"
DATA_INGESTION_VALID_DIR: str = "valid"
ANNOTATIONS_COCO_JSON_FILE = "_annotations.coco.json"


"""
Data Transformation Related Configuration
"""

INPUT_SIZE = 416
HORIZONTAL_FLIP = 0.3
VERTICAL_FLIP = 0.3
RANDOM_BRIGHTNESS_CONTRAST = 0.1
COLOR_JITTER = 0.1
BBOX_FORMAT = "coco"
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_FEATURE_STORE_DIR: str = "feature_store"
DATA_TRANSFORMATION_TEST_DIR = 'Test'
DATA_TRANSFORMATION_TRAIN_DIR = 'Train'
DATA_TRANSFORMATION_TRAIN_FILE_NAME = "train.pkl"
DATA_TRANSFORMATION_TEST_FILE_NAME = "test.pkl"
DATA_TRANSFORMATION_TEST_SPLIT = "test"
DATA_TRANSFORMATION_TRAIN_SPLIT = "train"


"""
Model Trainer Related Configuration
"""

MODEL_TRAINER_DIR_NAME: str = "model_trainer"
TRAINED_MODEL_DIR = "Model"
TRAINED_MODEL_NAME = "best.pt"
TRAINED_BATCH_SIZE = 2
TRAINED_SHUFFLE = False
TRAINED_NUM_WORKERS = 1
EPOCH = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


"""
Model Evaluation Related Configuration
"""

MODEL_EVALUATION_DIR_NAME: str = "model_evaluation"
MODEL_EVALUATION_FILE_NAME: str = "loss.csv"
