
ARTIFACTS_DIR: str = "artifacts"

"""
Data Ingestion Related Configuration
"""

DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
RAW_FILE_NAME:str = 'helmet'
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
