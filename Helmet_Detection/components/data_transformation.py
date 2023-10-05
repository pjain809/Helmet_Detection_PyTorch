
import os
import sys
import albumentations as A
from pycocotools.coco import COCO
from Helmet_Detection.constants import *
from Helmet_Detection.logging import logger
from Helmet_Detection.utils.main_utils import save_object
from albumentations.pytorch import ToTensorV2
from Helmet_Detection.exception import HelmetException
from Helmet_Detection.ml.feature.helmet_detection import HelmetDetection
from Helmet_Detection.entity.config_entity import DataTransformationConfig
from Helmet_Detection.entity.artifacts_entity import DataIngestionArtifact, DataTransformationArtifact


class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig, data_ingestion_artifact: DataIngestionArtifact):
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifact = data_ingestion_artifact

    def number_of_classes(self):
        try:
            coco = COCO(os.path.join(self.data_ingestion_artifact.train_file_path, ANNOTATIONS_COCO_JSON_FILE))
            categories = coco.cats
            n_classes = len([item[1]['name'] for item in categories.items()])
            return n_classes
        except Exception as e:
            raise HelmetException(e, sys) from e

    def get_transform(self, train=False):
        try:
            if train:
                transform = A.Compose([A.Resize(height=INPUT_SIZE, width=INPUT_SIZE),
                                       A.HorizontalFlip(p=HORIZONTAL_FLIP),
                                       A.VerticalFlip(p=VERTICAL_FLIP),
                                       A.RandomBrightnessContrast(p=RANDOM_BRIGHTNESS_CONTRAST),
                                       A.ColorJitter(p=COLOR_JITTER),
                                       ToTensorV2()], bbox_params=A.BboxParams(format=BBOX_FORMAT))
            else:
                transform = A.Compose([A.Resize(height=INPUT_SIZE, width=INPUT_SIZE),
                                       ToTensorV2()], bbox_params=A.BboxParams(format=BBOX_FORMAT))
            return transform
        except Exception as e:
            raise HelmetException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logger.info("Initiated Data Transformation stage (Reached DataTransformation class)...")

            n_classes = self.number_of_classes()
            logger.info(f"Number of Classes: {n_classes}")

            os.makedirs(self.data_transformation_config.data_transform_dir, exist_ok=True)
            train_dataset = HelmetDetection(root=self.data_transformation_config.features_dir,
                                            split=self.data_transformation_config.train_split,
                                            transforms=self.get_transform(True))
            logger.info("Training dataset prepared")

            test_dataset = HelmetDetection(root=self.data_transformation_config.features_dir,
                                           split=self.data_transformation_config.test_split,
                                           transforms=self.get_transform(False))
            logger.info("Testing dataset prepared")

            save_object(self.data_transformation_config.train_transform_object_path, train_dataset)
            save_object(self.data_transformation_config.test_transform_object_path, test_dataset)
            logger.info("Train & Test transformed objects are saved.")

            data_transformation_artifact = DataTransformationArtifact(
                number_of_classes=n_classes,
                transformed_train_object=self.data_transformation_config.train_transform_object_path,
                transformed_test_object=self.data_transformation_config.test_transform_object_path)
            logger.info("Finished Data Transformation stage (Exited DataTransformation class)...")

            return data_transformation_artifact
        except Exception as e:
            raise HelmetException(e, sys) from e
