
import os
import sys
import math
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision import models
from torch.utils.data import DataLoader
from Helmet_Detection.logging import logger
from Helmet_Detection.exception import HelmetException
from Helmet_Detection.utils.main_utils import load_object
from Helmet_Detection.entity.config_entity import ModelTrainerConfig
from Helmet_Detection.ml.models.model_optimiser import model_optimiser
from Helmet_Detection.entity.artifacts_entity import DataTransformationArtifact, ModelTrainerArtifact
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights


class ModelTrainer:
    def __init__(self,
                 data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):

        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifact = data_transformation_artifact

    def train(self, model, optimizer, loader, device, epoch):
        try:
            model.to(device)
            model.train()
            all_losses = []
            all_losses_dict = []

            for images, targets in tqdm(loader):
                images = list(image.to(device) for image in images)
                targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                loss_dict_append = {k: v.item() for k, v in loss_dict.items()}
                loss_value = losses.item()

                all_losses.append(loss_value)
                all_losses_dict.append(loss_dict_append)

                if not math.isfinite(loss_value):
                    print(f"Loss is {loss_value}, stopping training")  # train if loss becomes infinity
                    print(loss_dict)
                    sys.exit(1)

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
            all_losses_dict = pd.DataFrame(all_losses_dict)  # for printing

            print(
                "Epoch {}, lr: {:.6f}, loss: {:.6f}, loss_classifier: {:.6f}, loss_box: {:.6f}, loss_rpn_box: {:.6f}, loss_object: {:.6f}".format(
                    epoch, optimizer.param_groups[0]['lr'], np.mean(all_losses),
                    all_losses_dict['loss_classifier'].mean(),
                    all_losses_dict['loss_box_reg'].mean(),
                    all_losses_dict['loss_rpn_box_reg'].mean(),
                    all_losses_dict['loss_objectness'].mean()
                ))

        except Exception as e:
            raise HelmetException(e, sys) from e

    @staticmethod
    def collate_fn(batch):
        try:
            return tuple(zip(*batch))
        except Exception as e:
            raise HelmetException(e, sys) from e

    def initiate_model_trainer(self, ) -> ModelTrainerArtifact:
        if os.path.exists(self.model_trainer_config.trained_model_path):
            logger.info(f"Trained Model already available")

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_path=self.model_trainer_config.trained_model_path)
            logger.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        else:
            logger.info("Initiated Model Trainer stage (Reached ModelTrainer class)...")
            try:
                train_dataset = load_object(self.data_transformation_artifact.transformed_train_object)
                test_dataset = load_object(self.data_transformation_artifact.transformed_test_object)

                train_loader = DataLoader(train_dataset,
                                          batch_size=self.model_trainer_config.BATCH_SIZE,
                                          shuffle=self.model_trainer_config.SHUFFLE,
                                          num_workers=self.model_trainer_config.NUM_WORKERS,
                                          collate_fn=self.collate_fn)

                logger.info("Loaded training data loader object")
                model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1)
                logger.info("Loaded Faster RCNN model")

                in_features = model.roi_heads.box_predictor.cls_score.in_features  # we need to change the head
                model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features,
                                                                                               self.data_transformation_artifact.number_of_classes)
                optimiser = model_optimiser(model)
                logger.info("Loaded optimiser")

                self.train(model, optimiser, train_loader, self.model_trainer_config.DEVICE, 1)

                os.makedirs(self.model_trainer_config.trained_model_dir, exist_ok=True)
                torch.save(model, self.model_trainer_config.trained_model_path)
                logger.info(f"Saved the trained model")

                model_trainer_artifact = ModelTrainerArtifact(trained_model_path=self.model_trainer_config.trained_model_path)
                logger.info(f"Model trainer artifact: {model_trainer_artifact}")

                return model_trainer_artifact

            except Exception as e:
                raise HelmetException(e, sys) from e
