
import os
import sys
import math
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from Helmet_Detection.constants import *
from Helmet_Detection.logging import logger
from Helmet_Detection.exception import HelmetException
from Helmet_Detection.utils.main_utils import load_object
from Helmet_Detection.entity.config_entity import ModelEvaluationConfig
from Helmet_Detection.entity.artifacts_entity import (ModelTrainerArtifact, DataTransformationArtifact,
                                                      ModelEvaluationArtifact)


class ModelEvaluation:
    def __init__(self,
                 model_evaluation_config: ModelEvaluationConfig,
                 data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):

        self.model_evaluation_config = model_evaluation_config
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_artifact = model_trainer_artifact

    @staticmethod
    def collate_fn(batch):
        try:
            return tuple(zip(*batch))
        except Exception as e:
            raise HelmetException(e, sys) from e

    def evaluate(self, model, dataloader, device):
        try:
            model.to(device)
            all_losses = []
            all_losses_dict = []

            for images, targets in tqdm(dataloader):
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

                losses.backward()
            all_losses_dict = pd.DataFrame(all_losses_dict)

            print(f"loss: {np.mean(all_losses):.6f} \n"
                  f"loss_classifier: {all_losses_dict['loss_classifier'].mean():.6f}\n"
                  f"loss_box: {all_losses_dict['loss_box_reg'].mean():.6f}\n"
                  f"loss_rpn_box: {all_losses_dict['loss_rpn_box_reg'].mean():.6f}\n"
                  f"loss_object: {all_losses_dict['loss_objectness'].mean():.6f}")

            return all_losses_dict, np.mean(all_losses)
        except Exception as e:
            raise HelmetException(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            trained_model = torch.load(self.model_trainer_artifact.trained_model_path)
            test_dataset = load_object(self.data_transformation_artifact.transformed_test_object)
            test_loader = DataLoader(test_dataset,
                                     batch_size=self.model_evaluation_config.batch,
                                     shuffle=self.model_evaluation_config.shuffle,
                                     num_workers=self.model_evaluation_config.num_workers,
                                     collate_fn=self.collate_fn)

            logger.info("Loaded saved model..")
            trained_model = trained_model.to(DEVICE)
            all_losses_dict, all_losses = self.evaluate(trained_model, test_loader, device=DEVICE)
            os.makedirs(self.model_evaluation_config.model_evaluation_dir, exist_ok=True)
            all_losses_dict.to_csv(self.model_evaluation_config.model_evaluation_loss_path)

            model_evaluation_artifact = ModelEvaluationArtifact(
                all_losses_file_path=self.model_evaluation_config.model_evaluation_loss_path)

            return model_evaluation_artifact
        except Exception as e:
            raise HelmetException(e, sys) from e
