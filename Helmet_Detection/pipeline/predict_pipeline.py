
import os
import io
import sys
import base64
import torch
from PIL import Image
from io import BytesIO
from torchvision import transforms
from Helmet_Detection.constants import *
from Helmet_Detection.logging import logger
from torchvision.utils import draw_bounding_boxes
from Helmet_Detection.exception import HelmetException
from Helmet_Detection.entity.config_entity import ModelTrainerConfig


class PredictPipeline:
    def __init__(self, model_trainer_config: ModelTrainerConfig = ModelTrainerConfig()):
        self.trained_model_path = model_trainer_config.trained_model_path

    def image_loader(self, image_bytes):
        try:
            image = Image.open(io.BytesIO(image_bytes))
            convert_tensor = transforms.ToTensor()
            tensor_image = convert_tensor(image)

            image_int = torch.tensor(tensor_image*255, dtype=torch.uint8)
            return tensor_image, image_int
        except Exception as e:
            raise HelmetException(e, sys) from e

    def get_model(self):
        try:
            logger.info("Loading best model for prediction (Reached PredictPipeline class)...")
            best_model_path = self.trained_model_path
            logger.info("Loaded best model for prediction (Reached PredictPipeline class)...")
            return best_model_path

        except Exception as e:
            raise HelmetException(e, sys) from e

    def predict(self, best_model_path: str, image_tensor, image_int_tensor):
        logger.info("Predicting initiated on input image (Reached PredictPipeline class)...")
        try:
            model = torch.load(best_model_path, map_location=torch.device(DEVICE))
            model.eval()

            with torch.no_grad():
                prediction = model([image_tensor.to(DEVICE)])
                pred = prediction[0]

            bbox_tensor = draw_bounding_boxes(image_int_tensor, pred['boxes'][pred['scores'] > 0.8],
                                              [PREDICTION_CLASSES[i] for i in pred['labels'][pred['scores'] > 0.8].tolist()],
                                              width=4).permute(0, 2, 1)

            transform = transforms.ToPILImage()
            img = transform(bbox_tensor)
            buffered = BytesIO()
            img.save(buffered, format='JPEG')
            img_str = base64.b64encode(buffered.getvalue())

            logger.info("Predicting finished on input image (Reached PredictPipeline class)...")
            return img_str
        except Exception as e:
            raise HelmetException(e, sys) from e

    def run_pipeline(self, data):
        try:
            image, image_int = self.image_loader(data)
            logger.info(f"{image.shape}\n{image_int.shape}\n")

            best_model_path = self.get_model()
            detected_image = self.predict(best_model_path, image, image_int)
            return detected_image
        except Exception as e:
            raise HelmetException(e, sys) from e
