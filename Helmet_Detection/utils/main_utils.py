
import dill
import base64
import os, sys
import logging
from Helmet_Detection.logging import logger
from Helmet_Detection.exception import HelmetException


def save_object(file_path: str, obj: object) -> None:
    logger.info("Entered the method:`save_object` from utils")
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

        logger.info("Exited the method:`save_object` from utils")
    except Exception as e:
        raise HelmetException(e, sys) from e


def load_object(file_path: str) -> object:
    logger.info("Entered the method:`load_object` from utils")
    try:
        with open(file_path, 'rb') as file_obj:
            obj = dill.load(file_obj)

        logger.info("Exited the method:`load_object` from utils")
        return obj
    except Exception as e:
        raise HelmetException(e, sys) from e


def image_to_base64(image):
    with open(image, 'rb') as img_file:
        encoded_string = base64.b64encode(img_file.read())

    return encoded_string


