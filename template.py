
import os
import logging
from pathlib import Path

logging.basicConfig(level= logging.INFO, format= '[%(asctime)s]: %(message)s')

PROJECT_NAME = "Helmet_Detection"
LIST_OF_FILES = [
    "app.py",
    "setup.py",
    "Dockerfile",
    "data/.gitkeep",
    "requirements.txt",
    "docker-compose.yml",
    "research/trials.ipynb",
    f"{PROJECT_NAME}/__init__.py",
    f"{PROJECT_NAME}/components/__init__.py",
    f"{PROJECT_NAME}/components/data_ingestion.py",
    f"{PROJECT_NAME}/components/data_transformation.py",
    f"{PROJECT_NAME}/components/model_evaluation.py",
    f"{PROJECT_NAME}/components/model_pusher.py",
    f"{PROJECT_NAME}/components/model_trainer.py",
    f"{PROJECT_NAME}/utils/__init__.py",
    f"{PROJECT_NAME}/utils/main_utils.py",
    f"{PROJECT_NAME}/logging/__init__.py",
    f"{PROJECT_NAME}/exception/__init__.py",
    f"{PROJECT_NAME}/configuration/__init__.py",
    f"{PROJECT_NAME}/configuration/s3_operations.py",
    f"{PROJECT_NAME}/pipeline/__init__.py",
    f"{PROJECT_NAME}/entity/__init__.py",
    f"{PROJECT_NAME}/entity/config_entity.py",
    f"{PROJECT_NAME}/entity/artifacts_entity.py",
    f"{PROJECT_NAME}/constants/__init__.py",
    f"{PROJECT_NAME}/ml/__init__.py"]


for filepath in LIST_OF_FILES:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if os.path.exists(filepath):
        logging.info(f"File : <{filename}> already exists.")
        continue

    if filedir != "":
        if not (os.path.exists(filedir)):
            os.makedirs(filedir, exist_ok=True)
            logging.info(f"Creating Directory <{filedir}> for the File : {filename}.")
        else:
            logging.info(f"Existing Directory <{filedir}> for the File : {filename}.")

    if not(os.path.exists(filepath)):
        with open(filepath, 'w') as f:
            pass
        logging.info(f"Creating Empty File : <{filepath}>")
