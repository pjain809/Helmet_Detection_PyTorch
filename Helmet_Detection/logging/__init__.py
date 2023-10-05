
import os, sys
import logging
from datetime import datetime
from from_root import from_root

LOGS_FILE_NAME = f"Log @ {datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOGS_DIRS_NAME = f"Log @ {datetime.now().strftime('%m_%d_%Y')}"

os.makedirs(os.path.join(from_root(), "Logs", LOGS_DIRS_NAME), exist_ok=True)
LOGS_FILE_PATH = os.path.join(from_root(), "Logs", LOGS_DIRS_NAME, LOGS_FILE_NAME)

logging.basicConfig(handlers={logging.FileHandler(LOGS_FILE_PATH), logging.StreamHandler(sys.stdout)},
                    level=logging.INFO,
                    format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s")

logger = logging.getLogger("Helmet_Detection_PyTorch")
