import logging
import os
from datetime import datetime

# creating log file name
LOG_FILE_NAME = f"{datetime.now().strftime('%m%d%Y_%H_%M_%S')}.log"

# creating log file directory
LOG_FILE_DIR = os.path.join(os.getcwd(),"logs")

# Creating log folder if not available
os.makedirs(LOG_FILE_DIR,exist_ok=True)

# log file path
LOG_FILE_PATH = os.path.join(LOG_FILE_DIR,LOG_FILE_NAME)

logging.basicConfig(
    filename= LOG_FILE_PATH,
    format= "[ %(asctime)s ] %(lineno)d %(filename)s - %(levelname)s - %(message)s",
    level= logging.DEBUG
)