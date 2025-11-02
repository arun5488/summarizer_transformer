from src.summarizer import logger
import yaml
import os
import sys
from pathlib import Path
from box import ConfigBox

def read_yaml(file_path: Path):
    try:
        logger.info("Inside read_yaml method")
        with open(file_path) as file:
            content = yaml.safe_load(file)
            return ConfigBox(content)
    except Exception as e:
        logger.error(f"Error occured inside read_yaml method: {e}")
        raise e

def create_directories(path_to_folders: list):
    try:
        logger.info("inside create_directories method")
        for path in path_to_folders:
            os.makedirs(path, exist_ok=True)
            logger.info(f"Folder created in {path}")
    except Exception as e:
        logger.error(f"Error occured inside create_directories method:{e}")
        raise e