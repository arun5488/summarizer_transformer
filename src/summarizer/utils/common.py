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

    
    def chunk_tokenized_input(text, tokenizer, max_input_length, max_length, stride):
        try:
            logger.info("inside chunk_tokenized_input method")
            tokens = tokenizer(text, truncation = False, return_attention_mask = True)
            input_ids = tokens["input_ids"]
            logger.info(f"length of input ids:{input_ids}")
            attention_mask = tokens["attention_mask"]
            max_length = max_input_length
            logger.info(f"Max length of input for model: {max_length}")
            stride = stride
            logger.info(f"stride value:{stride}")
            chunks = []
            for i in range(0, len(input_ids), stride):
                chunk_ids = input_ids[i:i+max_length]
                chunk_mask = attention_mask[i:i+max_length]
                if len(chunk_ids) == 0:
                    break
                chunks.append({
                    "input_ids": chunk_ids,
                    "attention_mask": chunk_mask
                })
            return chunks 
        except Exception as e:
            logger.error(f"Error occured inside chunk_tokenized_input method:{e}")
            raise e