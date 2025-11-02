from src.summarizer import logger
from src.summarizer import constants as const
from src.summarizer.config.configuration import DataIngestionConfig
from datasets import load_dataset

class DataIngestion:
    def __init__(self, config = DataIngestionConfig):
        logger.info("Initialized DataIngestion")
        self.config = config
    
    def load_dataset(self):
        try:
            logger.info("inside load_dataset method")
            dataset_name = self.config.dataset_name
            logger.info(f"dataset name: {dataset_name}")
            dataset_version = self.config.dataset_version
            logger.info(f"dataset version: {dataset_version}")
            dataset_split = self.config.dataset_split
            logger.info(f"dataset split: {dataset_split}")
            dataset = load_dataset(dataset_name, dataset_version, split=dataset_split)
            logger.info(f"dataset loaded successfully. length: {len(dataset)}")
            logger.info(f"dataset type: {type(dataset)}")
            logger.info("splitting the dataset into train and test")
            dataset = dataset.train_test_split(train_size=const.TRAIN_TEST_SPLIT)
            logger.info(f"dataset splitted successfully. length: {len(dataset["train"])}")
            return dataset
        except Exception as e:
            logger.error(f"error occured inside load_dataset method: {e}")
            raise e
    
    def save_dataset_to_disk(self, dataset):
        try:
            logger.info("inside save_dataset_to_disc")
            dataset_path = self.config.dataset_path
            logger.info(f"saving dataset to; {dataset_path}")
            dataset.save_to_disk(dataset_path)
            logger.info("dataset saved to local folder")
        except Exception as e:
            logger.error(f"Error occured inside save_dataset_to_disc: {e}")
            raise e
    
    def initiate_data_ingestion(self):
        try:
            logger.info("Inside initiate_data_ingestion method")
            dataset = self.load_dataset()
            self.save_dataset_to_disk(dataset=dataset)
        except Exception as e:
            logger.error(f"Error occured inside initiate_data_ingestion:{e}")
            raise e
