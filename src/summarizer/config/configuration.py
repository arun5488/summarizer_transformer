from src.summarizer import logger
from src.summarizer.entity import DataIngestionConfig
from src.summarizer.utils.common import read_yaml, create_directories
from src.summarizer import constants as const

class ConfigurationManager:
    def __init__(self):
        logger.info("Initialized ConfigurationManager")
        logger.info("reading config yaml")
        self.config = read_yaml(const.CONFIG_YAML)
        logger.info("reading param yaml")
        self.params = read_yaml(const.PARAMS_YAML)
        logger.info("reading schema yaml")
        self.schema = read_yaml(const.SCHEMA_YAML)

        logger.info("creating root directory")
        create_directories([self.config.root_dir])
        logger.info(f"{self.config.root_dir} folder created")
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        try:
            logger.info("inside get_data_ingestion_config mmethod")
            config = self.config.data_ingestion
            logger.info("creating root directory for data_ingestion")
            create_directories([config.root_dir])
            return DataIngestionConfig(
                root_dir= config.root_dir,
                dataset_name= config.dataset_name,
                dataset_path= config.dataset_path,
                dataset_version = config.dataset_version,
                dataset_split = config.dataset_split
            )

        except Exception as e:
            logger.error(f"Error occured inside get_data_ingestion_config: {e}")
            raise e
