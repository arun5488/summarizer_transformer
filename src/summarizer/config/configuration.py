from src.summarizer import logger
from src.summarizer.entity import DataIngestionConfig, DataTransformationConfig, ModelevaluationConfig, ModelTrainerConfig
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
    def get_data_transformation_config(self) -> DataTransformationConfig:
        try:
            logger.info("Inside get_data_transformation_config")
            config = self.config.data_transformation
            logger.info("Creating folder for data transformation")
            create_directories([config.root_dir])
            return DataTransformationConfig(
                root_dir=config.root_dir,
                dataset_path = config.dataset_path,
                tokenized_data_path = config.tokenized_data_path,
                tokenizer = config.tokenizer,
                checkpoint = self.params.checkpoint,
                max_target_length = self.params.max_target_length,
                max_input_length = self.params.max_input_length,
                stride = self.params.stride
            )
        except Exception as e:
            logger.error(f"Error occured inside get_data_transformation_config: {e}")
            raise e
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        try:
            logger.info("inside get_model_trainer_config")
            config = self.config.model_trainer
            logger.info("creating model_trainer directory")
            create_directories([config.root_dir])
            logger.info("returning the model_trainer config")
            return ModelTrainerConfig(
                root_dir=config.root_dir,
                model_path = config.model_path,
                tokenized_data_path = config.tokenized_data_path,
                tokenizer = config.tokenizer,
                checkpoint = self.params.checkpoint
            )
        except Exception as e:
            logger.error(f"Error occured inside get_model_trainer_config:{e}")
            raise e
    
    def get_model_evaluation_config(self) -> ModelevaluationConfig:
        try:
            logger.info("inside get_model_evaluation_config method")
            config = self.config.model_evaluation
            logger.info("creating model_evaluation root directory")
            create_directories([config.root_dir])
            return ModelevaluationConfig(
                root_dir= config.root_dir,
                model_path = config.model_path,
                dataset_path = config.dataset_path,
                tokenizer = config.tokenizer
            )
        except Exception as e:
            logger.error(f"Error occured inside get_model_evaluation_config method:{e}")
            raise e
