from src.summarizer import logger
from src.summarizer.components.data_ingestion import DataIngestion
from src.summarizer.config.configuration import ConfigurationManager

class DataIngestionPipeline:
    def __init__(self):
        logger.info("Initialized DataIngestionPipeline")
        self.config = ConfigurationManager().get_data_ingestion_config()
    
    def initiate_data_ingestion_pipeline(self):
        try:
            logger.info("inside initiate_data_ingestion_pipeline")
            DataIngestion(self.config).initiate_data_ingestion()
            logger.info("data_ingestion pipeline completed")
        except Exception as e:
            logger.error(f"Error occured inside initiate_data_ingestion_pipeline:{e}")
            raise e