from src.summarizer import logger
from src.summarizer.components.data_transformation import DataTransformation
from src.summarizer.config.configuration import ConfigurationManager

class DataTransformationPipeline:
    def __init__(self):
        logger.info("initialized DataTransformationPipeline")
        self.config = ConfigurationManager().get_data_transformation_config()
    
    def initiate_data_transformation_pipeline(self):
        try:
            logger.info("Inside initiate_data_transformation_pipeline")
            DataTransformation(self.config).initiate_data_transformation_for_training()
        except Exception as e:
            logger.error(f"Error occured inside initiate_data_transformation_pipeline method:{e}")
            raise e