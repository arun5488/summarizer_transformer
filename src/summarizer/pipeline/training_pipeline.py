from src.summarizer import logger
from src.summarizer.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from src.summarizer.pipeline.data_transformation_pipeline import DataTransformationPipeline

class TrainingPipeline:
    def __init__(self):
        logger.info("Initialized Training Pipeline")
    
    def initiate_training_pipeline(self):
        try:
            logger.info("Inside initiate_training_pipeline")
            logger.info("Data Ingestion stage started")
            # DataIngestionPipeline().initiate_data_ingestion_pipeline()
            logger.info("Data Ingestion stage completed")
            logger.info("Data Transformation stage started")
            DataTransformationPipeline().initiate_data_transformation_pipeline()
            logger.info("Data Transformation stage completed")
        except Exception as e:
            logger.error(f"Error occured inside initiate_training_pipeline method: {e}")
            raise e