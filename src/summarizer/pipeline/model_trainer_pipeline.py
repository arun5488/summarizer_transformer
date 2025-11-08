from src.summarizer import logger
from src.summarizer.config.configuration import ConfigurationManager
from src.summarizer.components.model_trainer import ModelTrainer

class ModelTrainerPipeline:
    def __init__(self):
        logger.info("Initialized ModelTrainerPipeline")
        self.config = ConfigurationManager().get_model_trainer_config()
    
    def initiate_model_training_pipeline(self):
        try:
            logger.info("inside initiate_model_training method")
            ModelTrainer(self.config).initiate_model_training()
        except Exception as e:
            logger.error(f"Error occured inside initiate_model_training: {e}")
            raise e