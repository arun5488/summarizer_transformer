from src.summarizer.components.model_evaluation import ModelEvaluation
from src.summarizer import logger
from src.summarizer.config.configuration import ConfigurationManager

class ModelEvaluationPipeline:
    def __init__(self):
        logger.info("Initialized ModelEvaluationPipeline")
        self.config = ConfigurationManager().get_model_evaluation_config()

    def initiate_model_evaluation_pipeline(self):
        try:
            logger.info("Inside initiate_model_evaluation_pipeline method")
            ModelEvaluation(self.config).initiate_model_evaluation()
        except Exception as e:
            logger.error(f"Error occured inside initiate_model_evaluation_pipeline:{e}")
            raise e
