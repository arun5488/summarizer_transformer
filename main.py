from src.summarizer import logger
from src.summarizer.pipeline.training_pipeline import TrainingPipeline

if __name__ == "__main__":
    try:
        logger.info("inside main.py")
        logger.info("Initiating Training Pipeline")
        TrainingPipeline().initiate_training_pipeline()
    except Exception as e:
        logger.error(f"Error occured inside main:{e}")
        raise e