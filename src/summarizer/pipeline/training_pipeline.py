from src.summarizer import logger
from src.summarizer.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from src.summarizer.pipeline.data_transformation_pipeline import DataTransformationPipeline
from src.summarizer.pipeline.model_trainer_pipeline import ModelTrainerPipeline
from src.summarizer.cloud.s3syncer import S3Sync
from src.summarizer import constants as const
from src.summarizer.utils.common import read_yaml
from src.summarizer.pipeline.model_evaluation_pipeline import ModelEvaluationPipeline

class TrainingPipeline:
    def __init__(self):
        logger.info("Initialized Training Pipeline")
        self.s3syncer = S3Sync()
        self.config = read_yaml(const.CONFIG_YAML)
        self.artifacts_root = self.config.root_dir
        self.aws_bucket_url = f"s3://{const.AWS_BUCKET}/artifact/{self.artifacts_root}"
    
    def sync_artifacts_to_s3(self):
        try:
            logger.info("inside sync_artifacts_to_s3")
            self.s3syncer.sync_local_to_s3(local_dir_path=self.artifacts_root, aws_bucket_url=self.aws_bucket_url)
            logger.info("Data sync between artifacts folder and s3 bucket")
        except Exception as e:
            logger.error(f'Error occured inside sync_artifacts_to_s3:{e}')
            raise e    
    
    def sync_s3_to_artifacts(self):
        try:
            logger.info("Inside sync_s3_to_artifacts method")
            self.s3syncer.sync_s3_to_local(local_dir_path= self.artifacts_root, aws_bucket_url=self.aws_bucket_url)
            logger.info("data synced between s3 and local folder")
        except Exception as e:
            logger.error(f"Error occured inside sync_s3_to_artifacts:{e}")
            raise e
    

    def initiate_training_pipeline(self):
        try:
            if(const.RUN_FROM_LOCAL == True):
                logger.info("Initiate model evaluation stage")
                ModelEvaluationPipeline().initiate_model_evaluation_pipeline()
                logger.info("syncing local folder to s3")
                # self.sync_s3_to_artifacts()
                logger.info("s3 and artifacts are in sync now")
            else:

                logger.info("Inside initiate_training_pipeline")
                logger.info("Data Ingestion stage started")
                DataIngestionPipeline().initiate_data_ingestion_pipeline()
                logger.info("Data Ingestion stage completed")
                logger.info("Data Transformation stage started")
                DataTransformationPipeline().initiate_data_transformation_pipeline()
                logger.info("Data Transformation stage completed")
                logger.info("Initiating ModelTrainerPipeline stage")
                ModelTrainerPipeline().initiate_model_training_pipeline()
                logger.info("Model Trainer pipeline stage completed")
                logger.info("syncing s3 to local folder")
                self.sync_artifacts_to_s3()                
                logger.info("artifacs and s3 are in sync")
        

        except Exception as e:
            logger.error(f"Error occured inside initiate_training_pipeline method: {e}")
            raise e