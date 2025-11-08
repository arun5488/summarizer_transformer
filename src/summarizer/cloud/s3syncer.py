from src.summarizer import logger
import os 

class S3Sync:
    def __init__(self):
        logger.info("Initialized s3 syncer")

    def sync_local_to_s3(self, local_dir_path, aws_bucket_url):
        try:
            logger.info("Inside sync_local_to_s3 folder")
            logger.info(f"local_dir_path:{local_dir_path}")
            logger.info(f"aws bucket url:{aws_bucket_url}")
            command = f"aws s3 sync {local_dir_path} {aws_bucket_url}"
            os.system(command=command)
        except Exception as e:
            logger.error(f"Error occured inside sync_local_to_s3 folder")
            raise e
    
    def sync_s3_to_local(self, aws_bucket_url, local_dir_path):
        try:
            logger.info("inside sync_s3_to_local method")
            logger.info(f"aws_bucket_url:{aws_bucket_url}")
            logger.info(f"local_dir_path:{local_dir_path}")
            command = f"aws s3 sync {aws_bucket_url} {local_dir_path}"
            os.system(command= command)
        except Exception as e:
            logger.error(f"Error occured inside sync_s3_to_local: {e}" )
            raise e

