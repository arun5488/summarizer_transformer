from src.summarizer import logger
from src.summarizer.config.configuration import ConfigurationManager
from src.summarizer.components.text_summarizer import TextSummarizer

class TextSummarizationPipeline:
    def __init__(self):
        logger.info("initialized TextSummarizationPipeline")
        self.config = ConfigurationManager().get_text_summarizer_config()
    
    def initiate_text_summarization_pipeline(self, input_text):
        try:
            logger.info("Inside initiate_text_summarization_pipeline method")
            TextSummarizer(self.config).initiate_text_summarization(input_text=input_text)

        except Exception as e:
            logger.error(f"Error occured inside initiate_text_summarization_pipeline method:{e}")
            raise e