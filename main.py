from src.summarizer import logger
from src.summarizer.pipeline.training_pipeline import TrainingPipeline
from src.summarizer.pipeline.text_summarization_piepline import TextSummarizationPipeline
from src.summarizer import constants as const
if __name__ == "__main__":
    try:
        logger.info("inside main.py")
        if const.PIPELINE == 'predict':
            logger.info("initiating text_summarization pipeline")
            input_text = "What I love is a city I can describe to you not in miles or landmarks. Not even in the hauntings of what was once but is no longer: the ghost of the long-gone corner store, the ghost of the basketball court and the community center, which is now an empty parking lot encased by empty condos no one can afford to live in. I most love a city I can describe to you by sound or song. Landmarks created via sonic moments, rather than the fleeting nature of architecture.To put this another way: There is an outerbelt that runs around central Ohio. It is a circular highway, Interstate 270. It loops around the Columbus Metropolitan Area, which includes the center of the city, and then its outlying suburbs. There are few opportunities to tell any story of the American highway system without also telling a story of race, or displacement, or the long-tail impacts of prioritizing the convenience of people who have money and access and privilege over the people who do not. The outerbelt is no different. Though its construction in the 1950s didn’t tear through and effectively wipe out established Black neighborhoods like some of the highways it connects to, the function of 270 was, effectively, to create a loop for easy access to the suburbs while skirting the center of the city, avoiding Black neighborhoods entirely. It was a way of offering what could be perceived as a safe passage from one idyllic haven to the next."
            TextSummarizationPipeline().initiate_text_summarization_pipeline(input_text=input_text)
        else:
            logger.info("Initiating Training Pipeline")
            TrainingPipeline().initiate_training_pipeline()
    except Exception as e:
        logger.error(f"Error occured inside main:{e}")
        raise e