from src.summarizer import logger
from src.summarizer.entity import TextSummmarizerConfig
from transformers import AutoModelForSeq2SeqLM, PegasusTokenizer
import torch
from src.summarizer.utils.common import chunk_tokenized_input
from src.summarizer import constants as const
from collections import OrderedDict

class TextSummarizer:
    def __init__(self, config: TextSummmarizerConfig):
        logger.info("Initialized TextSummarizer class")
        self.config = config
        logger.info("loading tokenizer from local")
        self.tokenizer = PegasusTokenizer.from_pretrained(self.config.tokenizer)
        logger.info("loading model from local")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"device:{self.device}")

    def summarize_text(self, input_text):
        try:
            logger.info("inside summarize_text method")
            tokenized_input_chunks = chunk_tokenized_input(input_text, tokenizer = self.tokenizer, 
                                                         max_input_length = const.MAX_INPUT_LENGTH, stride = const.STRIDE)
            logger.info("Input text tokenized successfully")
            logger.info(f"Chunks from input:{len(tokenized_input_chunks)}")
            logger.info(f"Chunks type:{type(tokenized_input_chunks)}")
            decoded_summary=[]
            count = 0
            for chunk in tokenized_input_chunks:
                logger.info(f"generating summary for chunk_{count}")
                logger.info(f"chunk input_id type:{type(chunk)}")
                logger.info(f"chunk input_id type:{type(chunk["input_ids"])}")
                summary = self.model.to(self.device).eval().generate(
                    input_ids = torch.tensor(chunk["input_ids"]).unsqueeze(0).to(self.device),
                    attention_mask = torch.tensor(chunk["attention_mask"]).unsqueeze(0),
                    num_beams = const.NUM_BEAMS,
                    length_penalty = const.LENGTH_PENALTY,
                    max_length = const.MAX_LENGTH,
                    min_length = const.MIN_LENGTH,
                    no_repeat_ngram_size = const.NO_REPEAT_NGRAM_SIZE,
                    early_stopping = const.EARLY_STOPPING
                )
                decode_chunk = self.tokenizer.decode(summary[0],skip_special_tokens = True)
                logger.info(f"decoded chunk: {decode_chunk}")
                decoded_summary.append(decode_chunk)
                count = count + 1
            return decoded_summary

        except Exception as e:
            logger.error(f"Error occured inside summarize_text method")
            raise e

    def remove_duplicate_sentences_from_summary(self, decoded_summary):
        try:
            logger.info("Inside remove_duplicate_sentences_from_summary")
            flat_sentences = " ".join(decoded_summary).split(". ")
            unique_sentences = list(OrderedDict.fromkeys(flat_sentences))
            final_summary = ". ".join(unique_sentences)
            return final_summary
        except Exception as e:
            logger.error(f"Error occured inside remove_duplicate_sentences_from_summary:{e}")
            raise e
        
    def initiate_text_summarization(self, input_text):
        try:
            logger.info("inside initiate_text_summarization method")
            decoded_summary = self.summarize_text(input_text)
            final_summary = self.remove_duplicate_sentences_from_summary(decoded_summary)
            logger.info(f"Final summary:{final_summary}")
            return final_summary
        except Exception as e:
            logger.error(f"Error occured inside initiate_text_summarization method:{e}")
            raise e
