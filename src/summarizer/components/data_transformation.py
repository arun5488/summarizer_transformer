from src.summarizer import logger
from src.summarizer import constants as const
from src.summarizer.entity import DataTransformationConfig
from src.summarizer.utils.common import read_yaml
from transformers import PegasusTokenizer
from datasets import load_from_disk
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

class DataTransformation:
    def __init__(self, config = DataTransformationConfig):
        logger.info("initialized DataTransformation")
        self.config = config
        self.schema = read_yaml(const.SCHEMA_YAML)
        self.tokenizer = PegasusTokenizer.from_pretrained(config.checkpoint)
    


    def preprocess_function(self, examples):
        try:
            logger.info("inside preprocess_function")
            model_inputs = self.tokenizer(
            examples["article"],
            max_length=self.config.max_input_length,
            truncation=True,
            )
            labels = self.tokenizer(
                examples["highlights"], max_length=self.config.max_target_length, truncation=True
            )
            
            model_inputs["labels"] = labels["input_ids"]
            logger.info(f"datatype of model_inputs:{type(model_inputs)}")
            
            return model_inputs
        except Exception as e:
            logger.error(f"Error occured inside preprocess_function:{e}")
            raise e
    
    def chunk_tokenized_input(self, text):
        try:
            logger.info("inside chunk_tokenized_input method")
            tokens = self.tokenizer(text, truncation = False, return_attention_mask = True)
            input_ids = tokens["input_ids"]
            logger.info(f"length of input ids:{input_ids}")
            attention_mask = tokens["attention_mask"]
            max_length = self.config.max_input_length
            logger.info(f"Max length of input for model: {max_length}")
            stride = self.config.stride
            logger.info(f"stride value:{stride}")
            chunks = []
            for i in range(0, len(input_ids), stride):
                chunk_ids = input_ids[i:i+max_length]
                chunk_mask = attention_mask[i:i+max_length]
                if len(chunk_ids) == 0:
                    break
                chunks.append({
                    "input_ids": chunk_ids,
                    "attention_mask": chunk_mask
                })
            return chunks 
        except Exception as e:
            logger.error(f"Error occured inside chunk_tokenized_input method:{e}")
            raise e
    
    def initiate_data_transformation_for_training(self):
        try:
            logger.info("Inside initiate_data_transformation method")
            dataset = load_from_disk(self.config.dataset_path)['train']
            logger.info("dataset loaded successfully from local")
            logger.info(f"{type(dataset["article"])}")
            logger.info(f"{type(list(dataset["article"]))}")
            tokenized_dataset = dataset.map(self.preprocess_function, batched = True)
            logger.info("dataset tokenized successfully")
            tokenized_dataset.save_to_disk(self.config.tokenized_data_path)
            logger.info("tokenized dataset saved to artifacts folder")
            self.tokenizer.save_pretrained(self.config.tokenizer)
            logger.info("saved the tokenizer to local folder")

        except Exception as e:
            logger.error(f"Error occured inside initiate_data_transformation:{e}")
            raise e
    def initiate_data_transformation_for_summary_generation(self, text):
        try:
            logger.info("inside initiate_data_transformation_for_summary_generation")
            chunks = self.chunk_tokenized_input(text)
            return chunks

        except Exception as e:
            logger.error(f"Error occured inside initiate_data_transformation_for_summary_generation method:{e}")
            raise e
