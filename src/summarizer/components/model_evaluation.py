from src.summarizer import logger
from datasets import load_from_disk
from transformers import AutoModelForSeq2SeqLM, PegasusTokenizer, DataCollatorForSeq2Seq
from src.summarizer import constants as const
from src.summarizer.entity import ModelevaluationConfig
import evaluate
import torch
from pathlib import Path

class ModelEvaluation:
    def __init__(self, config: ModelevaluationConfig):
        logger.info("Initializing Model Evaluation")
        self.config = config
        logger.info("loading saved model from the local")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path)
        logger.info("loading saved tokenizer from local")
        self.tokenizer = PegasusTokenizer.from_pretrained(self.config.tokenizer)
        logger.info("loading dataset from local folder")
        self.datasets = load_from_disk(self.config.dataset_path)
        self.rouge_score = evaluate.load("rouge")
        logger.info("checking the device")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"device:{self.device}")

    def split_data_set(self, datasets):
        try:
            logger.info("inside split_data_set method")
            shuffled_dataset = datasets.shuffle(seed = 42)
            dataset = shuffled_dataset.train_test_split(train_size = const.TRAIN_TEST_SPLIT)
            eval_dataset = dataset['test']
            logger.info(f"Eval dataset is of length: {len(eval_dataset)}")
            return eval_dataset

        except Exception as e:
            logger.error(f"Error occured inside split_data_set:{e}")
            raise e

    def preprocess_function(self, examples):
        try:
            logger.info("inside preprocess_function in model_evaluation")
            model_inputs = self.tokenizer(
            examples["article"],
            max_length=self.config.max_input_length,
            truncation=True,
            padding = "max_length"
            )
            model_inputs["reference"] = examples["highlights"]
            logger.info(f"datatype of model_inputs:{type(model_inputs)}")
            
            return model_inputs
        except Exception as e:
            logger.error(f"Error occured inside preprocess_function:{e}")
            raise e
    
    def generate_predictions(self, eval_dataset):
        try:
            logger.info("inside generate_predictions method in model evaluation")
            logger.info("load the eval dataset")
            dataset = eval_dataset
            logger.info("loading model in eval mode")
            model = self.model.to(self.device).eval()
            predictions = []
            if Path(self.config.tokenized_eval_data_path).exists():
                tokenized_eval_dataset = load_from_disk(self.config.tokenized_eval_data_path)
            else:           
                tokenized_eval_dataset = dataset.map(self.preprocess_function, batched = True, remove_columns=dataset.column_names)
            
            batch_size = const.BATCH_SIZE
            logger.info("performing batch predictions on eval dataset")
            for i in range(0, len(tokenized_eval_dataset['input_ids']), batch_size):
                logger.info(f"total records:{len(tokenized_eval_dataset['input_ids'])}, record under process:{i}")
                batch = {
                "input_ids": torch.tensor(tokenized_eval_dataset["input_ids"][i:i+batch_size]).to(self.device),
                "attention_mask": torch.tensor(tokenized_eval_dataset["attention_mask"][i:i+batch_size]).to(self.device)
                }
                outputs = model.generate(
                    input_ids = batch["input_ids"],
                    attention_mask = batch['attention_mask'],
                    max_new_tokens = self.config.max_target_length,
                    num_beams = 4,
                    early_stopping = True
                )

                decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens = True)
                predictions.extend(decoded_outputs)
            
            return predictions
        except Exception as e:
            logger.error(f"Error occured inside generate_prediction:{e}")
            raise e
        
    def evaluate_predictions(self, predictions, references):
        try:
            logger.info("inside evaluate_predictions method")
            results = self.rouge_score.compute(predictions=predictions, references=references, use_stemmer = True)
            logger.info(f"ROUGE results: {results}")
            return {k: v["fmeasure"] * 100 for k,v in results.items()}

        except Exception as e:
            logger.error(f"Error occured inside evaluate_predictions method:{e}")
            raise e
        
    def initiate_model_evaluation(self):
        try:
            logger.info("inside initiate_model_evaluation")
            eval_dataset = self.split_data_set(self.datasets).select(range(100))
            predictions = self.generate_predictions(eval_dataset)
            references = eval_dataset['highlights']
            result = self.evaluate_predictions(predictions = predictions, references = references)
            logger.info(f"results:{result}")
        except Exception as e:
            logger.error(f"Error occured inside initiate_model_evaluation")
            raise e



    