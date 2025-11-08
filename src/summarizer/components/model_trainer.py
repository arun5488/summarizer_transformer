from src.summarizer import logger
from src.summarizer.entity import ModelTrainerConfig
from transformers import Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer, AutoModelForSeq2SeqLM, PegasusTokenizer
from datasets import load_from_disk
from src.summarizer import constants as const
import numpy as np
from nltk import sent_tokenize
import evaluate

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        logger.info("initialized ModelTrainer")
        self.config = config
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.checkpoint)
        self.tokenizer = PegasusTokenizer.from_pretrained(config.tokenizer)
        self.model_name = config.checkpoint.split("/")[-1]
        self.tokenized_dataset = load_from_disk(config.tokenized_data_path)
        self.logging_steps = len(self.tokenized_dataset["input_ids"]) // const.BATCH_SIZE
        self.rouge_score = evaluate.load("rouge")
        self.args = Seq2SeqTrainingArguments(
            output_dir = f"{self.model_name}",
            learning_rate = const.LEARNING_RATE,
            per_device_train_batch_size = const.BATCH_SIZE,
            per_device_eval_batch_size = const.BATCH_SIZE,
            weight_decay = const.WEIGHT_DECAY,
            save_total_limit=const.SAVE_TOTAL_LIMIT,
            num_train_epochs=const.NUM_TRAIN_EPOCHS,
            predict_with_generate = const.PREDICT_WITH_GENERATE,
            logging_steps = self.logging_steps,
            push_to_hub = const.PUSH_TO_HUB,
            report_to = const.REPORT_TO,
            fp16 = const.FP16,
            gradient_accumulation_steps= const.GRADIENT_ACCUMULATION_STEPS
        )
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer, self.model)
        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args= self.args,
            train_dataset= self.tokenized_dataset['train'],
            eval_dataset = self.tokenized_dataset['test'],
            data_collator= self.data_collator,
            processing_class= self.tokenizer,
            compute_metrics= self.compute_metrics

        )
    

    def compute_metrics(self, eval_pred):
        try:
            logger.info("Inside compute_metrics method")
            predictions, labels = eval_pred
            logger.info("Decode generated summaries into text")
            decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens = True)
            logger.info("replace -100 in tokens as we cant decode them")
            labels = np.where(labels != 100, labels, self.tokenizer.pad_token_id)
            logger.info("decode reference summary to text")
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens = True)
            logger.info("ROUGE expects a new line after each sentence")
            decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
            decoded_labels = ["\n".join(sent_tokenize(pred.strip()))for pred in decoded_labels]
            logger.info("Import ROUGE scores")
            result = self.rouge_score.compute(
                predictions= decoded_preds, references= decoded_labels, use_stemmer = True
            )
            logger.info("Extract the median score")
            result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
            return {k: round(v,4) for k, v in result.items()}

        except Exception as e:
            logger.error(f"Error occured inside compute_metrics: {e}")
            raise e

    def initiate_model_training(self):
        try:
            logger.info("inside initiate_model_training")
            self.trainer.train()
            logger.info("saving the trainer model in local")
            self.trainer.save_model(self.config.model_path)
        except Exception as e:
            logger.error(f"Error occured inside initiate_model_training:{e}")
            raise e