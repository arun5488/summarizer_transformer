## Text summarizer using Huggingface transformer

This project implements the Text Summarization task using "google/pegasus-cnn_dailymail" Model and tokenizer.
The training dataset is also taken from Huggingface datasets, "abisee/cnn_dailymail". Validation split of the dataset is considered for this project. Only a small percentage of data is being used for this project. the percentage of dataset for training is controlled by DATASET_SPLIT_PERCENTAGE constant. Handling of this dataset is done in src > data_ingestion component

Dataset is then tokenized using the PegasusTokenizer in src > data_transformation component. While tokenizing the max limit for tokens is set as 512 using max_input_length in params.yaml. the default max length for this tokenizer is also the same and can be changed if needed in the params.yaml. if the text being tokenized is greater than the max token, the remaining text is truncated. Tokenization was tried with retaining the remaining text as a new record having a stride window, but this resulted in huge number of records to get tokenzied so this approach was discarded. 

Model training is done on the AutoModelForSeq2SeqLM model using the trainer api. The training arguments are passed using Seq2SeqTrainingArguments object. these arguments are maintain in src > constants. Model takes around 15 mins for completing the training in the Google colab T4 GPU.

ROUGE score(Recall-Oriented Understudy for Gisting Evaluation) is used to evaluate the model in the src > model_evaluation component. The basic idea behind this metric is to compare a generated summary against a reference summary generate by humans. to implement this, random 100 records are taken from the dataset, the generated summary of these is then compared with the reference summaries. the results are retrieved in the following format - 
summarizer:results:{'rouge1': 37.84, 'rouge2': 18.37, 'rougeL': 27.26, 'rougeLsum': 27.27}; where - 
Rouge1: Measures overlap of unigrams (single words) between generated and reference summaries. Reflects basic content coverage.
Rouge2: Measures overlap of bigrams (two-word sequences). Captures fluency and phrase-level accuracy
RougeL: Measures the longest common subsequence. Reflects sentence-level structure and coherence.
RougeLsum: Variant of ROUGE-L tailored for multi-sentence summaries. Often used in document summarization.

Text Summarization pipeline is used to generate summary for any given input. here if the length of input goes beyond the token limit then the text is not truncated, rather is remaining part is stored as chunks, of length being equal to the max_length limit, is tokenized as a new record for which summary is generated separately. Once the summaries are generated for all chunks, the summaries are combined, duplicate sentences are removed if any, and shown as output.

the entire application is served on flask as an UI Web application. 


