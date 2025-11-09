from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    root_dir: Path
    dataset_name: str
    dataset_path: Path
    dataset_version: str
    dataset_split: str

@dataclass
class DataTransformationConfig:
    root_dir: Path
    dataset_path: Path
    tokenized_data_path: Path
    tokenizer: Path
    checkpoint: str
    max_input_length: int
    max_target_length: int
    stride: int

@dataclass
class ModelTrainerConfig:
    root_dir: Path
    model_path: Path
    tokenized_data_path: Path
    tokenizer : Path
    checkpoint: str

@dataclass
class ModelevaluationConfig:
    root_dir: Path
    model_path: Path
    tokenizer: Path
    max_input_length: int
    max_target_length: int
    dataset_path: Path
    tokenized_eval_data_path: Path

@dataclass
class TextSummmarizerConfig:
    model_path: Path
    tokenizer: Path
