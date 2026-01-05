import os
from datasets import load_dataset
from transformers import AutoTokenizer
import evaluate
from config.default_config import MODEL_NAME, MAX_SEQ_LENGTH, LABEL_MAPPING

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
HF_HOME = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".hf_cache")
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(HF_HOME, "datasets"))

# Load tokenizer once
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")


def _normalize_columns(ds):
    cols = set(ds.column_names)
    # Standardize to premise/hypothesis/label
    if "sentence1" in cols and "sentence2" in cols:
        ds = ds.rename_columns({"sentence1": "premise", "sentence2": "hypothesis"})
    if "premise" not in ds.column_names and "question1" in cols:
        ds = ds.rename_column("question1", "premise")
    if "hypothesis" not in ds.column_names and "question2" in cols:
        ds = ds.rename_column("question2", "hypothesis")
    # Label column variants
    if "gold_label" in cols and "label" not in cols:
        ds = ds.rename_column("gold_label", "label")
    return ds


def preprocess_function(examples):
    """
    Tokenizes the premise and hypothesis and returns the features required by the model.
    """
    # Tokenize the input text, truncating and padding to MAX_SEQ_LENGTH
    tokenized_inputs = tokenizer(
        examples["premise"],
        examples["hypothesis"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    )

    # Convert original labels (string or integer) to standardized integer IDs
    labels = []
    for label in examples["label"]:
        if isinstance(label, str):
            # Use the mapping defined in config for string labels
            labels.append(LABEL_MAPPING.get(label.lower(), -1))
        elif isinstance(label, int):
            # Directly use integer labels if they match the 0, 1, 2 convention
            labels.append(label if label in (0, 1, 2) else -1)
        else:
            # Handle cases where label is unknown/missing (e.g., -1 in MultiNLI mismatch)
            labels.append(LABEL_MAPPING.get(str(label), -1))

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def load_and_preprocess_dataset(dataset_name, split="train"):
    """
    Loads a dataset by name/split, normalizes columns, tokenizes, and returns torch-formatted features.
    """
    # Map friendly names
    hub_name = dataset_name
    hf_split = split
    if isinstance(dataset_name, dict):
        hub_name = dataset_name.get("name", dataset_name)
        hf_split = dataset_name.get("split", split)

    if isinstance(hub_name, str) and hub_name.lower() in ["medical_nli", "snli"]:
        hub_name = "snli"
        if split == "train": hf_split = "train"
        elif split in ["validation", "dev"]: hf_split = "validation"
        elif split == "test": hf_split = "test"

    try:
        if hub_name == "scitail":
            ds = load_dataset("scitail", "tsv_format", split=hf_split)
            ds = _normalize_columns(ds)
            # SciTail labels are "entails"/"neutral" or similar; normalize via LABEL_MAPPING
            if "label" not in ds.column_names and "gold_label" in ds.column_names:
                ds = ds.rename_column("gold_label", "label")
        elif hub_name in ["multi_nli", "mnli", "glue_mnli"]:
            # Use GLUE MNLI for MultiNLI
            ds = load_dataset("glue", "mnli", split=hf_split)
            # Glue mnli uses integer labels {0,1,2}
            if "label" not in ds.column_names:
                ds = ds.rename_column("labels", "label")
            # Normalize columns
            ds = _normalize_columns(ds)
        elif hub_name == "snli":
            ds = load_dataset("snli", split=hf_split)
            ds = _normalize_columns(ds)
        else:
            # Fallback: try hub_name as provided
            ds = load_dataset(hub_name, split=hf_split)
            ds = _normalize_columns(ds)
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}. Ensure the dataset is accessible.")
        return None

    required = {"premise", "hypothesis", "label"}
    if not required.issubset(set(ds.column_names)):
        print(f"Error: normalized columns missing. Got {ds.column_names}")
        return None

    tokenized = ds.map(
        preprocess_function,
        batched=True,
        remove_columns=[c for c in ds.column_names if c not in ["premise", "hypothesis", "label"]],
        load_from_cache_file=True,
        num_proc=1,
        keep_in_memory=True,
        desc=f"Tokenizing {hub_name}:{hf_split}"
    )
    # Proactive cache cleanup to drop mmaps before exit
    try:
        tokenized.cleanup_cache_files()
    except Exception:
        pass
    tokenized = tokenized.filter(
        lambda x: x["labels"] != -1,
        load_from_cache_file=True,
        num_proc=1,
        keep_in_memory=True
    )
    tokenized = tokenized.with_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


def compute_metrics(eval_pred):
    """
    Calculates Accuracy and F1-score for model evaluation.
    """
    logits, labels = eval_pred
    import numpy as np
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")
    return {"eval_accuracy": acc['accuracy'], "eval_f1_macro": f1['f1']}