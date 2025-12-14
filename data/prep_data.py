from datasets import load_dataset
from transformers import AutoTokenizer
import evaluate
from config.default_config import MODEL_NAME, MAX_SEQ_LENGTH, LABEL_MAPPING

# Load the tokenizer based on the chosen model name
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Load the evaluation metrics
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")


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
            labels.append(label)
        else:
            # Handle cases where label is unknown/missing (e.g., -1 in MultiNLI mismatch)
            labels.append(LABEL_MAPPING.get(str(label), -1))

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def load_and_preprocess_dataset(dataset_name, split="train"):
    """
    Loads a dataset from the Hugging Face Hub, preprocesses it, and removes unnecessary columns.
    """
    try:
        if dataset_name == "medical_nli":
            # MedNLI is hosted as 'anli' with a specific sub-split structure
            dataset = load_dataset("anli", split=split)
            # Filter for the specific MedNLI subset if needed, or adjust based on dataset source
            # For simplicity, we assume 'medical_nli' source for now (might require manual download)
            
            # NOTE: For MedNLI, you may need to use 'csv' or 'json' loading if it's not on the HF Hub.
            # E.g., load_dataset("csv", data_files={"train": "MedNLI_train.csv"})
            
            # --- Using a common NLI format as a placeholder for MedNLI ---
            # If the user provides the raw MedNLI files, this part needs adjustment.
            # Assuming a standard HF format for the moment:
            dataset = load_dataset('glue', 'mnli', split=split).rename_column('idx', 'ID')
            print(f"!!! WARNING: Using GLUE MNLI as placeholder for {dataset_name}. Please confirm the MedNLI source.")
        elif dataset_name == "scitail":
            # SciTail has specific splits
            dataset = load_dataset("scitail", "tsv_format", split=split)
            dataset = dataset.rename_column('sentence1', 'premise').rename_column('sentence2', 'hypothesis')
        else: # Covers multi_nli using the GLUE MNLI task format
            dataset = load_dataset("glue", "mnli", split=split)
            
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}. Ensure the dataset is accessible.")
        return None

    # Preprocess the dataset
    tokenized_datasets = dataset.map(
        preprocess_function, 
        batched=True, 
        remove_columns=[col for col in dataset.column_names if col not in ['input_ids', 'attention_mask', 'labels']],
        load_from_cache_file=False
    )
    
    # Filter out examples where the label could not be mapped (i.e., -1)
    tokenized_datasets = tokenized_datasets.filter(lambda example: example['labels'] != -1)
    
    return tokenized_datasets


def compute_metrics(eval_pred):
    """
    Calculates Accuracy and F1-score for model evaluation.
    """
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    
    # Calculate accuracy
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    
    # Calculate Macro F1 (average F1 for each class)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")

    return {"accuracy": acc['accuracy'], "f1_macro": f1['f1']}