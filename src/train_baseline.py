import os
import sys
import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer, DataCollatorWithPadding

# Ensure project root is on PYTHONPATH for module imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.default_config import (
    MODEL_NAME, 
    TRAIN_DATASET, 
    LEARNING_RATE, 
    BATCH_SIZE, 
    NUM_EPOCHS, 
    WEIGHT_DECAY, 
    SEED,
    MODEL_DIR
)
from data.prep_data import load_and_preprocess_dataset, compute_metrics


def train_baseline_model():
    """
    Fine-tunes the RoBERTa model on the MultiNLI training data.
    """
    print(f"--- 1. Loading and Preprocessing {TRAIN_DATASET} Data ---")
    
    # Load training data (using 'train' split)
    train_dataset = load_and_preprocess_dataset(TRAIN_DATASET, split="train")
    
    # Load evaluation data (using 'validation_matched' split for general comparison)
    eval_dataset = load_and_preprocess_dataset(TRAIN_DATASET, split="validation_matched")

    if train_dataset is None or eval_dataset is None:
        print("Data loading failed. Exiting.")
        return

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    
    
    print("\n--- 2. Loading Model ---")
    # The NLI task is 3-class classification
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    
    # Define the output directory for checkpoints
    output_dir = os.path.join(MODEL_DIR, f"{MODEL_NAME}_baseline_{TRAIN_DATASET}")

    # Define Training Arguments with version-aware kwargs
    _ta_kwargs = {
        "output_dir": output_dir,
        "num_train_epochs": NUM_EPOCHS,
        "per_device_train_batch_size": BATCH_SIZE,
        "per_device_eval_batch_size": BATCH_SIZE,
        "warmup_steps": 500,
        "weight_decay": WEIGHT_DECAY,
        "logging_dir": "./logs",
        "logging_steps": 500,
        "learning_rate": LEARNING_RATE,
        "seed": SEED,
        "fp16": torch.cuda.is_available(),
    }

    # Try to add strategy flags, then filter by actual __init__ signature to avoid unsupported args
    _ta_kwargs.update({
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "accuracy",
    })

    import inspect
    _allowed = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    # remove 'self' from parameters
    _allowed.discard("self")
    # If evaluation_strategy isn't supported, drop related flags to avoid inconsistency errors
    if "evaluation_strategy" not in _allowed:
        for key in ("evaluation_strategy", "save_strategy", "load_best_model_at_end", "metric_for_best_model"):
            _ta_kwargs.pop(key, None)
    _filtered = {k: v for k, v in _ta_kwargs.items() if k in _allowed}

    training_args = TrainingArguments(**_filtered)

    # Prepare tokenizer and data collator for dynamic padding
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _data_collator = DataCollatorWithPadding(tokenizer=_tokenizer)

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=_tokenizer,
        data_collator=_data_collator,
        compute_metrics=compute_metrics,
    )
    
    print("\n--- 3. Starting Fine-Tuning ---")
    
    # Start training
    train_result = trainer.train()
    
    # Save the best model
    trainer.save_model(output_dir)
    print(f"\nTraining complete. Best model saved to: {output_dir}")

    # Log the final in-domain evaluation results
    in_domain_metrics = trainer.evaluate(eval_dataset)
    print("\n--- Final In-Domain Metrics (MultiNLI Matched) ---")
    print(in_domain_metrics)


if __name__ == "__main__":
    train_baseline_model()