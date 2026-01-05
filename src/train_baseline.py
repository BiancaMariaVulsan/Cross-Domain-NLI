import os
import sys
import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer, DataCollatorWithPadding, AutoConfig
import argparse
import inspect
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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def train_baseline_model(base_model_dir: str | None = None, output_dir_override: str | None = None):
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
    # Allow training from either base model or a local DAPT-adapted checkpoint
    base_path = base_model_dir or MODEL_NAME
    # Ensure 3-class classification head
    config = AutoConfig.from_pretrained(base_path)
    config.num_labels = 3
    model, loading_info = AutoModelForSequenceClassification.from_pretrained(
        base_path, config=config, ignore_mismatched_sizes=True, output_loading_info=True
    )
    tokenizer = AutoTokenizer.from_pretrained(base_path)

    # Prefer dedicated GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # Verify loaded weights
    print("Loading info:")
    print(f"- missing_keys: {loading_info.get('missing_keys', [])}")        # should be classifier.* only
    print(f"- unexpected_keys: {loading_info.get('unexpected_keys', [])}")  # should be empty
    print(f"- error_msgs: {loading_info.get('error_msgs', [])}")            # should be empty

    # Diagnostics: confirm weights and shapes
    print(f"Model loaded: {model.config.name_or_path}, num_labels={model.config.num_labels}")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    # Embedding matrix should match vocab size
    emb_shape = tuple(model.base_model.embeddings.word_embeddings.weight.shape)
    print(f"Embedding weight shape: {emb_shape}")

    # List classifier params (newly initialized)
    head_params = [n for n, _ in model.named_parameters() if n.startswith(("classifier.", "score."))]
    print(f"Classification head params: {head_params}")

    # Quick sanity stats for encoder weights (should be non-zero and finite)
    with torch.no_grad():
        enc_w = model.base_model.encoder.layer[0].output.dense.weight
        print(f"Encoder L0 output.dense.weight: mean={enc_w.mean().item():.6f}, std={enc_w.std().item():.6f}")
        cls_w = None
        for n, p in model.named_parameters():
            if n.endswith("classifier.out_proj.weight") or n.endswith("classifier.weight"):
                cls_w = p
                break
        if cls_w is not None:
            print(f"Classifier weight: mean={cls_w.mean().item():.6f}, std={cls_w.std().item():.6f}")

    # Ensure the presence of expected new classifier parameters
    expected_new = {"classifier.dense.weight", "classifier.dense.bias", "classifier.out_proj.weight", "classifier.out_proj.bias"}
    present_new = set(p for p in head_params)
    assert present_new, "No classifier head params found."
    # Ensure no other unexpected head prefixes appear
    unexpected = [n for n in head_params if n not in expected_new]
    if unexpected:
        print(f"Warning: unexpected classifier params: {unexpected}")

    if output_dir_override:
        output_dir = output_dir_override
    else:
        # If training from a local path, include its basename in the run name
        if os.path.isabs(base_path) or os.path.sep in base_path or os.path.exists(base_path):
            base_name = os.path.basename(os.path.normpath(base_path))
        else:
            base_name = MODEL_NAME
        output_dir = os.path.join(MODEL_DIR, f"{base_name}_baseline_{TRAIN_DATASET}")

    # Training Arguments
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
        "dataloader_num_workers": 4,
        "pin_memory": torch.cuda.is_available(),
    }

    _ta_kwargs.update({
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "accuracy",
    })

    _allowed = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    _allowed.discard("self") # remove 'self' from parameters
    # If evaluation_strategy isn't supported, drop related flags to avoid inconsistency errors
    if "evaluation_strategy" not in _allowed:
        for key in ("evaluation_strategy", "save_strategy", "load_best_model_at_end", "metric_for_best_model"):
            _ta_kwargs.pop(key, None)
    _filtered = {k: v for k, v in _ta_kwargs.items() if k in _allowed}

    training_args = TrainingArguments(**_filtered)

    _tokenizer = tokenizer  # reuse loaded tokenizer for dynamic padding
    _data_collator = DataCollatorWithPadding(tokenizer=_tokenizer)

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
    trainer.train()
    trainer.save_model(output_dir)
    print(f"\nTraining complete. Best model saved to: {output_dir}")

    # Log the final in-domain evaluation results
    in_domain_metrics = trainer.evaluate(eval_dataset)
    print("\n--- Final In-Domain Metrics (MultiNLI Matched) ---")
    print(in_domain_metrics)


def _parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a classification head for NLI.")
    parser.add_argument("--base-model-dir", type=str, default=None, help="Base model to start from (HF id or local path). Use your DAPT checkpoint to fine-tune from DAPT.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional explicit output directory for the fine-tuned model.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_baseline_model(base_model_dir=args.base_model_dir, output_dir_override=args.output_dir)