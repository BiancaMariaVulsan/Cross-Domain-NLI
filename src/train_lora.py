import os
import sys
import torch
import argparse
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer, DataCollatorWithPadding, AutoConfig

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
    MODEL_DIR,
)
from data.prep_data import load_and_preprocess_dataset, compute_metrics

from peft import LoraConfig, get_peft_model, TaskType


def train_lora_model(
    base_model_dir: str | None = None,
    output_dir_override: str | None = None,
    output_subdir: str | None = None,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: list[str] | None = None,
    bias: str = "none",
    merge_and_save: bool = False,
):
    """
    Parameter-efficient fine-tuning (LoRA) for RoBERTa on MultiNLI.

    - Wraps the base sequence classification model with LoRA adapters.
    - Trains only adapter parameters; base model weights remain frozen.
    - Saves adapter weights to output_dir; optionally merges into base and saves full model.
    """

    print(f"--- 1. Loading and Preprocessing {TRAIN_DATASET} Data ---")
    train_dataset = load_and_preprocess_dataset(TRAIN_DATASET, split="train")
    eval_dataset = load_and_preprocess_dataset(TRAIN_DATASET, split="validation_matched")
    if train_dataset is None or eval_dataset is None:
        print("Data loading failed. Exiting.")
        return

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")

    print("\n--- 2. Loading Base Model ---")
    base_path = base_model_dir or MODEL_NAME
    config = AutoConfig.from_pretrained(base_path)
    config.num_labels = 3
    model = AutoModelForSequenceClassification.from_pretrained(
        base_path, config=config, ignore_mismatched_sizes=True
    )
    tokenizer = AutoTokenizer.from_pretrained(base_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # 2.a Configure LoRA
    if target_modules is None:
        # Recommended defaults for BERT/RoBERTa attention projections
        target_modules = ["query", "value"]

    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        target_modules=target_modules,
        task_type=TaskType.SEQ_CLS,
    )

    model = get_peft_model(model, lora_cfg)
    # Print trainable parameters summary
    trainable, total = 0, 0
    for n, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    pct = 100.0 * trainable / float(total)
    print(f"Trainable params with LoRA: {trainable:,} / {total:,} ({pct:.2f}%)")

    # 3. TrainingArguments (version-safe)
    # Resolve output directory
    _default_out = os.path.join(MODEL_DIR, f"{os.path.basename(os.path.normpath(base_path))}_lora_{TRAIN_DATASET}")
    _resolved_out = output_dir_override or (os.path.join(MODEL_DIR, output_subdir) if output_subdir else _default_out)
    _ta_kwargs = {
        "output_dir": _resolved_out,
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
        "save_total_limit": 10,
        "greater_is_better": True,
    })
    import inspect
    _allowed = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    _allowed.discard("self")
    if "evaluation_strategy" not in _allowed:
        for key in ("evaluation_strategy", "save_strategy", "load_best_model_at_end", "metric_for_best_model"):
            _ta_kwargs.pop(key, None)
    training_args = TrainingArguments(**{k: v for k, v in _ta_kwargs.items() if k in _allowed})

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\n--- 3. Starting LoRA Fine-Tuning ---")
    train_result = trainer.train()

    # Save adapter weights (PeftModel.save_pretrained via Trainer.save_model)
    output_dir = training_args.output_dir
    trainer.save_model(output_dir)
    # Save tokenizer alongside for convenience
    tokenizer.save_pretrained(output_dir)
    print(f"\nTraining complete. LoRA adapter saved to: {output_dir}")

    # Optional: merge adapters into base and save a full model for standalone inference
    if merge_and_save:
        print("Merging LoRA adapters into base weights and saving full model...")
        merged = model.merge_and_unload()
        merged.save_pretrained(os.path.join(output_dir, "merged"))
        print(f"Merged model saved to: {os.path.join(output_dir, 'merged')}")

    # Final evaluation on in-domain validation
    in_domain_metrics = trainer.evaluate(eval_dataset)
    print("\n--- Final In-Domain Metrics (MultiNLI Matched) ---")
    print(in_domain_metrics)


def _parse_args():
    ap = argparse.ArgumentParser(description="LoRA fine-tuning for NLI (RoBERTa)")
    ap.add_argument("--base-model-dir", type=str, default=None, help="Base model (HF id or local path). Use DAPT checkpoint to fine-tune from DAPT.")
    ap.add_argument("--output-dir", type=str, default=None, help="Explicit adapter save directory (defaults under models/).")
    ap.add_argument("--output-subdir", type=str, default=None, help="Subfolder under models/ for adapter outputs (e.g., my_lora_run).")
    ap.add_argument("--r", type=int, default=16, help="LoRA rank.")
    ap.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha scaling factor.")
    ap.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout.")
    ap.add_argument("--target-modules", type=str, nargs="*", default=None, help="Target module names (e.g., query value).")
    ap.add_argument("--bias", type=str, default="none", choices=["none", "all", "lora_only"], help="Bias handling for LoRA layers.")
    ap.add_argument("--merge-and-save", action="store_true", help="Merge adapters into base and save standalone model.")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_lora_model(
        base_model_dir=args.base_model_dir,
        output_dir_override=args.output_dir,
        output_subdir=args.output_subdir,
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias=args.bias,
        merge_and_save=args.merge_and_save,
    )
