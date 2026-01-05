import os
import argparse
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, DataCollatorWithPadding, AutoConfig
from typing import Optional
from config.default_config import (
    MODEL_NAME,
    TRAIN_DATASET,
    OOD_DATASETS,
    BATCH_SIZE,
    MODEL_DIR,
    RESULTS_DIR,
)
from data.prep_data import load_and_preprocess_dataset, compute_metrics
from transformers import TrainingArguments


def evaluate_cross_domain(model_dir: str = None, label: str = "Baseline", results_file: str = None, adapter_dir: Optional[str] = None):
    """
    Evaluates a trained sequence classification model on in-domain and OOD datasets.

    Parameters
    - model_dir: Path to the model directory (containing config.json, model weights, tokenizer).
                 If None, defaults to the baseline fine-tuned model path.
    - label:     A short label for the model (e.g., "Baseline", "DAPT").
    - results_file: Output CSV path. If None, defaults based on label.
    """

    # 1. Resolve Model Path
    default_baseline_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_baseline_{TRAIN_DATASET}")
    model_path = model_dir or default_baseline_path

    print(f"--- 1. Loading Trained Model from: {model_path} ---")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    # Ensure classification head has 3 labels even if starting from a DAPT checkpoint
    if not hasattr(config, "num_labels") or config.num_labels != 3:
        config.num_labels = 3
    # Allow initializing a fresh head when loading from non-classification checkpoints
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        config=config,
        ignore_mismatched_sizes=True,
    )

    # If a LoRA adapter is provided, attach it for inference
    if adapter_dir:
        try:
            from peft import PeftModel
            print(f"Attaching LoRA adapter from: {adapter_dir}")
            model = PeftModel.from_pretrained(model, adapter_dir)
        except Exception as e:
            print(f"Warning: Failed to load LoRA adapter at {adapter_dir}: {e}")

    # 2. Setup Trainer for Evaluation
    eval_kwargs = {
        "output_dir": "./tmp_eval",
        "per_device_eval_batch_size": BATCH_SIZE,
        "do_train": False,
        "do_eval": True,
        "dataloader_num_workers": 0,
    }
    eval_args = TrainingArguments(**eval_kwargs)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=eval_args,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 3. Perform Evaluation and Record Results
    results = []
    
    datasets_to_evaluate = {
        f"{TRAIN_DATASET}_matched": load_and_preprocess_dataset(TRAIN_DATASET, split="validation_matched"),
        f"{TRAIN_DATASET}_mismatched": load_and_preprocess_dataset(TRAIN_DATASET, split="validation_mismatched"),
    }
    
    # Add OOD datasets
    for spec in OOD_DATASETS:
        if isinstance(spec, dict):
            ds = load_and_preprocess_dataset(spec["name"], split=spec.get("split", "test"))
            name = spec["name"]
        else:
            ds = load_and_preprocess_dataset(spec, split="test")
            name = spec
        datasets_to_evaluate[name] = ds

    for name, dataset in datasets_to_evaluate.items():
        if dataset is None:
            continue
            
        print(f"\n--- Evaluating on {name} (Size: {len(dataset)}) ---")
        metrics = trainer.evaluate(eval_dataset=dataset)
        
        results.append({
            "Model": f"{MODEL_NAME} {label}",
            "Trained On": TRAIN_DATASET,
            "Evaluated On": name,
            "Accuracy": metrics.get("eval_accuracy"),
            "F1_Macro": metrics.get("eval_f1_macro"),
        })
        print(f"Results for {name}: Accuracy={metrics.get('eval_accuracy'):.4f}, F1={metrics.get('eval_f1_macro'):.4f}")

    # 4. Save and Summarize
    df_results = pd.DataFrame(results)
    
    # Calculate generalization gap relative to the In-Domain (MultiNLI Matched) Accuracy
    id_acc_row = df_results[df_results['Evaluated On'] == f"{TRAIN_DATASET}_matched"]
    if not id_acc_row.empty:
        id_acc = id_acc_row['Accuracy'].iloc[0]
        df_results['Generalization_Gap'] = df_results['Accuracy'].apply(lambda acc: id_acc - acc)
    else:
        df_results['Generalization_Gap'] = None

    if results_file is None:
        # Default filename driven by label
        fname = f"{label.lower()}_results.csv".replace(" ", "_")
        results_file = os.path.join(RESULTS_DIR, fname)
    else:
        # If a bare name is provided, save under RESULTS_DIR and ensure .csv
        if not os.path.isabs(results_file) and os.path.dirname(results_file) == "":
            fname = results_file if results_file.lower().endswith(".csv") else f"{results_file}.csv"
            results_file = os.path.join(RESULTS_DIR, fname)
    df_results.to_csv(results_file, index=False)

    print("\n" + "="*50)
    print(f"Cross-Domain Benchmarking Complete!")
    print(f"Detailed results saved to: {results_file}")
    print("="*50)
    print(df_results)


def _parse_args():
    parser = argparse.ArgumentParser(description="Evaluate in-domain and OOD datasets for a trained model.")
    parser.add_argument("--model-dir", type=str, default=None, help="Path to trained classification model directory.")
    parser.add_argument("--label", type=str, default="Baseline", help="Label for the model (e.g., Baseline, DAPT).")
    parser.add_argument("--results-file", type=str, default=None, help="Optional output CSV path.")
    parser.add_argument("--adapter-dir", type=str, default=None, help="Optional LoRA adapter directory for inference.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate_cross_domain(model_dir=args.model_dir, label=args.label, results_file=args.results_file, adapter_dir=args.adapter_dir)