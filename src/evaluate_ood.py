import os
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, DataCollatorWithPadding
from config.default_config import (
    MODEL_NAME, 
    TRAIN_DATASET, 
    OOD_DATASETS, 
    BATCH_SIZE, 
    MODEL_DIR, 
    RESULTS_DIR
)
from data.prep_data import load_and_preprocess_dataset, compute_metrics


def evaluate_cross_domain():
    """
    Loads the trained baseline model and evaluates its performance
    on the specified Out-of-Domain (OOD) datasets.
    """
    
    # 1. Define Model Path
    baseline_model_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_baseline_{TRAIN_DATASET}")

    if not os.path.exists(baseline_model_path):
        print(f"ERROR: Baseline model not found at {baseline_model_path}")
        print("Please run scripts/01_train_baseline.py successfully first.")
        return

    print(f"--- 1. Loading Trained Baseline Model from: {baseline_model_path} ---")
    
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(baseline_model_path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 2. Setup Trainer for Evaluation
    from transformers import TrainingArguments
    # Build args with conservative, widely-supported keys
    eval_kwargs = {
        "output_dir": "./tmp_eval",
        "per_device_eval_batch_size": BATCH_SIZE,
        "do_train": False,
        "do_eval": True,
        "dataloader_num_workers": 4,  # if your version complains, set to 0 or remove
    }

    # Create TrainingArguments
    eval_args = TrainingArguments(**eval_kwargs)

    # Use dynamic padding so variable-length input_ids/attention_mask work
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
    
    # Start with the in-domain test set for the generalization gap calculation
    datasets_to_evaluate = {
        f"{TRAIN_DATASET}_matched": load_and_preprocess_dataset(TRAIN_DATASET, split="validation_matched"),
        f"{TRAIN_DATASET}_mismatched": load_and_preprocess_dataset(TRAIN_DATASET, split="validation_mismatched"),
    }
    
    # Add OOD datasets
    for ood_name in OOD_DATASETS:
        datasets_to_evaluate[ood_name] = load_and_preprocess_dataset(ood_name, split="test")

    for name, dataset in datasets_to_evaluate.items():
        if dataset is None:
            continue
            
        print(f"\n--- Evaluating on {name} (Size: {len(dataset)}) ---")
        
        # Run evaluation
        metrics = trainer.evaluate(eval_dataset=dataset)
        
        # Record results
        results.append({
            "Model": f"{MODEL_NAME} Baseline",
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

    results_file = os.path.join(RESULTS_DIR, "baseline_results.csv")
    df_results.to_csv(results_file, index=False)

    print("\n" + "="*50)
    print(f"Cross-Domain Benchmarking Complete!")
    print(f"Detailed results saved to: {results_file}")
    print("="*50)
    print(df_results)


if __name__ == "__main__":
    evaluate_cross_domain()