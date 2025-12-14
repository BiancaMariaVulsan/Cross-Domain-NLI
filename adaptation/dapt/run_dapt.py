import os
import sys
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForMaskedLM,  # Use a different model class for MLM
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling # Crucial for dynamic masking
)

# Ensure project root is on PYTHONPATH for module imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.default_config import (
    MODEL_NAME, 
    LEARNING_RATE, 
    BATCH_SIZE, 
    WEIGHT_DECAY, 
    SEED,
    MODEL_DIR
)

# --- DAPT-Specific Configuration ---
DAPT_CORPUS_NAME = "dapt_corpus"  # Name used in data prep script
DAPT_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", f"{DAPT_CORPUS_NAME}_tokenized")
DAPT_MODEL_NAME = f"{MODEL_NAME}_dapt_{DAPT_CORPUS_NAME}"
DAPT_SAVE_PATH = os.path.join(MODEL_DIR, DAPT_MODEL_NAME)

# DAPT Hyperparameters (often fewer epochs than fine-tuning)
DAPT_NUM_EPOCHS = 1
MLM_PROBABILITY = 0.15 # 15% of tokens are masked, standard practice

def run_dapt():
    """
    Performs Domain-Adaptive Pretraining (DAPT) using the Masked Language Modeling (MLM)
    objective on the prepared OOD corpus (e.g., arXiv).
    """
    
    # 1. Load Data and Base Model for MLM
    print(f"--- 1. Loading Prepared DAPT Corpus from: {DAPT_OUTPUT_PATH} ---")
    if not os.path.exists(DAPT_OUTPUT_PATH):
        print("ERROR: DAPT corpus not found. Please run 03_prepare_dapt_corpus.py first.")
        return

    lm_datasets = load_from_disk(DAPT_OUTPUT_PATH)
    
    # DAPT is typically only done on the train split
    train_dataset = lm_datasets['train'] if 'train' in lm_datasets else lm_datasets

    print(f"Total MLM blocks loaded: {len(train_dataset)}")
    
    print("\n--- 2. Loading Model and Data Collator ---")
    
    # Crucially, load the model using AutoModelForMaskedLM
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # The Data Collator handles the random masking of tokens during training
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=True, 
        mlm_probability=MLM_PROBABILITY
    )
    
    # Define Training Arguments (minimal logging, long steps)
    training_args = TrainingArguments(
        output_dir=DAPT_SAVE_PATH,
        num_train_epochs=DAPT_NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        warmup_steps=500,
        weight_decay=WEIGHT_DECAY,
        logging_dir="./dapt_logs",
        logging_steps=1000,
        learning_rate=LEARNING_RATE / 2, # Often slightly lower LR for pretraining
        seed=SEED,
        fp16=torch.cuda.is_available(),
        save_strategy="epoch",
        dataloader_num_workers=4,
        # IMPORTANT: No evaluation or save strategy needed for this intermediate step
    )

    # 3. Initialize and Run Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    print(f"\n--- 3. Starting DAPT (MLM) on {DAPT_CORPUS_NAME} for {DAPT_NUM_EPOCHS} epoch(s) ---")
    
    # Start DAPT
    trainer.train()
    
    # Save the DAPT-adapted model
    trainer.save_model(DAPT_SAVE_PATH)
    print(f"\nDAPT complete. Adapted model saved to: {DAPT_SAVE_PATH}")


if __name__ == "__main__":
    run_dapt()