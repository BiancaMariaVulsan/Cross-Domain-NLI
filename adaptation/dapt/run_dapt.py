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

# Checkpointing configuration: save frequently to avoid long runs without checkpoints
CHECKPOINT_SAVE_STEPS = 1000  # save every N training steps
CHECKPOINT_TOTAL_LIMIT = 10   # keep last N checkpoints to limit disk usage

def _build_trainer(model, tokenizer, train_dataset, per_device_bs: int, global_bs: int):
    """Build a Trainer with supplied batch sizing and stable defaults."""
    # Keep effective global batch size approximately constant via accumulation
    grad_accum_steps = max(1, global_bs // max(1, per_device_bs))

    # Prefer bf16 on Ampere+ GPUs; otherwise use fp16 if CUDA is available
    use_cuda = torch.cuda.is_available()
    fp16 = False
    bf16 = False
    if use_cuda:
        # bf16 is more stable on newer GPUs
        bf16 = torch.cuda.get_device_capability(0)[0] >= 8
        fp16 = not bf16

    training_args = TrainingArguments(
        output_dir=DAPT_SAVE_PATH,
        num_train_epochs=DAPT_NUM_EPOCHS,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum_steps,
        warmup_steps=500,
        weight_decay=WEIGHT_DECAY,
        logging_dir="./dapt_logs",
        logging_steps=1000,
        learning_rate=LEARNING_RATE / 2,
        seed=SEED,
        fp16=fp16,
        bf16=bf16,
        save_strategy="steps",
        save_steps=CHECKPOINT_SAVE_STEPS,
        save_total_limit=CHECKPOINT_TOTAL_LIMIT,
        dataloader_num_workers=2,  # lower workers to reduce memory footprint
        torch_compile=False,
        optim="adamw_torch",
    )

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=MLM_PROBABILITY
        ),
    )


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
    # Use lighter dtype and safe init to reduce VRAM pressure
    load_kwargs = {}
    if torch.cuda.is_available():
        # Prefer bf16 on newer GPUs if possible, else fp16 for weights
        if torch.cuda.get_device_capability(0)[0] >= 8:
            load_kwargs["torch_dtype"] = torch.bfloat16
        else:
            load_kwargs["torch_dtype"] = torch.float16
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Ensure dedicated GPU is used if available and inform the user
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)
        try:
            dev_name = torch.cuda.get_device_name(0)
            total_mem_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"Using dedicated GPU: {dev_name} ({total_mem_gib:.1f} GiB)")
        except Exception:
            print("Using dedicated GPU (CUDA device detected)")
    else:
        print("CUDA not available. Training will run on CPU.")
    
    # The Data Collator handles the random masking of tokens during training
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=True, 
        mlm_probability=MLM_PROBABILITY
    )
    
    # 3. Initialize Trainer with adaptive batch sizing
    # Start with configured global batch size; reduce per-device if OOM
    global_bs = max(1, BATCH_SIZE)
    per_device_bs = min(global_bs, 8)  # cautious default
    trainer = _build_trainer(model, tokenizer, train_dataset, per_device_bs, global_bs)
    
    print(f"\n--- 3. Starting DAPT (MLM) on {DAPT_CORPUS_NAME} for {DAPT_NUM_EPOCHS} epoch(s) ---")
    
    # Start DAPT with automatic OOM fallback
    try:
        trainer.train()
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and per_device_bs > 1:
            torch.cuda.empty_cache()
            per_device_bs = max(1, per_device_bs // 2)
            print(f"OOM encountered. Retrying with per_device_train_batch_size={per_device_bs}.")
            trainer = _build_trainer(model, tokenizer, train_dataset, per_device_bs, global_bs)
            trainer.train()
        else:
            raise
    
    # Save the DAPT-adapted model
    trainer.save_model(DAPT_SAVE_PATH)
    print(f"\nDAPT complete. Adapted model saved to: {DAPT_SAVE_PATH}")


if __name__ == "__main__":
    run_dapt()