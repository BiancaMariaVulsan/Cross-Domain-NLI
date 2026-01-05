import os
import sys
from datasets import load_dataset
from transformers import AutoTokenizer
from config.default_config import MODEL_NAME, MAX_SEQ_LENGTH, BASE_DIR

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# Configuration for DAPT
PRIMARY_DATASET = ("the_pile", "arXiv", "text")  # (hub_name, config, text_field)
FALLBACK_DATASET = ("cc_news", None, "text")     # widely available news dataset
CORPUS_SIZE_LIMIT = 200000
DAPT_OUTPUT_PATH = os.path.join(BASE_DIR, "data", "dapt_corpus_tokenized")

# Local HF cache to reduce Windows file locks
HF_HOME = os.path.join(BASE_DIR, ".hf_cache")
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(HF_HOME, "datasets"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def prepare_dapt_corpus():
    """
    Loads a DAPT corpus (arXiv subset of The Pile if available, else CC-News),
    tokenizes, and groups for MLM.
    """
    def try_load(hub, config, split):
        return load_dataset(hub, config, split=split) if config else load_dataset(hub, split=split)

    # 1) Try primary corpus
    hub, config, text_field = PRIMARY_DATASET
    split = f"train[:{CORPUS_SIZE_LIMIT}]"
    print(f"--- 1. Loading {hub}/{config or ''} Corpus (limit: {CORPUS_SIZE_LIMIT}) ---")
    try:
        raw_dataset = try_load(hub, config, split)
        active_text_field = text_field
    except Exception as e:
        print(f"Failed to load {hub}/{config}: {e}")
        # 2) Fallback
        hub, config, text_field = FALLBACK_DATASET
        print(f"Falling back to {hub}...")
        try:
            raw_dataset = try_load(hub, config, split)
            active_text_field = text_field
        except Exception as e2:
            print(f"Failed to load fallback {hub}: {e2}")
            print("Ensure internet access or install an older datasets version if you require deprecated scripts.")
            return

    # Filter out empty docs
    raw_dataset = raw_dataset.filter(
        lambda x: x.get(active_text_field) is not None and len(x[active_text_field].strip()) > 0
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"Number of documents after filtering: {len(raw_dataset)}")

    # 2) Tokenization
    def tokenize_function(examples):
        return tokenizer(examples[active_text_field], truncation=False)

    tokenized_datasets = raw_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=1,  # Windows-friendly
        remove_columns=raw_dataset.column_names,
        load_from_cache_file=True,
        desc="Tokenizing corpus"
    )

    # 3) Group texts into MAX_SEQ_LENGTH chunks for MLM
    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(next(iter(concatenated.values())))
        total_length = (total_length // MAX_SEQ_LENGTH) * MAX_SEQ_LENGTH
        return {
            k: [t[i:i + MAX_SEQ_LENGTH] for i in range(0, total_length, MAX_SEQ_LENGTH)]
            for k, t in concatenated.items()
        }

    print(f"--- 2. Grouping texts into blocks of {MAX_SEQ_LENGTH} tokens for MLM ---")
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=1,
        load_from_cache_file=True,
        desc="Grouping into fixed-length blocks"
    )

    # Save prepared dataset
    os.makedirs(os.path.dirname(DAPT_OUTPUT_PATH), exist_ok=True)
    lm_datasets.save_to_disk(DAPT_OUTPUT_PATH)
    print(f"\nDAPT Corpus preparation complete.")
    print(f"Total blocks for MLM training: {len(lm_datasets)}")
    print(f"Saved to: {DAPT_OUTPUT_PATH}")

if __name__ == "__main__":
    prepare_dapt_corpus()