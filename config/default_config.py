import os

# Project Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# Model & Data Settings
MODEL_NAME = "roberta-base"
TRAIN_DATASET = "multi_nli"
# Use canonical HF names and explicit splits
OOD_DATASETS = [
    {"name": "scitail", "split": "test"},   # SciTail test split
    {"name": "mednli",  "split": "test_r3"} # MedNLI r3 test split
]
MAX_SEQ_LENGTH = 128
SEED = 42


# Training Hyperparameters
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01


# NLI Label Mapping
# Ensures consistent mapping for the 3-class NLI task across all datasets
LABEL_MAPPING = {
    "entailment": 0,
    "entails": 0,  # SciTail often uses 'entails'
    "neutral": 1,
    "contradiction": 2,
    "not_entailment": 2,  # Used in some simplified NLI tasks
    "not_entails": 2,     # SciTail variant if present
    "-1": 1  # MultiNLI sometimes uses '-1' for the 'neutral' label
}


# Evaluation Settings
METRIC_NAME = "accuracy"