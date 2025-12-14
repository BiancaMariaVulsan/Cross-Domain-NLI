
# Prepare the environment
python -m venv cross_domain_nli_env

Linux
source cross_domain_nli_env/bin/activate

Windows
cross_domain_nli_env\Scripts\activate

pip install torch transformers datasets scikit-learn
pip install evaluate

To enable training on my local dedicated gpu. (RTX 4060)
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124


# Run the training
cd src
python .\train_baseline.py