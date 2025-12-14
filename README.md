
# Prepare the environment
python -m venv cross_domain_nli_env

Linux
source cross_domain_nli_env/bin/activate

Windows
cross_domain_nli_env\Scripts\activate

pip install torch transformers datasets scikit-learn
pip install evaluate

# Run the training
cd src
python .\train_baseline.py