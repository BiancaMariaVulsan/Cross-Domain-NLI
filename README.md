
# Environment Setup
python -m venv cross_domain_nli_env

Linux
```source cross_domain_nli_env/bin/activate```

Windows
```cross_domain_nli_env\Scripts\activate```

```
pip install torch transformers datasets scikit-learn
pip install evaluate

# Optional: PEFT (LoRA)
pip install peft

# To enable training on my local dedicated gpu. (RTX 4060)
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

# Run the fine-tunning (or use launch.json)
```
cd src
python .\train_baseline.py
python .\train_baseline.py --output-subdir roberta-base_baseline_multi_nli_custom
```

# Run LoRA fine-tuning (or use launch.json)
```
python .\train_lora.py --base-model-dir roberta-base --r 16 --lora-alpha 32 --lora-dropout 0.1 --merge-and-save
python .\train_lora.py --base-model-dir roberta-base --output-subdir my_lora_run --r 16 --lora-alpha 32 --lora-dropout 0.1
```

# Evaluation

- dapt
```
python evaluate_ood.py --model-dir roberta-base_dapt_dapt_corpus --label dapt --results-file dapt_results
```

- LoRA adapter evaluation (attach adapter to base)
```
python evaluate_ood.py --adapter-dir models/roberta-base_lora_multi_nli/checkpoint-73632 --label lora --results-file lora_results
```

- LoRA merged model evaluation (if you used --merge-and-save)
```
python evaluate_ood.py --model-dir models/roberta-base_lora_multi_nli/merged --label lora-merged --results-file lora_merged_results
```

# Results

| Method                    | Eval Domain   | Accuracy | F1-Macro | Generalization Gap |
| ------------------------- | ------------- | -------- | -------- | ------------------ |
| **Baseline (Full FT)**    | MultiNLI (In) | 87.43%   | 87.38%   | 0.00 p.p.          |
|                           | SciTail (OOD) | 70.41%   | 27.54%   | 17.02 p.p.         |
| **DAPT + Full FT**        | MultiNLI (In) | 87.54%   | 87.49%   | 0.00 p.p.          |
|                           | SciTail (OOD) | 70.41%   | 49.79%   | 17.13 p.p.         |
| **LoRA Fine-Tuning**      | MultiNLI (In) | 87.43%   | 87.38%   | 0.00 p.p.          |
|                           | SciTail (OOD) | 70.13%   | 49.85%   | 17.30 p.p.         |
| **LoRA DAPT Fine-Tuning** | MultiNLI (In) | 83.90%   | 83.78%   | 0.00 p.p.          |
|                           | SciTail (OOD) | 68.53%   | 48.33%   | 15.37 p.p.         |


The F1-Macro "Quality" Jump: While raw accuracy on SciTail remained stagnant at ~70%, both DAPT and LoRA nearly doubled the F1-Macro score compared to the baseline. This indicates that the baseline was heavily biased (likely predicting a single majority class), whereas the adapted models learned to distinguish between Entailment, Neutral, and Contradiction in a scientific context.

DAPT Effectiveness: Domain-Adaptive Pretraining successfully "sensitized" the model to scientific vocabulary, providing a slight boost to in-domain performance while significantly balancing OOD (out of domain) predictions.

LoRA Efficiency: LoRA (Low-Rank Adaptation) matched DAPT’s F1-Macro performance while training <1% of parameters. By restricting updates to low-rank matrices, LoRA acted as a regularizer, preventing the model from over-fitting to MultiNLI quirks and allowing for cleaner domain transfer.

LoRA DAPT Hybrid (The Robustness Winner): This hybrid achieved the lowest Generalization Gap (15.37 p.p.). By combining a domain-aware backbone (DAPT) with restricted tuning (LoRA), the model became the most domain-consistent, though it suffered from "Catastrophic Interference," leading to a slight drop in in-domain accuracy.

The Reasoning Gap: A ~17% Accuracy Gap persists across all methods. This proves that while we solved the lexical shift (vocabulary), the logical reasoning shift (how scientific evidence is structured) remains a fundamental challenge that vocabulary alone cannot fix.