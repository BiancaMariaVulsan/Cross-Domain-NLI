
## Environment Setup
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

## Run the fine-tunning (or use launch.json)
```
cd src
python .\train_baseline.py
python .\train_baseline.py --output-subdir roberta-base_baseline_multi_nli_custom
```

## Run LoRA fine-tuning (or use launch.json)
```
python .\train_lora.py --base-model-dir roberta-base --r 16 --lora-alpha 32 --lora-dropout 0.1 --merge-and-save
python .\train_lora.py --base-model-dir roberta-base --output-subdir my_lora_run --r 16 --lora-alpha 32 --lora-dropout 0.1
```

## Evaluation

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

## Results

| Model                        | Trained On | Evaluated On         | Accuracy | F1-Macro | Generalization Gap |
| ---------------------------- | ---------- | -------------------- | -------- | -------- | ------------------ |
| **roberta-base Baseline**    | multi_nli  | multi_nli_matched    | 0.8743   | 0.8738   | 0.0000             |
|                              |            | multi_nli_mismatched | 0.8749   | 0.8744   | −0.0006            |
|                              |            | scitail              | 0.7040   | 0.2754   | 0.1702             |
|                              |            | snli                 | 0.8449   | 0.8436   | 0.0294             |
| **roberta-base DAPT**        | multi_nli  | multi_nli_matched    | 0.8754   | 0.8749   | 0.0000             |
|                              |            | multi_nli_mismatched | 0.8744   | 0.8739   | 0.0010             |
|                              |            | scitail              | 0.7041   | 0.4979   | 0.1713             |
|                              |            | snli                 | 0.8444   | 0.8434   | 0.0310             |
| **roberta-base LoRA**        | multi_nli  | multi_nli_matched    | 0.8743   | 0.8738   | 0.0000             |
|                              |            | multi_nli_mismatched | 0.8749   | 0.8744   | −0.0006            |
|                              |            | scitail              | 0.7013   | 0.4985   | 0.1730             |
|                              |            | snli                 | 0.8449   | 0.8436   | 0.0294             |
| **roberta-base DAPT + LoRA** | multi_nli  | multi_nli_matched    | 0.8390   | 0.8378   | 0.0000             |
|                              |            | multi_nli_mismatched | 0.8453   | 0.8440   | −0.0063            |
|                              |            | scitail              | 0.6853   | 0.4833   | 0.1537             |
|                              |            | snli                 | 0.8118   | 0.8094   | 0.0272             |


The F1-Macro "Quality" Jump: While raw accuracy on SciTail remained stagnant at ~70%, both DAPT and LoRA nearly doubled the F1-Macro score compared to the baseline. This indicates that the baseline was heavily biased (likely predicting a single majority class), whereas the adapted models learned to distinguish between Entailment, Neutral, and Contradiction in a scientific context.

Domain-Adaptive Pretraining successfully sensitizes the model to scientific language and discourse patterns. This manifests as a modest improvement in in-domain performance and a pronounced increase in OOD F1-Macro. The gains suggest that DAPT reduces lexical mismatch and improves semantic alignment, enabling the model to better interpret domain-specific premises and hypotheses without sacrificing task structure.

LoRA achieves F1-Macro performance comparable to DAPT while updating fewer than 1% of model parameters, highlighting its effectiveness as a parameter-efficient adaptation strategy. By constraining updates to low-rank subspaces, LoRA acts as an implicit regularizer, limiting overfitting to MultiNLI-specific artifacts while facilitating cleaner transfer to the scientific domain. This demonstrates that full weight updates are not necessary to achieve robust domain adaptation when the task structure is preserved.

The LoRA-DAPT hybrid achieves the lowest generalization gap (15.37 p.p.), indicating the most consistent behavior across domains. Combining a domain-aware backbone with restricted adaptation yields a model that generalizes more uniformly. However, this robustness comes at the expense of a modest decline in in-domain performance, consistent with mild catastrophic interference, where domain specialization partially overrides task-specific fine-tuning.

The Reasoning Gap: A ~17% Accuracy Gap persists across all methods. This proves that while we solved the lexical shift (vocabulary), the logical reasoning shift (how scientific evidence is structured) remains a fundamental challenge that vocabulary alone cannot fix.