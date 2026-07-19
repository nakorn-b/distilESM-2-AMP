# Experiment Results Summary

## Phase 1 — Config Ablation Study

Ablation was performed on **10% of the training data** across 9 configurations, sweeping temperature `T ∈ {0.2, 0.3, 0.5}` and alpha `α ∈ {0.0, 0.5, 1.0}`. Each pretrained checkpoint was then finetuned and evaluated on the test set. The selection metric is **MCC**.

### Pretraining Metrics (10% data)

| Config | Train Accuracy | Loss MLM | Loss KL | Loss Total | Runtime (min) |
|--------|---------------|----------|---------|------------|---------------|
| T=0.2, α=0.0 | 0.2345 | 2.5355 | 0.0425 | 2.5355 | 250.3 |
| T=0.2, α=0.5 | 0.2335 | 2.5422 | 0.0282 | 1.2852 | 243.8 |
| T=0.2, α=1.0 | 0.2309 | 2.6925 | 0.0226 | 0.0226 | 242.8 |
| T=0.3, α=0.0 | 0.2345 | 2.5355 | 0.0629 | 2.5355 | 244.4 |
| T=0.3, α=0.5 | 0.2337 | 2.5425 | 0.0426 | 1.2926 | 243.3 |
| T=0.3, α=1.0 | 0.2321 | 2.6144 | 0.0380 | 0.0380 | 244.6 |
| T=0.5, α=0.0 | 0.2345 | 2.5355 | 0.1021 | 2.5355 | 256.2 |
| T=0.5, α=0.5 | 0.2339 | 2.5421 | 0.0709 | 1.3065 | 241.9 |
| T=0.5, α=1.0 | 0.2337 | 2.5586 | 0.0670 | 0.0670 | 244.8 |

> When α=0.0, the KL loss has no weight so `loss_total = loss_mlm`. When α=1.0, only KL loss is used.

### Finetuning Results (Config Selection by MCC)

| Config | Accuracy | Precision | Recall | F1 | MCC | ROC-AUC |
|--------|----------|-----------|--------|----|-----|---------|
| T=0.2, α=0.0 | 0.9810 | 0.9781 | 0.9840 | 0.9811 | 0.9620 | 0.9975 |
| T=0.2, α=0.5 | 0.9750 | 0.9666 | 0.9840 | 0.9752 | 0.9502 | 0.9975 |
| T=0.2, α=1.0 | 0.9770 | 0.9704 | 0.9840 | 0.9772 | 0.9541 | 0.9966 |
| T=0.3, α=0.0 | 0.9810 | 0.9781 | 0.9840 | 0.9811 | 0.9620 | 0.9975 |
| T=0.3, α=0.5 | 0.9760 | 0.9685 | 0.9840 | 0.9762 | 0.9521 | 0.9976 |
| T=0.3, α=1.0 | 0.9760 | 0.9685 | 0.9840 | 0.9762 | 0.9521 | 0.9969 |
| T=0.5, α=0.0 | 0.9810 | 0.9781 | 0.9840 | 0.9811 | 0.9620 | 0.9975 |
| **T=0.5, α=0.5** | **0.9820** | **0.9782** | **0.9860** | **0.9821** | **0.9640** | **0.9975** |
| T=0.5, α=1.0 | 0.9790 | 0.9743 | 0.9840 | 0.9791 | 0.9580 | 0.9973 |

**Best config: T=0.5, α=0.5** — highest MCC (0.9640) and F1 (0.9821).

---

## Phase 3 — Final Model Comparison

The best config (T=0.5, α=0.5) was used to pretrain on the **full dataset**, then finetuned with **5 random seeds** (42–46). Results are averaged and compared against DistilProtBert and facebook/ESM2-8M, both finetuned under the same protocol.

### Averaged Results (5 Seeds)

| Model | #Params | Size (MB) | Accuracy | Precision | Recall | F1 | MCC | ROC-AUC |
|-------|---------|-----------|----------|-----------|--------|----|-----|---------|
| **distilESM2 (T=0.5, α=0.5)** | 3.81M | 14.56 | 0.9746 ± 0.0034 | 0.9652 ± 0.0081 | 0.9848 ± 0.0027 | 0.9749 ± 0.0032 | 0.9495 ± 0.0066 | 0.9970 ± 0.0008 |
| DistilProtBert | 230.99M | 881.77 | 0.9778 ± 0.0033 | 0.9743 ± 0.0070 | 0.9816 ± 0.0029 | 0.9779 ± 0.0032 | 0.9557 ± 0.0066 | 0.9961 ± 0.0012 |
| facebook/ESM2-8M | 7.51M | 28.67 | 0.9778 ± 0.0039 | 0.9742 ± 0.0057 | 0.9816 ± 0.0050 | 0.9779 ± 0.0039 | 0.9557 ± 0.0078 | 0.9960 ± 0.0010 |

### Per-Seed MCC Breakdown

| Seed | distilESM2 (T=0.5, α=0.5) | DistilProtBert | facebook/ESM2-8M |
|------|--------------------------|----------------|-----------------|
| 42 | 0.9463 | 0.9520 | 0.9562 |
| 43 | 0.9581 | 0.9540 | 0.9640 |
| 44 | 0.9560 | 0.9462 | 0.9500 |
| 45 | 0.9406 | 0.9640 | 0.9441 |
| 46 | 0.9462 | 0.9620 | 0.9640 |
| **Mean** | **0.9495** | **0.9557** | **0.9557** |
| **Std** | **0.0066** | **0.0066** | **0.0078** |
