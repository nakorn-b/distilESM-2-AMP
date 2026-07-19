#!/usr/bin/env python3
"""
generate_finetune_configs.py
============================
Generates 9 finetune YAML configs (3 temps × 3 alphas) and a SLURM sweep.sh.

Usage:
    python generate_finetune_configs.py

Outputs:
    configs/finetune/t{T}_a{A}.yaml  (×9)
    finetune_sweep.sh
"""

import os
import yaml

TEMPS  = [0.2, 0.3, 0.5]
ALPHAS = [0.0, 0.5, 1.0]

BASE = {
    "train_data_path":    "dataset/AMP-dataset/AMP.csv",
    "test_data_path":     "dataset/test-dataset/abp.indtest1.csv",
    "val_split":          0.2,
    "max_length":         256,
    "space_separate_seq": False,
    "learning_rate":      3e-5,
    "num_epochs":         10,
    "batch_size":         32,
    "weight_decay":       0.05,
    "warmup_steps":       20,
    "grad_accum_steps":   1,
    "early_stopping_patience": 3,
    "fp16":               True,
    "output_dir":         "./outputs/finetune",
    "mlflow_experiment":  "distilESM2-AMP-finetune",
    "seed":               42,
}

os.makedirs("configs/finetune", exist_ok=True)

config_paths = []

for temp in TEMPS:
    for alpha in ALPHAS:
        run_name   = f"t{temp}_a{alpha}"
        model_path = f"./outputs/ablation/t{temp}_a{alpha}/student_model"
        model_name = f"distilESM-2-AMP (t={temp}, a={alpha})"

        cfg = {
            **BASE,
            "run_name":   run_name,
            "model_name": model_name,
            "model_path": model_path,
        }

        path = f"configs/finetune/t{temp}_a{alpha}.yaml"
        with open(path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

        config_paths.append(path)
        print(f"[+] Generated: {path}")

print(f"\n[INFO] {len(config_paths)} configs generated.")

# ── SLURM sweep ───────────────────────────────────────────────────────────────

SLURM_HEADER = """#!/bin/bash
# finetune_sweep.sh — Submit one SLURM job per finetune config
# Usage: bash finetune_sweep.sh
"""

JOB_TEMPLATE = """
echo "Submitting: {config_path}"
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=ft_{run_name}
#SBATCH --output=logs/ft_{run_name}_%j.out
#SBATCH --error=logs/ft_{run_name}_%j.err
#SBATCH --time=1-0:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

source ~/.bashrc
conda activate MLM

python finetune.py --config {config_path}
EOF
"""

os.makedirs("logs", exist_ok=True)

with open("finetune_sweep.sh", "w") as f:
    f.write(SLURM_HEADER)
    for path in config_paths:
        run_name = path.split("/")[-1].replace(".yaml", "")
        f.write(JOB_TEMPLATE.format(config_path=path, run_name=run_name))

print("[INFO] finetune_sweep.sh generated. Run with: bash finetune_sweep.sh")
print("[INFO] Monitor runs: mlflow ui --port 5000")