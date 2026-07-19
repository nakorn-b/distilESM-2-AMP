#!/usr/bin/env python3
"""
generate_ablations.py
=====================
Auto-generates all 15 ablation configs (3 temps × 5 alphas)
and a SLURM sweep script that submits one job per config.

Usage:
    python generate_ablations.py

Outputs:
    configs/ablations/t{T}_a{A}.yaml  (×15)
    sweep.sh
"""

import os
import yaml

TEMPS  = [0.2, 0.3, 0.5]
ALPHAS = [0.0, 0.5, 1.0]

# Base config shared by all runs
BASE = {
    "data_path": "../tokenized_data/",
    "teacher_model": "facebook/esm2_t6_8M_UR50D",
    "student_num_layers": 3,
    "batch_size": 256,
    "learning_rate": 5e-5,
    "num_epochs": 3,
    "warmup_steps": 500,
    "grad_accum_steps": 1,
    "save_strategy": "no",
    "eval_strategy": "no",
    "mlflow_experiment": "distilESM2-AMP-ablations",
    "seed": 42,
    "output_dir": "./outputs/ablation",
    "data_subset_size": 800000,
    "data_subset_seed": 42

}

os.makedirs("configs/ablations", exist_ok=True)

config_paths = []

for temp in TEMPS:
    for alpha in ALPHAS:
        # Format: t0.2_a0.25, t0.5_a1.0, etc.
        t_str = f"{temp}".replace(".", "")  
        a_str = f"{alpha}".replace(".", "")  
        run_name = f"t{temp}_a{alpha}"

        cfg = {**BASE, "temperature": temp, "alpha": alpha, "run_name": run_name}

        path = f"configs/ablations/t{temp}_a{alpha}.yaml"
        with open(path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

        config_paths.append(path)
        print(f"[+] Generated: {path}")

print(f"\n[INFO] {len(config_paths)} configs generated.")

# ── Generate SLURM sweep script ───────────────────────────────────────────────

SLURM_HEADER = """#!/bin/bash
# sweep.sh — Submit one SLURM job per ablation config
# Usage: bash sweep.sh
# Each job runs independently; results tracked in MLflow.
"""

JOB_TEMPLATE = """
echo "Submitting: {config_path}"
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=distil_{run_name}
#SBATCH --output=logs/{run_name}_%j.out
#SBATCH --error=logs/{run_name}_%j.err
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

source ~/.bashrc
conda activate MLM

python proxy.py --config {config_path}
EOF
"""

os.makedirs("logs", exist_ok=True)

with open("sweep.sh", "w") as f:
    f.write(SLURM_HEADER)
    for path in config_paths:
        run_name = path.split("/")[-1].replace(".yaml", "")
        f.write(JOB_TEMPLATE.format(config_path=path, run_name=run_name))

print("[INFO] sweep.sh generated.")
