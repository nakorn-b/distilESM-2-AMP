import os
import random
import argparse
import time
 
import yaml
import numpy as np
import torch
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import pandas as pd
 
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed as hf_set_seed,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path: str) -> dict:
    """Load and validate a YAML fine-tune config."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
 
    required = [
        "model_path", "train_data_path", "test_data_path",
        "output_dir", "run_name"
    ]
    missing = [k for k in required if k not in cfg]

    if missing:
        raise ValueError(f"Config missing required keys: {missing}")
     
    if "seeds" in cfg:
        cfg["seed"] = cfg["seeds"][0]
    elif "seed" in cfg:
        cfg["seeds"] = [cfg["seed"]]
    else:
        raise ValueError("Config must specify either 'seeds' (list) or 'seed' (int)")


    return cfg


def load_datasets(cfg: dict, seed:int):
    """
    Load train/test CSVs.
    Expects columns: seq_name (amino-acid sequence), label (0/1).
    DistilProtBert-style space-separated sequences are handled automatically
    if cfg['space_separate_seq'] is True.
    """
    def _load(path):
        df = pd.read_csv(path)
        if cfg.get("space_separate_seq", False):
            df["seq_name"] = df["seq_name"].str.replace(r"[UZOB]", "X", regex=True)
            df["seq_name"] = df["seq_name"].apply(lambda x: " ".join(list(x)))
        return Dataset.from_pandas(df).rename_column("label", "labels")
 
    full = _load(cfg["train_data_path"])
    split = full.train_test_split(
        test_size=cfg.get("val_split", 0.2),
        seed=seed,
    )
    test_dataset = _load(cfg["test_data_path"])
    return split, test_dataset
 

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall, "mcc": mcc}

def log_model_size(model, model_name: str) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buf_bytes   = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_bytes + buf_bytes) / (1024 ** 2)
 
    print(f"\n{'='*50}")
    print(f"{model_name} — Size Analysis")
    print(f"  Total Parameters : {total:,} ({total/1e6:.2f}M)")
    print(f"  Trainable : {trainable:,} ({trainable/1e6:.2f}M)")
    print(f"  In-memory size : {size_mb:.2f} MB")
    print(f"{'='*50}\n")
 
    return {"total_params": total, "trainable_params": trainable, "model_size_mb": size_mb}

def evaluate_on_test(trainer: Trainer, test_dataset, output_dir: str) -> dict:
    """Evaluate on test set: classification metrics + confusion matrix + ROC curve."""
    model = trainer.model
    model.eval()
    device = model.device

    dataloader = trainer.get_eval_dataloader(test_dataset)
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            labels_batch = batch["labels"].cpu().numpy()
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            outputs = model(**batch)

            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels_batch)

    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    mcc = matthews_corrcoef(all_labels, all_preds)

    print("\n========== Evaluation on Test Set ==========")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall : {recall:.4f}")
    print(f"F1 : {f1:.4f}")
    print(f"MCC : {mcc:.4f}")
    print("=" * 44 + "\n")

    os.makedirs(output_dir, exist_ok=True)

    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        confusion_matrix(all_labels, all_preds),
        display_labels=["non-AMP", "AMP"]
    ).plot(ax=ax)
    plt.title("Confusion Matrix")
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    roc_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(roc_path, bbox_inches="tight")
    plt.close()

    return {
        "test/accuracy": acc,
        "test/precision": precision,
        "test/recall": recall,
        "test/f1": f1,
        "test/mcc": mcc,
        "test/roc_auc": roc_auc,
    }, cm_path, roc_path

def main():
    parser = argparse.ArgumentParser(description="AMP Fine-Tuning")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
 
    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device : {device}")
    print(f"[INFO] Run name : {cfg['run_name']}")
    print(f"[INFO] Model : {cfg['model_path']}")

    seeds = cfg.get("seeds")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(cfg.get("mlflow_experiment"))

    all_test_metrics = []

    with mlflow.start_run(run_name=cfg["run_name"]) as parant_run:
        mlflow.log_params({
            "model_path":    cfg["model_path"],
            "learning_rate": cfg.get("learning_rate", 2e-5),
            "num_epochs":    cfg.get("num_epochs", 5),
            "batch_size":    cfg.get("batch_size", 32),
            "weight_decay":  cfg.get("weight_decay", 0.01),
            "warmup_steps":  cfg.get("warmup_steps", 20),
            "max_length":    cfg.get("max_length", 256),
            "seed":          str(seeds),
            "val_split":     cfg.get("val_split", 0.2),
        })
        # mlflow.log_params(model_stats)
        mlflow.set_tag("config_path", args.config)
        mlflow.set_tag("device", str(device))
        mlflow.set_tag("slurm_job_id", os.environ.get("SLURM_JOB_ID", "local"))

        for i, seed in enumerate(seeds):
            print(f"\n{'='*60}")
            print(f"[INFO] Running seed {seed} ({i+1}/{len(seeds)})")
            print(f"{'='*60}")

            set_seed(seed)

            train_val_split, test_dataset = load_datasets(cfg, seed)

            tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"])

            def tokenize_fn(examples):
                return tokenizer(
                    examples["seq_name"],
                    padding="max_length",
                    truncation=True,
                    max_length=cfg.get("max_length", 256),
                )
            
            print("[INFO] Tokenizing datasets …")
            tokenized_train = train_val_split.map(tokenize_fn, batched=True)
            tokenized_test = test_dataset.map(tokenize_fn, batched=True)

            model = AutoModelForSequenceClassification.from_pretrained(
                cfg["model_path"], num_labels=2
            )
            model_stats = log_model_size(model, cfg["model_name"] if "model_name" in cfg else cfg["model_path"])

            seed_run_name  = f"{cfg['run_name']}_seed{seed}"
            seed_output_dir = os.path.join(cfg["output_dir"], cfg["run_name"], f"seed{seed}")
            os.makedirs(seed_output_dir, exist_ok=True)

            callbacks = []
            if cfg.get("early_stopping_patience"):
                callbacks.append(EarlyStoppingCallback(
                    early_stopping_patience=cfg["early_stopping_patience"]
                ))

            training_args = TrainingArguments(
                output_dir=seed_output_dir,
                seed=seed,
                data_seed=seed,
                # Batch & optimiser
                num_train_epochs=cfg.get("num_epochs", 5),
                per_device_train_batch_size=cfg.get("batch_size", 32),
                per_device_eval_batch_size=cfg.get("batch_size", 32),
                learning_rate=cfg.get("learning_rate", 2e-5),
                weight_decay=cfg.get("weight_decay", 0.01),
                warmup_steps=cfg.get("warmup_steps", 20),
                # Evaluation & saving
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="eval_mcc",
                greater_is_better=True,
                # Logging
                logging_strategy="epoch",
                report_to="mlflow",
                # Performance
                fp16=cfg.get("fp16", False),
                gradient_accumulation_steps=cfg.get("grad_accum_steps", 1),
                disable_tqdm=False,
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train["train"],
                eval_dataset=tokenized_train["test"],
                processing_class=tokenizer,
                compute_metrics=compute_metrics,
                callbacks=callbacks if callbacks else None,
            )

            with mlflow.start_run(run_name=seed_run_name, nested=True):
                mlflow.log_params({"seed": seed})
                mlflow.log_params(model_stats)
                mlflow.set_tag("parent_run_name", cfg["run_name"])



                # Train
                t0 = time.time()
                train_result = trainer.train()
                elapsed = time.time() - t0
                print(f"[INFO] Training done in {elapsed/60:.1f} min")

                mlflow.log_metrics({
                    "train_runtime_minutes":      elapsed / 60,
                    "train_samples_per_second":   train_result.metrics.get("train_samples_per_second", 0),
                })

                # Test evaluation
                test_metrics, cm_path, roc_path = evaluate_on_test(
                    trainer, tokenized_test, seed_output_dir
                )
                mlflow.log_metrics(test_metrics)
                mlflow.log_artifact(cm_path,  artifact_path="plots")
                mlflow.log_artifact(roc_path, artifact_path="plots")
                mlflow.log_artifact(args.config, artifact_path="config")

                # Save model
                save_dir = os.path.join(seed_output_dir, "model")
                trainer.save_model(save_dir)
                tokenizer.save_pretrained(save_dir)
                mlflow.log_artifacts(save_dir,   artifact_path="model")
                mlflow.log_artifact(args.config, artifact_path="config")
            
            all_test_metrics.append(test_metrics)
            print(f"[INFO] Seed {seed} — MCC: {test_metrics['test/mcc']:.4f} | F1: {test_metrics['test/f1']:.4f} | Accuracy: {test_metrics['test/accuracy']:.4f}")

        metric_keys = all_test_metrics[0].keys()
        agg = {}
        for key in metric_keys:
            vals = np.array([metric[key] for metric in all_test_metrics])
            agg[f"mean_{key}"] = float(np.mean(vals))
            agg[f"std_{key}"] = float(np.std(vals))

        mlflow.log_metrics(agg)

        print(f"\n{'='*60}")
        print(f"[SUMMARY] Results across {len(seeds)} seeds:")
        for k in metric_keys:
            print(f"{k:20s} mean={agg[f'mean_{k}']:.4f}  std={agg[f'std_{k}']:.4f}")
        print(f"{'='*60}")
 
    print("[DONE]")

if __name__ == "__main__":
    main()
