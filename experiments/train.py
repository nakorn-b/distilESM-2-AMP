from transformers import (
    EsmConfig,
    EsmForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    set_seed
)
import random
import yaml
import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
from datasets import Dataset, load_from_disk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, matthews_corrcoef
import numpy as np
import wandb
import os
import time
import mlflow
import mlflow.pytorch

def set_everything_seed(seed):
  """
  Set seed for everything for full reproducibility.
  """
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  set_seed(seed) # HuggingFace trainer seed
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


# 1. Config Loader
def load_config(config_path: str):
  """
  Loads a YAML config file and validates required keys for each ablations hyperparameters
  """
  with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

  required = ["temperature", "alpha", "seed", "output_dir", "data_path", "run_name"]
  missing = [k for k in required if k not in cfg]
  if missing:
      raise ValueError(f"Config missing required keys: {missing}")

  return cfg


# 2. Model initialization
def build_teacher(model_name: str = "facebook/esm2_t6_8M_UR50D"):
  # teacher model: ESM 8M
  # Original Model: https://huggingface.co/facebook/esm2_t6_8M_UR50D
  teacher = EsmForMaskedLM.from_pretrained('facebook/esm2_t6_8M_UR50D')
  teacher.eval()
  for param in teacher.parameters():
    param.require_grads = False

  return teacher

def build_student(teacher: EsmForMaskedLM):
  # Student Model
  # More information on ESMConfig: https://huggingface.co/docs/transformers/en/model_doc/esm#transformers.EsmConfig
  student_config = EsmConfig.from_pretrained('facebook/esm2_t6_8M_UR50D')
  student_config.num_hidden_layers = 3
  student = EsmForMaskedLM(student_config)

  # Initialize weights from teacher
  student.esm.embeddings.load_state_dict(teacher.esm.embeddings.state_dict())

  # Copy Encoder Layers (student encoder index[0, 1, 2] = teacher encoder layer[0, 2, 4])
  for i in range(3):
    student.esm.encoder.layer[i].load_state_dict(teacher.esm.encoder.layer[i*2].state_dict())

  # lm_head for training MLM
  student.lm_head.load_state_dict(teacher.lm_head.state_dict())

  return student

# def tokenize_function(examples):
#   return tokenizer(examples['seq'], padding="max_length", truncation=True, max_length=256)

# 3. Customed Trainer with KD Loss
# For more information on Custom trainer: https://huggingface.co/docs/transformers/main/en/trainer
class DistilledESMTrainer(Trainer):
  def __init__(self, teacher:EsmForMaskedLM, temperature:float, alpha:float, log_frequency:int = 100, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.teacher = teacher
    self.temperature = temperature
    self.alpha = alpha

    self.train_preds = []
    self.train_labels = []
    self.steps_since_log = 0
    self.log_frequency = log_frequency

  def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
    """
    Compute combined KD + MLM Loss
    """
    labels = inputs.get("labels")
    mask = labels != -100

    # Student forward pass
    outputs_student = model(**inputs, output_hidden_states=True)
    # MLM loss
    loss_mlm = outputs_student.loss
    
    # Teacher forward pass (no gradients needed)
    with torch.no_grad():
      outputs_teacher = self.teacher(**inputs, output_hidden_states=True)

    # KD loss
    # Scale student and teacher logits with temperature before softmax
    student_logits = outputs_student.logits / self.temperature
    teacher_logits = outputs_teacher.logits / self.temperature


    loss_kl = F.kl_div(
        F.log_softmax(student_logits[mask], dim=-1),
        F.softmax(teacher_logits[mask], dim=-1), reduction="batchmean") * (self.temperature ** 2)

    # Combined loss
    loss = self.alpha * loss_kl + (1-self.alpha) * loss_mlm

    # Accumulate training predictions for periodic metric logging
    with torch.no_grad():
      preds = torch.argmax(outputs_student.logits, dim=-1)
      preds_flat = preds.cpu().numpy().flatten()
      labels_flat = labels.cpu().numpy().flatten()

      mask_flat = labels_flat != -100

      self.train_preds.extend(preds_flat[mask_flat])
      self.train_labels.extend(labels_flat[mask_flat])

    self.steps_since_log += 1

    if self.steps_since_log >= self.log_frequency:
      if len(self.train_labels) > 0:
        train_accuracy = accuracy_score(self.train_labels, self.train_preds)
        train_mcc = matthews_corrcoef(self.train_labels, self.train_preds)

        self.log({
              "train/accuracy": train_accuracy,
              "train/mcc": train_mcc,
              "train/loss_mlm": loss_mlm.item(),
              "train/loss_kl": loss_kl.item(),
              "train/loss_total": loss.item()
            })
        
        # Reset buffers
        self.train_preds = []
        self.train_labels = []
        self.steps_since_log = 0

    return (loss, outputs_student) if return_outputs else loss
  

def compute_metrics(eval_pred):
      logits, labels = eval_pred

      if isinstance(logits, (tuple, list)):
          logits = logits[0]

      # Get predictions: argmax over vocabulary (last dimension)
      preds = np.argmax(logits, axis=-1)  # axis=-1 for vocab dimension

      # Flatten both predictions and labels
      preds = preds.flatten()
      labels = labels.flatten()

      # Filter out ignored tokens (-100)
      mask = labels != -100  # -100 is the ignore index
      preds = preds[mask]
      labels = labels[mask]

      # Calculate accuracy
      acc = accuracy_score(labels, preds)
      return {"accuracy": acc}

# MLM Training Session

def main():
  parser = argparse.ArgumentParser(description="DistilESM-2-AMP Training")
  parser.add_argument("--config", required=True, help="Path to YAML config file")
  args = parser.parse_args()

  cfg = load_config(args.config)

  set_everything_seed(cfg['seed'])

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"[INFO] Using device: {device}")
  print(f"[INFO] Config: temperature={cfg['temperature']}, alpha={cfg['alpha']}, seed={cfg['seed']}")

  # Dataset
  dataset = load_from_disk(cfg["data_path"])['train']
  subset_size = cfg.get("data_subset_size", None)
  if subset_size:
     dataset = dataset.shuffle(
        seed=cfg['data_subset_seed']
     ).select(range(subset_size))

  print(f"[INFO] Dataset loaded. Using {dataset.shape[0]} sequence subset.")

  os.environ["WANDB_PROJECT"]="distil-esm2"
  os.environ["WANDB_WATCH"]="false"

  # Tokenize
  tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D') # Tokenzier for ESM

  teacher_model = build_teacher().to(device)
  student_model = build_student(teacher_model).to(device)

  data_collator = DataCollatorForLanguageModeling(
      tokenizer=tokenizer,
      mlm_probability=0.15,
  )

  # Output directory
  run_output_dir = os.path.join(cfg["output_dir"], cfg["run_name"])
  os.makedirs(run_output_dir, exist_ok=True)

  training_args = TrainingArguments(
      output_dir=run_output_dir,
      seed=cfg["seed"],
      # Batch and hyperparameter
      per_device_train_batch_size=cfg.get("batch_size", 64),
      learning_rate=cfg.get("learning_rate", 5e-5),
      num_train_epochs=cfg.get("num_epochs", 3),
      warmup_steps=cfg.get("warmup_steps", 1000),
      # Evaluation
      eval_strategy='steps',
      per_device_eval_batch_size=cfg.get("eval_batch_size", 16),
      eval_accumulation_steps=cfg.get("eval_accumulation_steps", 16),
      eval_steps=cfg.get("eval_steps", 10000), 
      metric_for_best_model='loss',
      greater_is_better=False, # False for loss
      load_best_model_at_end=True,
      per_device_eval_batch_size=32,
      # Logging
      logging_first_step=True,
      logging_strategy='steps',
      logging_steps=100,
      report_to='none',
      # Saving
      save_strategy='steps',
      save_steps=cfg.get("save_steps", 50000),
      save_total_limit=3,
      # Performance
      fp16=True,
      gradient_checkpointing=True,
      gradient_accumulation_steps=cfg.get("grad_accum_steps", 4),
      # Memory Optimization
      eval_accumulation_steps=4,
     
      disable_tqdm=False,
      resume_from_checkpoint=False,
  )
  
  trainer = DistilledESMTrainer(
      teacher=teacher_model,
      temperature=cfg["temperature"],
      alpha=cfg["alpha"],
      model=student_model,
      args=training_args,
      data_collator=data_collator,
      train_dataset=dataset,
      eval_dataset=None,
      tokenizer=tokenizer,
      compute_metrics=compute_metrics,
  )

  # MLflow experiment tracking
  mlflow.set_experiment(cfg.get("mlflow_experiment", "distilESM2-AMP-ablations"))

  with mlflow.start_run(run_name=cfg["run_name"]):
        # Log all hyperparameters
        mlflow.log_params({
            "temperature": cfg["temperature"],
            "alpha": cfg["alpha"],
            "seed": cfg["seed"],
            "batch_size": cfg.get("batch_size", 256),
            "learning_rate": cfg.get("learning_rate", 5e-5),
            "num_epochs": cfg.get("num_epochs", 3),
            "warmup_steps": cfg.get("warmup_steps", 1000),
        })

        # Tag for traceability
        mlflow.set_tag("config_path", args.config)
        mlflow.set_tag("slurm_job_id", os.environ.get("SLURM_JOB_ID", "local"))
        mlflow.set_tag("device", str(device))

        # Training
        start = time.time()
        train_result = trainer.train()
        elapsed = time.time() - start
        print(f"[INFO] Training completed in {elapsed/60:.2f} minutes")

        # Log training summary metrics
        mlflow.log_metrics({
            "train_runtime_minutes": elapsed / 60,
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
        })

        eval_results = trainer.evaluate()
        print(f"[INFO] Final eval results: {eval_results}")

        mlflow.log_metrics({
          "eval/loss": eval_results.get("eval_loss", -1),
          "eval/accuracy": eval_results.get("eval_accuracy", -1),
          "eval/mcc": eval_results.get("eval_mcc", -1),
          "eval/f1": eval_results.get("eval_f1", -1),
          "eval/precision": eval_results.get("eval_precision", -1),
          "eval/recall": eval_results.get("eval_recall", -1),
          })

      
        # Save model
        save_dir = os.path.join(run_output_dir, "student_model")
        student_model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

        # Log model artifacts
        mlflow.log_artifacts(save_dir, artifact_path="student_model")
        mlflow.log_artifact(args.config, artifact_path="config")

        print(f"[INFO] Artifacts logged to MLflow run: {mlflow.active_run().info.run_id}")

  print("[DONE] Run complete.")

if __name__ == "__main__":
  main()
