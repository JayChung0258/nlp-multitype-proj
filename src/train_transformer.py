"""
Transformer model training script for NLP multi-type classification.

This script trains HuggingFace transformer models for 4-class single-sentence
classification (T1/T2/T3/T4).

Supports: DistilBERT, BERT, RoBERTa, DeBERTa, ELECTRA, and other
AutoModelForSequenceClassification-compatible models.

Usage:
    python -m src.train_transformer --model_name distilbert-base-uncased
    python -m src.train_transformer --model_name bert-base-uncased --max_seq_length 256
    python -m src.train_transformer --model_name microsoft/deberta-v3-base --train_batch_size 8
"""

import argparse
import json
import time
import os
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any, Optional

import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from .constants import LABELS, LABEL2ID, ID2LABEL


# ============================================================
# DATA LOADING
# ============================================================

def load_config(config_path: str = "configs/data_config.yaml") -> dict:
    """
    Load data configuration from YAML file.
    
    Args:
        config_path: Path to data configuration YAML
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_splits(processed_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train/val/test splits from processed JSONL files.
    
    Args:
        processed_dir: Directory containing processed JSONL files
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    train_path = Path(processed_dir) / "train_4class.jsonl"
    val_path = Path(processed_dir) / "val_4class.jsonl"
    test_path = Path(processed_dir) / "test_4class.jsonl"
    
    print(f"Loading datasets from {processed_dir}...")
    train_df = pd.read_json(train_path, lines=True)
    val_df = pd.read_json(val_path, lines=True)
    test_df = pd.read_json(test_path, lines=True)
    
    print(f"  Train: {len(train_df):,} samples")
    print(f"  Val:   {len(val_df):,} samples")
    print(f"  Test:  {len(test_df):,} samples")
    
    return train_df, val_df, test_df


# ============================================================
# MODEL SETUP
# ============================================================

def build_tokenizer_and_model(
    model_name: str,
    num_labels: int = 4
) -> Tuple[Any, Any]:
    """
    Build tokenizer and model for sequence classification.
    
    Args:
        model_name: HuggingFace model identifier
        num_labels: Number of classification labels
        
    Returns:
        Tuple of (tokenizer, model)
    """
    print(f"\nLoading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model with classification head
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Tokenizer loaded: {tokenizer.__class__.__name__}")
    print(f"  Model loaded: {model.__class__.__name__}")
    print(f"  Trainable parameters: {num_params:,}")
    
    return tokenizer, model


# ============================================================
# DATA PREPROCESSING
# ============================================================

class TextClassificationDataset(Dataset):
    """
    PyTorch Dataset for text classification.
    """
    
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)


def tokenize_dataset(
    df: pd.DataFrame,
    tokenizer: Any,
    max_seq_length: int = 256
) -> TextClassificationDataset:
    """
    Tokenize text data and create PyTorch Dataset.
    
    Args:
        df: DataFrame with 'text' and 'label_id' columns
        tokenizer: HuggingFace tokenizer
        max_seq_length: Maximum sequence length for tokenization
        
    Returns:
        TextClassificationDataset object
    """
    texts = df['text'].tolist()
    labels = df['label_id'].tolist()
    
    # Tokenize texts
    encodings = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=max_seq_length,
        return_tensors=None  # Return lists, not tensors
    )
    
    dataset = TextClassificationDataset(encodings, labels)
    
    return dataset


# ============================================================
# METRICS COMPUTATION
# ============================================================

def compute_metrics_for_trainer(eval_pred):
    """
    Compute metrics for Trainer's eval during training.
    
    Args:
        eval_pred: EvalPrediction object with predictions and label_ids
        
    Returns:
        Dictionary with accuracy and macro_f1
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average='macro')
    
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1
    }


def compute_full_metrics(
    predictions: np.ndarray,
    labels: np.ndarray
) -> Dict[str, Any]:
    """
    Compute comprehensive metrics for evaluation.
    
    Args:
        predictions: Predicted class labels
        labels: True class labels
        
    Returns:
        Dictionary with all metrics
    """
    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average='macro')
    f1_per_class = f1_score(labels, predictions, average=None, labels=[0, 1, 2, 3])
    cm = confusion_matrix(labels, predictions, labels=[0, 1, 2, 3])
    
    # Convert per-class F1 to dictionary
    f1_dict = {
        "T1": float(f1_per_class[0]),
        "T2": float(f1_per_class[1]),
        "T3": float(f1_per_class[2]),
        "T4": float(f1_per_class[3])
    }
    
    metrics = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "f1_per_class": f1_dict,
        "confusion_matrix": cm.tolist()
    }
    
    return metrics


# ============================================================
# TRAINING
# ============================================================

def train_and_evaluate(
    model: Any,
    tokenizer: Any,
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    args: argparse.Namespace,
    output_dir: Path
) -> Tuple[Dict[str, Any], Dict[str, Any], float, float, int, float]:
    """
    Train model using HuggingFace Trainer and evaluate on val/test.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        args: Command-line arguments
        output_dir: Output directory for this model
        
    Returns:
        Tuple of (test_metrics, val_metrics, train_time_sec, eval_time_sec, num_params, model_size_mb)
    """
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        logging_dir=str(output_dir / "logs"),
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,  # Keep only best checkpoint
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=args.seed,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        report_to="none",  # Disable wandb/tensorboard
        disable_tqdm=False
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_for_trainer
    )
    
    # Train
    print(f"\nTraining {args.model_name}...")
    print(f"  Device: {training_args.device}")
    print(f"  Epochs: {args.num_train_epochs}")
    print(f"  Train batch size: {args.train_batch_size}")
    print(f"  Eval batch size: {args.eval_batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    
    start_time = time.perf_counter()
    train_result = trainer.train()
    end_time = time.perf_counter()
    
    train_time_sec = end_time - start_time
    
    print(f"\n✓ Training completed in {train_time_sec:.2f} seconds")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_predictions = trainer.predict(val_dataset)
    val_pred_labels = np.argmax(val_predictions.predictions, axis=-1)
    val_metrics = compute_full_metrics(val_pred_labels, val_predictions.label_ids)
    
    print(f"  Val Accuracy:  {val_metrics['accuracy']:.4f}")
    print(f"  Val Macro-F1:  {val_metrics['macro_f1']:.4f}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    start_eval_time = time.perf_counter()
    test_predictions = trainer.predict(test_dataset)
    end_eval_time = time.perf_counter()
    
    eval_time_sec = end_eval_time - start_eval_time
    
    test_pred_labels = np.argmax(test_predictions.predictions, axis=-1)
    test_metrics = compute_full_metrics(test_pred_labels, test_predictions.label_ids)
    
    print(f"  Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Test Macro-F1:  {test_metrics['macro_f1']:.4f}")
    print(f"  F1 per class:")
    for label, f1_val in test_metrics['f1_per_class'].items():
        print(f"    {label}: {f1_val:.4f}")
    
    # Save model temporarily to compute size
    temp_model_dir = output_dir / "model_temp"
    temp_model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(temp_model_dir)
    tokenizer.save_pretrained(temp_model_dir)
    
    # Compute model size
    model_size_bytes = sum(
        f.stat().st_size for f in temp_model_dir.rglob('*') if f.is_file()
    )
    model_size_mb = round(model_size_bytes / (1024 * 1024), 3)
    
    # Clean up temp directory
    import shutil
    shutil.rmtree(temp_model_dir)
    
    return test_metrics, val_metrics, train_time_sec, eval_time_sec, num_params, model_size_mb


# ============================================================
# SAVING ARTIFACTS
# ============================================================

def save_model_and_tokenizer(
    model: Any,
    tokenizer: Any,
    output_dir: Path
):
    """
    Save model and tokenizer to directory.
    
    Args:
        model: Trained HuggingFace model
        tokenizer: HuggingFace tokenizer
        output_dir: Directory to save model
    """
    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving model and tokenizer to {model_dir}...")
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    
    print(f"  ✓ Model saved")
    print(f"  ✓ Tokenizer saved")


def save_metrics_json(
    test_metrics: Dict[str, Any],
    val_metrics: Dict[str, Any],
    args: argparse.Namespace,
    train_samples: int,
    val_samples: int,
    test_samples: int,
    num_params: int,
    train_time_sec: float,
    eval_time_sec: float,
    model_size_mb: float,
    output_path: Path
):
    """
    Save comprehensive metrics to JSON file.
    
    Args:
        test_metrics: Test set metrics
        val_metrics: Validation set metrics
        args: Command-line arguments
        train_samples: Number of training samples
        val_samples: Number of validation samples
        test_samples: Number of test samples
        num_params: Number of model parameters
        train_time_sec: Training time in seconds
        eval_time_sec: Evaluation time on test in seconds
        model_size_mb: Model size in MB
        output_path: Path to save JSON file
    """
    model_slug = args.model_name.replace('/', '-')
    
    # Calculate throughput
    throughput = test_samples / eval_time_sec if eval_time_sec > 0 else 0
    
    output = {
        "model_name": args.model_name,
        "model_slug": model_slug,
        "timestamp_utc": datetime.now().isoformat() + "Z",
        "train_samples": train_samples,
        "val_samples": val_samples,
        "test_samples": test_samples,
        "max_seq_length": args.max_seq_length,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "seed": args.seed,
        "num_parameters": num_params,
        "accuracy_test": test_metrics["accuracy"],
        "macro_f1_test": test_metrics["macro_f1"],
        "f1_per_class_test": test_metrics["f1_per_class"],
        "confusion_matrix_test": test_metrics["confusion_matrix"],
        "accuracy_val": val_metrics["accuracy"],
        "macro_f1_val": val_metrics["macro_f1"],
        "f1_per_class_val": val_metrics["f1_per_class"],
        "train_time_sec": float(train_time_sec),
        "eval_time_test_sec": float(eval_time_sec),
        "throughput_samples_per_sec": float(throughput),
        "model_size_mb": model_size_mb
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"✓ Metrics saved to: {output_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    output_path: Path
):
    """
    Plot confusion matrix as heatmap and save to file.
    
    Args:
        cm: Confusion matrix (4x4 array)
        model_name: Model name for title
        output_path: Path to save PNG file
    """
    plt.figure(figsize=(8, 7))
    
    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=LABELS,
        yticklabels=LABELS,
        cbar_kws={'label': 'Count'}
    )
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(f'Confusion Matrix: {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Confusion matrix plot saved to: {output_path}")


def save_classification_report(
    test_metrics: Dict[str, Any],
    val_metrics: Dict[str, Any],
    args: argparse.Namespace,
    num_params: int,
    train_time_sec: float,
    eval_time_sec: float,
    output_path: Path
):
    """
    Save human-readable classification report to text file.
    
    Args:
        test_metrics: Test set metrics
        val_metrics: Validation set metrics
        args: Command-line arguments
        num_params: Number of model parameters
        train_time_sec: Training time in seconds
        eval_time_sec: Evaluation time on test in seconds
        output_path: Path to save text file
    """
    cm = np.array(test_metrics["confusion_matrix"])
    
    report = f"""Transformer Model - Classification Report
{'='*70}

MODEL: {args.model_name}
Generated: {datetime.now().isoformat()}Z

TRAINING CONFIGURATION
{'-'*70}
Max sequence length:   {args.max_seq_length}
Train batch size:      {args.train_batch_size}
Eval batch size:       {args.eval_batch_size}
Epochs:                {args.num_train_epochs}
Learning rate:         {args.learning_rate}
Weight decay:          {args.weight_decay}
Warmup ratio:          {args.warmup_ratio}
Random seed:           {args.seed}
Trainable parameters:  {num_params:,}

VALIDATION SET METRICS
{'-'*70}
Accuracy:  {val_metrics['accuracy']:.4f}
Macro-F1:  {val_metrics['macro_f1']:.4f}

F1 per class:
  T1 (Human Original):       {val_metrics['f1_per_class']['T1']:.4f}
  T2 (LLM Generated):        {val_metrics['f1_per_class']['T2']:.4f}
  T3 (Human Paraphrased):    {val_metrics['f1_per_class']['T3']:.4f}
  T4 (LLM Paraphrased):      {val_metrics['f1_per_class']['T4']:.4f}

TEST SET METRICS (FINAL)
{'-'*70}
Accuracy:  {test_metrics['accuracy']:.4f}
Macro-F1:  {test_metrics['macro_f1']:.4f}

F1 per class:
  T1 (Human Original):       {test_metrics['f1_per_class']['T1']:.4f}
  T2 (LLM Generated):        {test_metrics['f1_per_class']['T2']:.4f}
  T3 (Human Paraphrased):    {test_metrics['f1_per_class']['T3']:.4f}
  T4 (LLM Paraphrased):      {test_metrics['f1_per_class']['T4']:.4f}

CONFUSION MATRIX (TEST)
{'-'*70}
          Predicted
          T1    T2    T3    T4
Actual
  T1    {cm[0,0]:>4}  {cm[0,1]:>4}  {cm[0,2]:>4}  {cm[0,3]:>4}
  T2    {cm[1,0]:>4}  {cm[1,1]:>4}  {cm[1,2]:>4}  {cm[1,3]:>4}
  T3    {cm[2,0]:>4}  {cm[2,1]:>4}  {cm[2,2]:>4}  {cm[2,3]:>4}
  T4    {cm[3,0]:>4}  {cm[3,1]:>4}  {cm[3,2]:>4}  {cm[3,3]:>4}

EFFICIENCY METRICS
{'-'*70}
Training time:     {train_time_sec:.2f} seconds ({train_time_sec/60:.2f} minutes)
Test eval time:    {eval_time_sec:.2f} seconds
Throughput:        {len(test_metrics['confusion_matrix'][0])*4/eval_time_sec:.1f} samples/sec

{'='*70}
"""
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"✓ Classification report saved to: {output_path}")


def save_training_args(
    training_args: TrainingArguments,
    output_path: Path
):
    """
    Save training arguments to JSON file.
    
    Args:
        training_args: Trainer's training arguments
        output_path: Path to save JSON file
    """
    args_dict = training_args.to_dict()
    
    with open(output_path, 'w') as f:
        json.dump(args_dict, f, indent=2)
    
    print(f"✓ Training args saved to: {output_path}")


# ============================================================
# MAIN PIPELINE
# ============================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train transformer models for 4-class text classification"
    )
    
    # Required arguments
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help='HuggingFace model name (e.g., distilbert-base-uncased, bert-base-uncased, roberta-base)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--config',
        type=str,
        default='configs/data_config.yaml',
        help='Path to data config file (default: configs/data_config.yaml)'
    )
    parser.add_argument(
        '--max_seq_length',
        type=int,
        default=256,
        help='Maximum sequence length for tokenization (default: 256)'
    )
    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=16,
        help='Training batch size per device (default: 16)'
    )
    parser.add_argument(
        '--eval_batch_size',
        type=int,
        default=32,
        help='Evaluation batch size per device (default: 32)'
    )
    parser.add_argument(
        '--num_train_epochs',
        type=float,
        default=3.0,
        help='Number of training epochs (default: 3.0)'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=2e-5,
        help='Learning rate (default: 2e-5)'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.01,
        help='Weight decay (default: 0.01)'
    )
    parser.add_argument(
        '--warmup_ratio',
        type=float,
        default=0.1,
        help='Warmup ratio (default: 0.1)'
    )
    parser.add_argument(
        '--max_grad_norm',
        type=float,
        default=1.0,
        help='Max gradient norm for clipping (default: 1.0)'
    )
    parser.add_argument(
        '--logging_steps',
        type=int,
        default=100,
        help='Log every N steps (default: 100)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--output_root',
        type=str,
        default='results/transformer',
        help='Root directory for outputs (default: results/transformer)'
    )
    
    return parser.parse_args()


def main():
    """
    Main training pipeline for transformer models.
    """
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    print("="*70)
    print("NLP Multi-Type Classification: Transformer Model Training")
    print(f"Model: {args.model_name}")
    print("="*70)
    
    # Create model-specific output directory
    model_slug = args.model_name.replace('/', '-')
    output_dir = Path(args.output_root) / model_slug
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    # Step 1: Load configuration and data
    print("\n[1/7] Loading configuration and data")
    config = load_config(args.config)
    processed_dir = config.get('processed_dir', 'data/processed')
    
    train_df, val_df, test_df = load_splits(processed_dir)
    
    # Step 2: Build tokenizer and model
    print("\n[2/7] Building tokenizer and model")
    tokenizer, model = build_tokenizer_and_model(args.model_name, num_labels=4)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Using device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Step 3: Tokenize datasets
    print("\n[3/7] Tokenizing datasets")
    print(f"  Max sequence length: {args.max_seq_length}")
    
    train_dataset = tokenize_dataset(train_df, tokenizer, args.max_seq_length)
    val_dataset = tokenize_dataset(val_df, tokenizer, args.max_seq_length)
    test_dataset = tokenize_dataset(test_df, tokenizer, args.max_seq_length)
    
    print(f"  ✓ Train dataset: {len(train_dataset)} samples")
    print(f"  ✓ Val dataset:   {len(val_dataset)} samples")
    print(f"  ✓ Test dataset:  {len(test_dataset)} samples")
    
    # Step 4: Train and evaluate
    print("\n[4/7] Training and evaluation")
    test_metrics, val_metrics, train_time_sec, eval_time_sec, num_params, model_size_mb = train_and_evaluate(
        model, tokenizer, train_dataset, val_dataset, test_dataset, args, output_dir
    )
    
    # Step 5: Save model and tokenizer
    print("\n[5/7] Saving model and tokenizer")
    save_model_and_tokenizer(model, tokenizer, output_dir)
    
    # Step 6: Save metrics JSON
    print("\n[6/7] Saving metrics and reports")
    metrics_path = output_dir / "metrics.json"
    save_metrics_json(
        test_metrics, val_metrics, args,
        train_samples=len(train_df),
        val_samples=len(val_df),
        test_samples=len(test_df),
        num_params=num_params,
        train_time_sec=train_time_sec,
        eval_time_sec=eval_time_sec,
        model_size_mb=model_size_mb,
        output_path=metrics_path
    )
    
    # Step 7: Save visualizations and reports
    print("\n[7/7] Generating visualizations and reports")
    
    # Confusion matrix plot
    cm_plot_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(
        np.array(test_metrics["confusion_matrix"]),
        args.model_name,
        cm_plot_path
    )
    
    # Classification report
    report_path = output_dir / "report.txt"
    save_classification_report(
        test_metrics, val_metrics, args, num_params, train_time_sec, eval_time_sec, report_path
    )
    
    # Print final summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Model:               {args.model_name}")
    print(f"Model slug:          {model_slug}")
    print(f"Device:              {device}")
    print(f"Parameters:          {num_params:,}")
    print(f"\nTest Set Performance:")
    print(f"  Accuracy:          {test_metrics['accuracy']:.4f}")
    print(f"  Macro-F1:          {test_metrics['macro_f1']:.4f}")
    print(f"  F1 per class:")
    for label, f1_val in test_metrics['f1_per_class'].items():
        print(f"    {label}:             {f1_val:.4f}")
    print(f"\nEfficiency:")
    print(f"  Training time:     {train_time_sec:.2f} sec ({train_time_sec/60:.2f} min)")
    print(f"  Test eval time:    {eval_time_sec:.2f} sec")
    print(f"  Throughput:        {len(test_df)/eval_time_sec:.1f} samples/sec")
    print(f"  Model size:        {model_size_mb:.3f} MB")
    print(f"\nResults saved to: {output_dir}")
    print(f"  - metrics.json")
    print(f"  - confusion_matrix.png")
    print(f"  - report.txt")
    print(f"  - model/ (full HF model + tokenizer)")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

