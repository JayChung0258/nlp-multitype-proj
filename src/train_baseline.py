"""
Baseline model training script for NLP multi-type classification.

This script trains a TF-IDF + Logistic Regression baseline model for
4-class single-sentence classification (T1/T2/T3/T4).

Usage:
    python -m src.train_baseline
    or
    python src/train_baseline.py
"""

import json
import time
import os
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any

import pandas as pd
import numpy as np
import yaml
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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

def load_splits_from_config(config_path: str = "configs/data_config.yaml") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train/val/test splits from processed JSONL files.
    
    Args:
        config_path: Path to data configuration YAML
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    print("Loading data configuration...")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    processed_dir = config.get('processed_dir', 'data/processed')
    
    # Construct paths
    train_path = Path(processed_dir) / "train_4class.jsonl"
    val_path = Path(processed_dir) / "val_4class.jsonl"
    test_path = Path(processed_dir) / "test_4class.jsonl"
    
    print(f"  Processed directory: {processed_dir}")
    
    # Load JSONL files
    print(f"Loading datasets...")
    train_df = pd.read_json(train_path, lines=True)
    val_df = pd.read_json(val_path, lines=True)
    test_df = pd.read_json(test_path, lines=True)
    
    print(f"  Train: {len(train_df):,} samples")
    print(f"  Val:   {len(val_df):,} samples")
    print(f"  Test:  {len(test_df):,} samples")
    
    return train_df, val_df, test_df


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def build_tfidf_vectorizer(
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 20000,
    lowercase: bool = True
) -> TfidfVectorizer:
    """
    Create TF-IDF vectorizer with specified parameters.
    
    Args:
        ngram_range: Range of n-grams to extract
        max_features: Maximum number of features
        lowercase: Convert text to lowercase
        
    Returns:
        Configured TfidfVectorizer
    """
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        lowercase=lowercase,
        strip_accents='unicode',
        stop_words=None,  # Keep all words for this task
        sublinear_tf=True,  # Use sublinear term frequency scaling
        min_df=2,  # Minimum document frequency
        max_df=0.95  # Maximum document frequency (filter very common terms)
    )
    
    print(f"\nTF-IDF Configuration:")
    print(f"  N-gram range: {ngram_range}")
    print(f"  Max features: {max_features:,}")
    print(f"  Lowercase: {lowercase}")
    
    return vectorizer


def extract_features(
    train_texts: pd.Series,
    val_texts: pd.Series,
    test_texts: pd.Series,
    vectorizer: TfidfVectorizer
) -> Tuple[Any, Any, Any, int]:
    """
    Extract TF-IDF features from text data.
    
    Args:
        train_texts: Training text data
        val_texts: Validation text data
        test_texts: Test text data
        vectorizer: TfidfVectorizer instance
        
    Returns:
        Tuple of (X_train, X_val, X_test, vocab_size)
    """
    print("\nExtracting TF-IDF features...")
    
    # Fit on training data and transform all splits
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)
    
    vocab_size = len(vectorizer.vocabulary_)
    
    print(f"  Vocabulary size: {vocab_size:,} features")
    print(f"  Train matrix shape: {X_train.shape}")
    print(f"  Val matrix shape: {X_val.shape}")
    print(f"  Test matrix shape: {X_test.shape}")
    
    return X_train, X_val, X_test, vocab_size


# ============================================================
# MODEL TRAINING
# ============================================================

def train_logreg_model(
    X_train: Any,
    y_train: np.ndarray,
    multi_class: str = "multinomial",
    solver: str = "lbfgs",
    max_iter: int = 2000,
    class_weight: str = "balanced",
    random_state: int = 42
) -> Tuple[LogisticRegression, float]:
    """
    Train Logistic Regression model.
    
    Args:
        X_train: Training features (TF-IDF matrix)
        y_train: Training labels (numeric)
        multi_class: Multi-class strategy
        solver: Optimization solver
        max_iter: Maximum iterations
        class_weight: Class weighting strategy
        random_state: Random seed
        
    Returns:
        Tuple of (trained model, training time in seconds)
    """
    print("\nTraining Logistic Regression model...")
    print(f"  Multi-class: {multi_class}")
    print(f"  Solver: {solver}")
    print(f"  Max iterations: {max_iter}")
    print(f"  Class weight: {class_weight}")
    
    # Initialize model
    clf = LogisticRegression(
        multi_class=multi_class,
        solver=solver,
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=random_state,
        verbose=0
    )
    
    # Train and measure time
    start_time = time.perf_counter()
    clf.fit(X_train, y_train)
    end_time = time.perf_counter()
    
    train_time_sec = end_time - start_time
    
    print(f"  Training completed in {train_time_sec:.2f} seconds")
    print(f"  Converged: {clf.n_iter_[0] < max_iter}")
    print(f"  Iterations: {clf.n_iter_[0]}")
    
    return clf, train_time_sec


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model: LogisticRegression,
    X_test: Any,
    y_test: np.ndarray
) -> Dict[str, Any]:
    """
    Evaluate model on test set and compute all metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels (numeric)
        
    Returns:
        Dictionary with all metrics
    """
    print("\nEvaluating on test set...")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    f1_per_class = f1_score(y_test, y_pred, average=None)  # Returns array [f1_0, f1_1, f1_2, f1_3]
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])
    
    # Convert per-class F1 to dictionary
    f1_dict = {
        "T1": float(f1_per_class[0]),
        "T2": float(f1_per_class[1]),
        "T3": float(f1_per_class[2]),
        "T4": float(f1_per_class[3])
    }
    
    # Print summary
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Macro-F1:  {macro_f1:.4f}")
    print(f"  F1 per class:")
    for label, f1_score_val in f1_dict.items():
        print(f"    {label}: {f1_score_val:.4f}")
    
    metrics = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "f1_per_class": f1_dict,
        "confusion_matrix": cm.tolist()
    }
    
    return metrics


# ============================================================
# SAVING ARTIFACTS
# ============================================================

def save_model(model: LogisticRegression, vectorizer: TfidfVectorizer, output_dir: Path) -> float:
    """
    Save trained model and vectorizer, compute model size.
    
    Args:
        model: Trained Logistic Regression model
        vectorizer: Fitted TF-IDF vectorizer
        output_dir: Directory to save model
        
    Returns:
        Model size in MB
    """
    print("\nSaving model artifacts...")
    
    model_path = output_dir / "logreg_model.joblib"
    vectorizer_path = output_dir / "logreg_vectorizer.joblib"
    
    # Save model and vectorizer
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    # Compute model size (model + vectorizer)
    model_size_bytes = os.path.getsize(model_path) + os.path.getsize(vectorizer_path)
    model_size_mb = model_size_bytes / (1024 * 1024)
    
    print(f"  Model saved to: {model_path}")
    print(f"  Vectorizer saved to: {vectorizer_path}")
    print(f"  Model size: {model_size_mb:.3f} MB")
    
    return round(model_size_mb, 3)


def save_metrics_json(
    metrics: Dict[str, Any],
    train_samples: int,
    test_samples: int,
    vocab_size: int,
    train_time_sec: float,
    model_size_mb: float,
    output_path: Path
):
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Dictionary with evaluation metrics
        train_samples: Number of training samples
        test_samples: Number of test samples
        vocab_size: TF-IDF vocabulary size
        train_time_sec: Training time in seconds
        model_size_mb: Model size in MB
        output_path: Path to save JSON file
    """
    output = {
        "model_name": "tfidf_logreg",
        "timestamp_utc": datetime.now().isoformat() + "Z",
        "train_samples": train_samples,
        "test_samples": test_samples,
        "vocab_size": vocab_size,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "f1_per_class": metrics["f1_per_class"],
        "confusion_matrix": metrics["confusion_matrix"],
        "train_time_sec": float(train_time_sec),
        "model_size_mb": model_size_mb
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Metrics saved to: {output_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list,
    output_path: Path
):
    """
    Plot confusion matrix as heatmap and save to file.
    
    Args:
        cm: Confusion matrix (4x4 array)
        labels: Label names (["T1", "T2", "T3", "T4"])
        output_path: Path to save PNG file
    """
    plt.figure(figsize=(8, 7))
    
    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'}
    )
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix: TF-IDF + Logistic Regression', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Confusion matrix plot saved to: {output_path}")


def save_report_txt(
    metrics: Dict[str, Any],
    train_time_sec: float,
    model_size_mb: float,
    output_path: Path
):
    """
    Save human-readable metrics report to text file.
    
    Args:
        metrics: Dictionary with evaluation metrics
        train_time_sec: Training time in seconds
        model_size_mb: Model size in MB
        output_path: Path to save text file
    """
    cm = np.array(metrics["confusion_matrix"])
    
    report = f"""TF-IDF + Logistic Regression Baseline - Evaluation Report
{'='*70}

Generated: {datetime.now().isoformat()}Z

OVERALL METRICS
{'-'*70}
Accuracy:  {metrics['accuracy']:.4f}
Macro-F1:  {metrics['macro_f1']:.4f}

PER-CLASS F1 SCORES
{'-'*70}
T1 (Human Original):       {metrics['f1_per_class']['T1']:.4f}
T2 (LLM Generated):        {metrics['f1_per_class']['T2']:.4f}
T3 (Human Paraphrased):    {metrics['f1_per_class']['T3']:.4f}
T4 (LLM Paraphrased):      {metrics['f1_per_class']['T4']:.4f}

CONFUSION MATRIX
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
Training time: {train_time_sec:.2f} seconds
Model size:    {model_size_mb:.3f} MB

{'='*70}
"""
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"✓ Text report saved to: {output_path}")


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    """
    Main training pipeline for baseline model.
    """
    print("="*70)
    print("NLP Multi-Type Classification: Baseline Model Training")
    print("TF-IDF + Logistic Regression")
    print("="*70)
    
    # Create output directory
    output_dir = Path("results/baseline")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load data
    print("\n[1/6] Loading data")
    train_df, val_df, test_df = load_splits_from_config()
    
    # Extract text and labels
    X_train_text = train_df['text']
    y_train = train_df['label_id'].values
    
    X_val_text = val_df['text']
    y_val = val_df['label_id'].values
    
    X_test_text = test_df['text']
    y_test = test_df['label_id'].values
    
    print(f"\n  Label distribution (train):")
    for label_id in sorted(LABEL2ID.values()):
        count = (y_train == label_id).sum()
        pct = count / len(y_train) * 100
        print(f"    {ID2LABEL[label_id]}: {count:,} ({pct:.1f}%)")
    
    # Step 2: Build TF-IDF features
    print("\n[2/6] Building TF-IDF features")
    vectorizer = build_tfidf_vectorizer(
        ngram_range=(1, 2),
        max_features=20000,
        lowercase=True
    )
    
    X_train, X_val, X_test, vocab_size = extract_features(
        X_train_text, X_val_text, X_test_text, vectorizer
    )
    
    # Step 3: Train model
    print("\n[3/6] Training Logistic Regression model")
    model, train_time_sec = train_logreg_model(
        X_train,
        y_train,
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    )
    
    # Step 4: Evaluate on test set
    print("\n[4/6] Evaluating on test set")
    metrics = evaluate_model(model, X_test, y_test)
    
    # Step 5: Save model and compute size
    print("\n[5/6] Saving model artifacts")
    model_size_mb = save_model(model, vectorizer, output_dir)
    
    # Step 6: Save metrics and visualizations
    print("\n[6/6] Saving metrics and visualizations")
    
    # Save metrics JSON
    metrics_path = output_dir / "logreg_metrics.json"
    save_metrics_json(
        metrics,
        train_samples=len(train_df),
        test_samples=len(test_df),
        vocab_size=vocab_size,
        train_time_sec=train_time_sec,
        model_size_mb=model_size_mb,
        output_path=metrics_path
    )
    
    # Save confusion matrix plot
    cm_plot_path = output_dir / "logreg_confusion_matrix.png"
    plot_confusion_matrix(
        np.array(metrics["confusion_matrix"]),
        labels=LABELS,
        output_path=cm_plot_path
    )
    
    # Save text report
    report_path = output_dir / "logreg_report.txt"
    save_report_txt(metrics, train_time_sec, model_size_mb, report_path)
    
    # Print summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Model:      TF-IDF + Logistic Regression")
    print(f"Accuracy:   {metrics['accuracy']:.4f}")
    print(f"Macro-F1:   {metrics['macro_f1']:.4f}")
    print(f"Train time: {train_time_sec:.2f} seconds")
    print(f"Model size: {model_size_mb:.3f} MB")
    print(f"\nAll results saved to: {output_dir}")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

