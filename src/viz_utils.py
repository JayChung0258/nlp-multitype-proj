"""
Visualization utilities for the NLP multi-type classification project.

This module provides functions to create plots for EDA, results analysis,
and model comparison.
"""

from typing import Dict, List, Any, Optional
import numpy as np

# TODO: Add imports when implementing
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd


# ============================================================
# DATA EXPLORATION VISUALIZATIONS
# ============================================================

def plot_class_distribution(
    labels: List[str],
    counts: List[int],
    title: str = "Class Distribution",
    output_path: Optional[str] = None
):
    """
    Create bar chart of class proportions.
    
    Args:
        labels: Class labels (T1, T2, T3, T4)
        counts: Sample counts per class
        title: Plot title
        output_path: Path to save figure (if None, display only)
        
    TODO: Implement using matplotlib/seaborn
    """
    # TODO: Implement
    # Use seaborn barplot or matplotlib bar
    # Show percentages on bars
    # Save if output_path provided
    pass


def plot_text_length_distribution(
    df: Any,
    length_col: str = "text_len_word",
    group_col: str = "label",
    output_path: Optional[str] = None
):
    """
    Create boxplots of text length distribution per class.
    
    Args:
        df: DataFrame with processed samples
        length_col: Column name for length metric (text_len_word or text_len_char)
        group_col: Column to group by (typically 'label')
        output_path: Path to save figure
        
    TODO: Implement
    """
    # TODO: Implement
    # Use seaborn boxplot or violin plot
    # Log scale may be useful if distribution is skewed
    pass


def plot_length_histogram(
    lengths: np.ndarray,
    bins: int = 50,
    title: str = "Text Length Distribution",
    output_path: Optional[str] = None
):
    """
    Create histogram of text lengths.
    
    Args:
        lengths: Array of text lengths
        bins: Number of histogram bins
        title: Plot title
        output_path: Path to save figure
        
    TODO: Implement
    """
    # TODO: Implement
    pass


def plot_pairwise_class_comparison(
    df: Any,
    class1: str,
    class2: str,
    feature: str,
    output_path: Optional[str] = None
):
    """
    Create scatter or density plot comparing two classes on a feature.
    
    Useful for investigating T3 vs T4 confusions.
    
    Args:
        df: DataFrame with samples and features
        class1: First class label (e.g., "T3")
        class2: Second class label (e.g., "T4")
        feature: Feature to plot (e.g., text_len_word)
        output_path: Path to save figure
        
    TODO: Implement
    """
    # TODO: Implement
    pass


# ============================================================
# RESULTS VISUALIZATIONS
# ============================================================

def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    title: str = "Confusion Matrix",
    output_path: Optional[str] = None
):
    """
    Create heatmap of confusion matrix.
    
    Args:
        cm: Confusion matrix (4x4)
        labels: Class labels (T1, T2, T3, T4)
        title: Plot title
        output_path: Path to save figure
        
    TODO: Implement using seaborn heatmap
    """
    # TODO: Implement
    # Use seaborn heatmap with annotations
    # Add row/column labels
    # Use appropriate color scheme (e.g., Blues or viridis)
    pass


def plot_per_class_metrics(
    metrics_df: Any,
    metric: str = "f1",
    output_path: Optional[str] = None
):
    """
    Create bar chart comparing per-class metrics across models.
    
    Args:
        metrics_df: DataFrame with model results
        metric: Metric to plot (precision, recall, f1)
        output_path: Path to save figure
        
    TODO: Implement
    """
    # TODO: Implement
    # Group by model_name and class
    # Use grouped bar chart
    pass


def plot_model_comparison(
    results_df: Any,
    x_metric: str = "latency_ms",
    y_metric: str = "macro_f1",
    output_path: Optional[str] = None
):
    """
    Create scatter plot comparing models on two metrics.
    
    Useful for latency-F1 tradeoff analysis.
    
    Args:
        results_df: DataFrame with aggregated results
        x_metric: Metric for x-axis (e.g., latency_ms)
        y_metric: Metric for y-axis (e.g., macro_f1)
        output_path: Path to save figure
        
    TODO: Implement
    """
    # TODO: Implement
    # Use scatter plot with model names as labels
    # Add quadrant lines or Pareto frontier if applicable
    pass


def plot_training_curves(
    history: Dict[str, List[float]],
    output_path: Optional[str] = None
):
    """
    Plot training and validation curves over epochs.
    
    For transformer models only (if using Trainer with logging).
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'val_f1', etc.
        output_path: Path to save figure
        
    TODO: Implement
    """
    # TODO: Implement
    # Plot loss and metric curves
    # Use dual y-axes if needed
    pass


# ============================================================
# ROBUSTNESS ANALYSIS VISUALIZATIONS
# ============================================================

def plot_perturbation_results(
    baseline_scores: Dict[str, float],
    perturbed_scores: Dict[str, Dict[str, float]],
    output_path: Optional[str] = None
):
    """
    Visualize performance under perturbations.
    
    Args:
        baseline_scores: Original scores (model_name → metric)
        perturbed_scores: Scores after perturbations 
                         (perturbation_type → model_name → metric)
        output_path: Path to save figure
        
    TODO: Implement
    """
    # TODO: Implement
    # Bar chart showing performance drop for each perturbation
    pass


def plot_length_stratified_performance(
    df: Any,
    length_bins: List[str],
    metric: str = "accuracy",
    output_path: Optional[str] = None
):
    """
    Plot performance across text length buckets.
    
    Args:
        df: DataFrame with predictions and length information
        length_bins: Bin labels (e.g., ["short", "medium", "long"])
        metric: Metric to plot
        output_path: Path to save figure
        
    TODO: Implement
    """
    # TODO: Implement
    # Line plot or bar chart showing metric per length bin
    pass


# ============================================================
# ERROR ANALYSIS VISUALIZATIONS
# ============================================================

def plot_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Optional[str] = None
):
    """
    Visualize which classes are most often confused.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        output_path: Path to save figure
        
    TODO: Implement
    """
    # TODO: Implement
    # Can use confusion matrix or Sankey diagram
    pass


# ============================================================
# STYLE CONFIGURATION
# ============================================================

def set_plot_style():
    """
    Set consistent matplotlib/seaborn style for all plots.
    
    TODO: Implement style configuration
    """
    # TODO: Implement
    # Set seaborn style (e.g., 'whitegrid', 'darkgrid')
    # Set color palette
    # Set figure size defaults
    # Set font sizes
    pass


# Default figure size for different plot types
FIGURE_SIZES = {
    "bar": (10, 6),
    "box": (12, 6),
    "heatmap": (8, 8),
    "scatter": (10, 8),
    "line": (12, 6),
}

