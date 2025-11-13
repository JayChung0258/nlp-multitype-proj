"""
Data schemas and validation helpers for the NLP multi-type classification project.

This module defines the structure of raw and processed data, config files,
and evaluation results. It provides a single source of truth for all data
contracts in the project.
"""

from typing import Optional, Dict, List, Any
from dataclasses import dataclass


# ============================================================
# RAW DATA SCHEMA
# ============================================================
@dataclass
class RawFamilyRow:
    """
    Schema for raw input data organized by family.
    
    Each family represents variants of the same base content across
    different generation/paraphrase types.
    
    Attributes:
        family_id: Unique identifier for grouping related samples
        type1: Human original text (may be None if not available)
        type2: LLM generated text (may be None if not available)
        type3: Human paraphrased text (may be None if not available)
        type4: LLM paraphrased text (may be None if not available)
        metadata: Optional additional fields (source, domain, etc.)
    """
    family_id: str
    type1: Optional[str] = None
    type2: Optional[str] = None
    type3: Optional[str] = None
    type4: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


# ============================================================
# PROCESSED DATA SCHEMA
# ============================================================
@dataclass
class ProcessedRow:
    """
    Schema for processed single-sentence classification samples.
    
    This is the format used in train_4class.csv, val_4class.csv, and test_4class.csv.
    
    Required fields:
        family_id: Links back to original family (for leakage checks)
        text: The sentence to classify (non-empty, stripped)
        label: One of {T1, T2, T3, T4} or integer {0, 1, 2, 3}
    
    Optional fields:
        text_len_char: Character count (for analysis)
        text_len_word: Word count (for analysis)
        split: Split assignment (train/val/test), useful for merged manifests
    """
    family_id: str
    text: str
    label: str  # or int, depending on implementation choice
    text_len_char: Optional[int] = None
    text_len_word: Optional[int] = None
    split: Optional[str] = None


# ============================================================
# CONFIG SCHEMAS
# ============================================================
@dataclass
class DataConfig:
    """
    Schema for data_config.yaml.
    
    Defines paths, label mappings, split parameters, and preprocessing rules.
    """
    raw_dir: str
    processed_dir: str
    train_file: str
    val_file: str
    test_file: str
    label_map: Dict[str, int]
    split_config: Dict[str, Any]  # Contains train_ratio, val_ratio, test_ratio, seed, group_key
    preprocessing: Dict[str, Any]  # Contains strip_whitespace, normalize_unicode, etc.


@dataclass
class BaselineModelConfig:
    """
    Schema for models_baseline.yaml.
    
    Defines TF-IDF parameters and classical model hyperparameters.
    """
    tfidf_params: Dict[str, Any]
    models: List[Dict[str, Any]]  # List of model definitions with name and hyperparams


@dataclass
class TransformerModelConfig:
    """
    Schema for models_transformer.yaml.
    
    Defines common training parameters and list of transformer models to benchmark.
    """
    common: Dict[str, Any]  # Shared training hyperparameters
    models: List[Dict[str, str]]  # List with 'name' keys pointing to HF model IDs


@dataclass
class ProjectConfig:
    """
    Schema for project.yaml.
    
    Project-level settings for output directories, reporting options, and runtime.
    """
    output_dirs: Dict[str, str]
    reporting: Dict[str, bool]
    runtime: Dict[str, Any]


# ============================================================
# EVALUATION RESULT SCHEMAS
# ============================================================
@dataclass
class ModelRunResult:
    """
    Schema for per-run evaluation results (saved as JSON).
    
    This structure will be saved after each model training/evaluation run.
    """
    model_name: str
    accuracy: float
    macro_f1: float
    per_class_f1: Dict[str, float]  # {"T1": 0.85, "T2": 0.90, ...}
    train_time_sec: float
    inference_latency_ms_per_sample: float
    num_parameters: int
    seed: int
    timestamp_utc: str
    
    # Optional additional metrics
    per_class_precision: Optional[Dict[str, float]] = None
    per_class_recall: Optional[Dict[str, float]] = None
    auroc_ovr: Optional[float] = None
    confusion_matrix: Optional[List[List[int]]] = None


# ============================================================
# VALIDATION RULES (TO BE IMPLEMENTED)
# ============================================================

def validate_processed_row(row: ProcessedRow) -> bool:
    """
    Validate a processed row against schema requirements.
    
    Rules:
    - family_id must be non-empty string
    - text must be non-empty and stripped
    - label must be in {T1, T2, T3, T4} or {0, 1, 2, 3}
    - text length must not exceed MAX_TEXT_LENGTH_CHARS
    
    TODO: Implement validation logic
    
    Args:
        row: ProcessedRow instance to validate
        
    Returns:
        True if valid, False otherwise
        
    Raises:
        ValueError: If critical validation fails
    """
    # TODO: Implement
    pass


def validate_split_integrity(train_df, val_df, test_df) -> Dict[str, Any]:
    """
    Validate family-aware split integrity across splits.
    
    Checks:
    - No family_id appears in multiple splits
    - All labels are valid
    - Class distributions are reasonable (warn if imbalanced)
    - No exact duplicate texts within splits
    
    TODO: Implement validation logic
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        
    Returns:
        Dictionary with validation results and any warnings
    """
    # TODO: Implement
    pass


def validate_config(config: Dict[str, Any], config_type: str) -> bool:
    """
    Validate a configuration dictionary against expected schema.
    
    TODO: Implement schema validation for each config type
    
    Args:
        config: Configuration dictionary loaded from YAML
        config_type: One of {"data", "baseline", "transformer", "project"}
        
    Returns:
        True if valid, False otherwise
    """
    # TODO: Implement
    pass

