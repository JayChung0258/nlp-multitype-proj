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
class RawFamily:
    """
    Schema for raw input data organized by family.
    
    Each family represents variants of the same base content across
    different generation/paraphrase types.
    
    Attributes:
        family_id: Unique identifier constructed as "<source>_<idx>"
        source: Data source (mrpc, paws, or hlpc)
        type1: Human original text (may be None if not available)
        type2: LLM generated text (may be None if not available)
        type3: Human paraphrased text (may be None if not available)
        type4: LLM paraphrased text (may be None if not available)
    """
    family_id: str
    source: str
    type1: Optional[str] = None
    type2: Optional[str] = None
    type3: Optional[str] = None
    type4: Optional[str] = None


# ============================================================
# PROCESSED DATA SCHEMA
# ============================================================
@dataclass
class ProcessedRow:
    """
    Schema for processed single-sentence classification samples.
    
    This is the format used in train_4class.jsonl, val_4class.jsonl, and test_4class.jsonl.
    
    All fields are required:
        id: Globally unique row identifier "<family_id>__<label>"
        family_id: Links back to original family (for leakage checks)
        source: Data source (mrpc, paws, or hlpc)
        text: The sentence to classify (non-empty, normalized)
        label: One of {T1, T2, T3, T4}
        label_id: Integer mapping (0, 1, 2, 3)
        text_len_char: Character count
        text_len_word: Word count (whitespace split)
    """
    id: str
    family_id: str
    source: str
    text: str
    label: str
    label_id: int
    text_len_char: int
    text_len_word: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "family_id": self.family_id,
            "source": self.source,
            "text": self.text,
            "label": self.label,
            "label_id": self.label_id,
            "text_len_char": self.text_len_char,
            "text_len_word": self.text_len_word,
        }


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

