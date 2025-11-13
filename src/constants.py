"""
Central constants for the NLP multi-type classification project.

This module defines canonical label mappings, column names, split ratios,
and other project-wide constants to ensure consistency across all components.
"""

# ============================================================
# LABEL DEFINITIONS
# ============================================================
# Four-class single-sentence classification task:
# - T1: Human Original
# - T2: LLM Generated
# - T3: Human Paraphrased
# - T4: LLM Paraphrased

LABELS = ["T1", "T2", "T3", "T4"]

# Canonical label mapping (string → integer)
LABEL2ID = {
    "T1": 0,
    "T2": 1,
    "T3": 2,
    "T4": 3,
}

# Reverse mapping (integer → string)
ID2LABEL = {
    0: "T1",
    1: "T2",
    2: "T3",
    3: "T4",
}

# ============================================================
# RAW DATA FIELD NAMES
# ============================================================
# Field names in raw JSON data
RAW_FIELD_TYPE1 = "human_original_text(type1)"
RAW_FIELD_TYPE2 = "llm_generated_text(type2)"
RAW_FIELD_TYPE3 = "human_paraphrased_text(type3)"
RAW_FIELD_TYPE4 = "llm_paraphrased_original_text(type4)-prompt-based"

# Mapping from raw field names to labels
RAW_FIELD_TO_LABEL = {
    RAW_FIELD_TYPE1: "T1",
    RAW_FIELD_TYPE2: "T2",
    RAW_FIELD_TYPE3: "T3",
    RAW_FIELD_TYPE4: "T4",
}

# Valid data sources
VALID_SOURCES = ["mrpc", "paws", "hlpc"]

# ============================================================
# DATA SCHEMA (JSONL)
# ============================================================
# Required fields in processed JSONL (train/val/test)
REQUIRED_FIELDS = ["id", "family_id", "source", "text", "label", "label_id", "text_len_char", "text_len_word"]

# All field names for reference
FIELD_NAMES = REQUIRED_FIELDS

# ============================================================
# SPLIT CONFIGURATION
# ============================================================
# Default seed for reproducibility across all random operations
DEFAULT_SEED = 42

# Split ratios (must sum to 1.0)
DEFAULT_SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}

# Group key for family-aware splitting (prevents leakage)
GROUP_KEY = "family_id"

# ============================================================
# DATA VALIDATION THRESHOLDS
# ============================================================
# Maximum allowed text length in characters (guardrail for outliers)
MAX_TEXT_LENGTH_CHARS = 4000

# Minimum class proportion threshold for imbalance warning
MIN_CLASS_PROPORTION = 0.10

# ============================================================
# FILE NAMING CONVENTIONS
# ============================================================
TRAIN_FILENAME = "train_4class.jsonl"
VAL_FILENAME = "val_4class.jsonl"
TEST_FILENAME = "test_4class.jsonl"
MANIFEST_FILENAME = "manifest.json"

# ============================================================
# EVALUATION METRICS
# ============================================================
# Primary metric for model selection
PRIMARY_METRIC = "macro_f1"

# All metrics to compute and report
EVALUATION_METRICS = [
    "accuracy",
    "macro_f1",
    "per_class_precision",
    "per_class_recall",
    "per_class_f1",
    "auroc_ovr",  # One-vs-rest AUROC
    "confusion_matrix",
]

# Runtime/efficiency metrics
EFFICIENCY_METRICS = [
    "train_time_sec",
    "inference_latency_ms_per_sample",
    "num_parameters",
    "peak_vram_mb",  # Optional, if available
]

