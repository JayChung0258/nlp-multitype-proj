"""
Data preparation script for NLP multi-type classification project.

This script:
1. Reads raw JSON/JSONL files from data/raw/
2. Normalizes text and materializes rows for each type
3. Performs family-aware train/val/test split (no leakage)
4. Writes processed JSONL files to data/processed/
5. Generates manifest with statistics

Usage:
    python -m src.data_prep
"""

import json
import hashlib
import unicodedata
import re
from pathlib import Path
from typing import List, Dict, Tuple, Any, Set
from collections import defaultdict, Counter
from datetime import datetime
import sys

try:
    import yaml
    import numpy as np
except ImportError as e:
    print(f"Error: Missing required package: {e}")
    print("Please install requirements: pip install pyyaml numpy")
    sys.exit(1)

from .constants import (
    LABELS,
    LABEL2ID,
    VALID_SOURCES,
    RAW_FIELD_TYPE1,
    RAW_FIELD_TYPE2,
    RAW_FIELD_TYPE3,
    RAW_FIELD_TYPE4,
    RAW_FIELD_TO_LABEL,
    DEFAULT_SEED,
    DEFAULT_SPLIT_RATIOS,
    MAX_TEXT_LENGTH_CHARS,
    TRAIN_FILENAME,
    VAL_FILENAME,
    TEST_FILENAME,
    MANIFEST_FILENAME,
)
from .schema import RawFamily, ProcessedRow


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize text according to specification.
    
    Steps:
    1. Unicode NFKC normalization
    2. Strip leading/trailing whitespace
    3. Replace runs of whitespace with single space
    4. Remove control characters except standard whitespace
    
    Args:
        text: Input text string
        
    Returns:
        Normalized text string
    """
    if not text:
        return ""
    
    # Apply Unicode NFKC normalization
    text = unicodedata.normalize('NFKC', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    # Replace runs of whitespace (space, tab, newline) with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove control characters except standard whitespace
    # Keep printable characters and standard whitespace
    text = ''.join(char for char in text if not unicodedata.category(char).startswith('C') or char in ' \t\n')
    
    # Final strip and whitespace normalization
    text = ' '.join(text.split())
    
    return text


# ============================================================
# RAW DATA LOADING
# ============================================================

def load_raw_data(raw_dir: Path) -> List[RawFamily]:
    """
    Load all raw JSON/JSONL files from raw_dir.
    
    Args:
        raw_dir: Path to directory containing raw data files
        
    Returns:
        List of RawFamily objects
    """
    families = []
    raw_files = list(raw_dir.glob("*.json")) + list(raw_dir.glob("*.jsonl"))
    
    if not raw_files:
        print(f"Warning: No JSON/JSONL files found in {raw_dir}")
        return families
    
    print(f"Found {len(raw_files)} raw data file(s)")
    
    for file_path in raw_files:
        print(f"  Loading: {file_path.name}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            # Handle both JSON array and JSONL formats
            if file_path.suffix == '.jsonl':
                records = [json.loads(line) for line in f if line.strip()]
            else:
                content = f.read()
                if content.strip():
                    data = json.loads(content)
                    # Handle both list and single object
                    records = data if isinstance(data, list) else [data]
                else:
                    records = []
        
        for record in records:
            # Extract required fields
            idx = record.get('idx')
            source = record.get('dataset_source', '').lower()
            
            # Skip if missing idx or source
            if idx is None or not source:
                continue
            
            # Construct family_id
            family_id = f"{source}_{idx}"
            
            # Extract type fields
            type1 = record.get(RAW_FIELD_TYPE1)
            type2 = record.get(RAW_FIELD_TYPE2)
            type3 = record.get(RAW_FIELD_TYPE3)
            type4 = record.get(RAW_FIELD_TYPE4)
            
            # Create RawFamily object
            family = RawFamily(
                family_id=family_id,
                source=source,
                type1=type1,
                type2=type2,
                type3=type3,
                type4=type4,
            )
            families.append(family)
    
    print(f"Loaded {len(families)} families from raw data")
    return families


# ============================================================
# ROW MATERIALIZATION
# ============================================================

def materialize_rows(families: List[RawFamily]) -> Tuple[List[ProcessedRow], Dict[str, int]]:
    """
    Convert families into individual classification rows.
    
    For each family, create up to 4 rows (one per valid type).
    Apply text normalization and compute statistics.
    
    Args:
        families: List of RawFamily objects
        
    Returns:
        Tuple of (list of ProcessedRow objects, dict of drop counts by reason)
    """
    rows = []
    drop_counts = {
        "missing_family_id": 0,
        "empty_text": 0,
        "too_long": 0,
    }
    
    for family in families:
        if not family.family_id:
            drop_counts["missing_family_id"] += 1
            continue
        
        family_has_valid_row = False
        
        # Process each type field
        for field_name, label in RAW_FIELD_TO_LABEL.items():
            # Get the corresponding text from family
            if field_name == RAW_FIELD_TYPE1:
                text = family.type1
            elif field_name == RAW_FIELD_TYPE2:
                text = family.type2
            elif field_name == RAW_FIELD_TYPE3:
                text = family.type3
            elif field_name == RAW_FIELD_TYPE4:
                text = family.type4
            else:
                continue
            
            # Skip if text is None or empty
            if not text or not text.strip():
                continue
            
            # Normalize text
            normalized_text = normalize_text(text)
            
            # Skip if empty after normalization
            if not normalized_text:
                drop_counts["empty_text"] += 1
                continue
            
            # Check length constraint
            if len(normalized_text) > MAX_TEXT_LENGTH_CHARS:
                drop_counts["too_long"] += 1
                continue
            
            # Create row
            row_id = f"{family.family_id}__{label}"
            label_id = LABEL2ID[label]
            text_len_char = len(normalized_text)
            text_len_word = len(normalized_text.split())
            
            row = ProcessedRow(
                id=row_id,
                family_id=family.family_id,
                source=family.source,
                text=normalized_text,
                label=label,
                label_id=label_id,
                text_len_char=text_len_char,
                text_len_word=text_len_word,
            )
            rows.append(row)
            family_has_valid_row = True
    
    print(f"Materialized {len(rows)} rows from families")
    return rows, drop_counts


# ============================================================
# FAMILY-AWARE SPLITTING
# ============================================================

def family_aware_split(
    rows: List[ProcessedRow],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int
) -> Tuple[List[ProcessedRow], List[ProcessedRow], List[ProcessedRow]]:
    """
    Perform family-aware train/val/test split.
    
    All rows from the same family must go to the same split.
    
    Args:
        rows: List of all ProcessedRow objects
        train_ratio: Proportion for training (e.g., 0.7)
        val_ratio: Proportion for validation (e.g., 0.15)
        test_ratio: Proportion for test (e.g., 0.15)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_rows, val_rows, test_rows)
    """
    # Group rows by family_id
    family_to_rows = defaultdict(list)
    for row in rows:
        family_to_rows[row.family_id].append(row)
    
    # Get list of unique family IDs
    family_ids = sorted(family_to_rows.keys())
    num_families = len(family_ids)
    
    print(f"Splitting {num_families} families into train/val/test")
    
    # Shuffle family IDs deterministically
    rng = np.random.RandomState(seed)
    shuffled_families = family_ids.copy()
    rng.shuffle(shuffled_families)
    
    # Calculate split points
    train_end = int(num_families * train_ratio)
    val_end = train_end + int(num_families * val_ratio)
    
    train_families = set(shuffled_families[:train_end])
    val_families = set(shuffled_families[train_end:val_end])
    test_families = set(shuffled_families[val_end:])
    
    print(f"  Train families: {len(train_families)}")
    print(f"  Val families: {len(val_families)}")
    print(f"  Test families: {len(test_families)}")
    
    # Assign rows to splits
    train_rows = []
    val_rows = []
    test_rows = []
    
    for family_id, family_rows in family_to_rows.items():
        if family_id in train_families:
            train_rows.extend(family_rows)
        elif family_id in val_families:
            val_rows.extend(family_rows)
        elif family_id in test_families:
            test_rows.extend(family_rows)
    
    print(f"  Train rows: {len(train_rows)}")
    print(f"  Val rows: {len(val_rows)}")
    print(f"  Test rows: {len(test_rows)}")
    
    return train_rows, val_rows, test_rows


# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate_within_split(rows: List[ProcessedRow]) -> List[ProcessedRow]:
    """
    Remove exact duplicate rows within a split.
    
    Rows with same text and label are considered duplicates.
    Keep only the first occurrence.
    
    Args:
        rows: List of ProcessedRow objects
        
    Returns:
        Deduplicated list of ProcessedRow objects
    """
    seen = set()
    deduplicated = []
    
    for row in rows:
        key = (row.text, row.label)
        if key not in seen:
            seen.add(key)
            deduplicated.append(row)
    
    num_duplicates = len(rows) - len(deduplicated)
    if num_duplicates > 0:
        print(f"    Removed {num_duplicates} duplicate rows")
    
    return deduplicated


# ============================================================
# VALIDATION
# ============================================================

def validate_splits(
    train_rows: List[ProcessedRow],
    val_rows: List[ProcessedRow],
    test_rows: List[ProcessedRow]
) -> bool:
    """
    Validate that splits meet all requirements.
    
    Checks:
    1. No family_id overlap between splits
    2. All required fields present and valid
    3. Each split has at least one example of each label (warn if not)
    
    Args:
        train_rows: Training rows
        val_rows: Validation rows
        test_rows: Test rows
        
    Returns:
        True if all critical checks pass
    """
    print("\nValidating splits...")
    
    # Check 1: No family_id overlap
    train_families = set(row.family_id for row in train_rows)
    val_families = set(row.family_id for row in val_rows)
    test_families = set(row.family_id for row in test_rows)
    
    overlap_train_val = train_families & val_families
    overlap_train_test = train_families & test_families
    overlap_val_test = val_families & test_families
    
    if overlap_train_val or overlap_train_test or overlap_val_test:
        print(f"  ✗ FAIL: Family leakage detected!")
        print(f"    Train-Val overlap: {len(overlap_train_val)} families")
        print(f"    Train-Test overlap: {len(overlap_train_test)} families")
        print(f"    Val-Test overlap: {len(overlap_val_test)} families")
        return False
    else:
        print(f"  ✓ PASS: No family leakage detected")
    
    # Check 2: Validate field values
    all_rows = train_rows + val_rows + test_rows
    for row in all_rows:
        assert row.text and row.text.strip(), f"Empty text in row {row.id}"
        assert row.source in VALID_SOURCES, f"Invalid source: {row.source}"
        assert row.label in LABELS, f"Invalid label: {row.label}"
        assert row.label_id == LABEL2ID[row.label], f"Label ID mismatch in row {row.id}"
    
    print(f"  ✓ PASS: All rows have valid field values")
    
    # Check 3: Label distribution in each split
    for split_name, split_rows in [("Train", train_rows), ("Val", val_rows), ("Test", test_rows)]:
        label_counts = Counter(row.label for row in split_rows)
        missing_labels = set(LABELS) - set(label_counts.keys())
        
        if missing_labels:
            print(f"  ⚠ WARNING: {split_name} split missing labels: {missing_labels}")
        else:
            print(f"  ✓ {split_name} split has all labels")
    
    return True


# ============================================================
# OUTPUT WRITING
# ============================================================

def write_jsonl(rows: List[ProcessedRow], output_path: Path):
    """
    Write rows to JSONL file (one JSON object per line).
    
    Args:
        rows: List of ProcessedRow objects
        output_path: Path to output JSONL file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for row in rows:
            json_line = json.dumps(row.to_dict(), ensure_ascii=False)
            f.write(json_line + '\n')
    
    print(f"  Wrote {len(rows)} rows to {output_path.name}")


def compute_manifest(
    train_rows: List[ProcessedRow],
    val_rows: List[ProcessedRow],
    test_rows: List[ProcessedRow],
    drop_counts: Dict[str, int],
    seed: int,
    split_ratios: Dict[str, float]
) -> Dict[str, Any]:
    """
    Compute manifest statistics.
    
    Args:
        train_rows: Training rows
        val_rows: Validation rows
        test_rows: Test rows
        drop_counts: Counts of dropped rows by reason
        seed: Random seed used
        split_ratios: Split ratios used
        
    Returns:
        Manifest dictionary
    """
    all_rows = train_rows + val_rows + test_rows
    
    # Collect unique families and compute hash
    all_families = sorted(set(row.family_id for row in all_rows))
    families_str = ''.join(all_families)
    families_hash = hashlib.sha1(families_str.encode('utf-8')).hexdigest()
    
    # Label counts per split
    def label_counts(rows):
        counts = Counter(row.label for row in rows)
        return {label: counts.get(label, 0) for label in LABELS}
    
    # Average lengths per split
    def avg_lengths(rows):
        if not rows:
            return 0.0, 0.0
        avg_char = sum(row.text_len_char for row in rows) / len(rows)
        avg_word = sum(row.text_len_word for row in rows) / len(rows)
        return round(avg_char, 2), round(avg_word, 2)
    
    train_avg_char, train_avg_word = avg_lengths(train_rows)
    val_avg_char, val_avg_word = avg_lengths(val_rows)
    test_avg_char, test_avg_word = avg_lengths(test_rows)
    
    manifest = {
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
        "seed": seed,
        "split_ratios": split_ratios,
        "total_rows": len(all_rows),
        "total_families": len(all_families),
        "rows_per_split": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
        "rows_per_label_per_split": {
            "train": label_counts(train_rows),
            "val": label_counts(val_rows),
            "test": label_counts(test_rows),
        },
        "avg_len_char_per_split": {
            "train": train_avg_char,
            "val": val_avg_char,
            "test": test_avg_char,
        },
        "avg_len_word_per_split": {
            "train": train_avg_word,
            "val": val_avg_word,
            "test": test_avg_word,
        },
        "num_dropped_rows": drop_counts,
        "families_hash_sha1": families_hash,
    }
    
    return manifest


def write_manifest(manifest: Dict[str, Any], output_path: Path):
    """
    Write manifest JSON file.
    
    Args:
        manifest: Manifest dictionary
        output_path: Path to output JSON file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    print(f"  Wrote manifest to {output_path.name}")


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    """
    Main data preparation pipeline.
    """
    print("=" * 60)
    print("NLP Multi-Type Classification: Data Preparation")
    print("=" * 60)
    
    # Load configuration
    config_path = Path("configs/data_config.yaml")
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    raw_dir = Path(config.get('raw_dir', 'data/raw'))
    processed_dir = Path(config.get('processed_dir', 'data/processed'))
    
    split_config = config.get('split', {})
    seed = split_config.get('seed', DEFAULT_SEED)
    split_ratios = {
        'train': split_config.get('train_ratio', DEFAULT_SPLIT_RATIOS['train']),
        'val': split_config.get('val_ratio', DEFAULT_SPLIT_RATIOS['val']),
        'test': split_config.get('test_ratio', DEFAULT_SPLIT_RATIOS['test']),
    }
    
    print(f"\nConfiguration:")
    print(f"  Raw data directory: {raw_dir}")
    print(f"  Processed data directory: {processed_dir}")
    print(f"  Split ratios: {split_ratios}")
    print(f"  Random seed: {seed}")
    
    # Create output directory
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load raw data
    print(f"\n[1/7] Loading raw data from {raw_dir}")
    families = load_raw_data(raw_dir)
    
    if not families:
        print("Error: No families loaded from raw data")
        sys.exit(1)
    
    # Step 2: Materialize rows
    print(f"\n[2/7] Materializing rows from families")
    rows, drop_counts = materialize_rows(families)
    
    if not rows:
        print("Error: No valid rows materialized")
        sys.exit(1)
    
    # Step 3: Family-aware split
    print(f"\n[3/7] Performing family-aware split")
    train_rows, val_rows, test_rows = family_aware_split(
        rows,
        split_ratios['train'],
        split_ratios['val'],
        split_ratios['test'],
        seed
    )
    
    # Step 4: Deduplicate within splits
    print(f"\n[4/7] Deduplicating within splits")
    print(f"  Train split:")
    train_rows = deduplicate_within_split(train_rows)
    print(f"  Val split:")
    val_rows = deduplicate_within_split(val_rows)
    print(f"  Test split:")
    test_rows = deduplicate_within_split(test_rows)
    
    # Step 5: Validate splits
    print(f"\n[5/7] Validating splits")
    if not validate_splits(train_rows, val_rows, test_rows):
        print("\nError: Validation failed")
        sys.exit(1)
    
    # Step 6: Write JSONL files
    print(f"\n[6/7] Writing JSONL files to {processed_dir}")
    write_jsonl(train_rows, processed_dir / TRAIN_FILENAME)
    write_jsonl(val_rows, processed_dir / VAL_FILENAME)
    write_jsonl(test_rows, processed_dir / TEST_FILENAME)
    
    # Step 7: Compute and write manifest
    print(f"\n[7/7] Computing and writing manifest")
    manifest = compute_manifest(train_rows, val_rows, test_rows, drop_counts, seed, split_ratios)
    write_manifest(manifest, processed_dir / MANIFEST_FILENAME)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total rows: {manifest['total_rows']}")
    print(f"Total families: {manifest['total_families']}")
    print(f"\nRows per split:")
    for split, count in manifest['rows_per_split'].items():
        print(f"  {split}: {count}")
    print(f"\nRows per label per split:")
    for split, label_counts in manifest['rows_per_label_per_split'].items():
        print(f"  {split}:")
        for label, count in label_counts.items():
            print(f"    {label}: {count}")
    print(f"\nDropped rows:")
    for reason, count in manifest['num_dropped_rows'].items():
        if count > 0:
            print(f"  {reason}: {count}")
    print("\nData preparation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
