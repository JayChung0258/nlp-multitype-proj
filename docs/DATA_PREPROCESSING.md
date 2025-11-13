# Data Preprocessing Pipeline

**NLP Multi-Type Classification Project**

This document describes the data preprocessing pipeline that transforms raw paraphrase data into training-ready JSONL datasets.

---

## Table of Contents

1. [Overview](#overview)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Input Data Format](#input-data-format)
4. [Processing Steps](#processing-steps)
5. [Output Data Format](#output-data-format)
6. [Running the Pipeline](#running-the-pipeline)
7. [Validation and Quality Checks](#validation-and-quality-checks)
8. [Troubleshooting](#troubleshooting)

---

## Overview

### Purpose

The preprocessing pipeline converts raw multi-type paraphrase data from three sources (MRPC, PAWS, HLPC) into a unified dataset for 4-class single-sentence classification.

### Task Definition

- **Input:** One sentence
- **Output:** One of four labels:
  - **T1** (0): Human Original
  - **T2** (1): LLM Generated
  - **T3** (2): Human Paraphrased
  - **T4** (3): LLM Paraphrased

### Key Features

- ✅ **JSONL format** for efficient processing
- ✅ **Family-aware splitting** to prevent data leakage
- ✅ **Text normalization** (Unicode NFKC, whitespace cleanup)
- ✅ **Automatic validation** with leakage detection
- ✅ **Comprehensive statistics** in manifest file

---

## Pipeline Architecture

### 7-Step Processing Pipeline

```
┌─────────────────────┐
│  1. Load Raw Data   │  Read JSON/JSONL from data/raw/
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ 2. Materialize Rows │  Extract Type1-Type4, create individual rows
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ 3. Family-Aware     │  Split families: 70% train, 15% val, 15% test
│    Split            │  No family appears in multiple splits
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ 4. Deduplication    │  Remove exact duplicates within each split
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ 5. Validation       │  Check for leakage, valid labels, sources
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ 6. Write JSONL      │  Output train/val/test JSONL files
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ 7. Generate         │  Create manifest with statistics
│    Manifest         │
└─────────────────────┘
```

### Implementation

- **Script:** `src/data_prep.py` (~650 lines)
- **Entry point:** `python3 -m src.data_prep`
- **Configuration:** `configs/data_config.yaml`

---

## Input Data Format

### Raw Data Structure

**Location:** `data/raw/`

**Format:** JSON or JSONL files with the following structure:

```json
{
  "idx": 0,
  "dataset_source": "mrpc",
  "human_original_text(type1)": "Original human-written text.",
  "llm_generated_text(type2)": "LLM-generated text.",
  "human_paraphrased_text(type3)": "Human paraphrase of original.",
  "llm_paraphrased_original_text(type4)-prompt-based": "LLM paraphrase of original.",
  "llm_paraphrased_generated_text(type5)-1st": "...",
  "llm_paraphrased_generated_text(type5)-3rd": "..."
}
```

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `idx` | integer | Unique index within source dataset |
| `dataset_source` | string | Data source: "mrpc", "paws", or "hlpc" |
| `human_original_text(type1)` | string/null | Type1 text |
| `llm_generated_text(type2)` | string/null | Type2 text |
| `human_paraphrased_text(type3)` | string/null | Type3 text |
| `llm_paraphrased_original_text(type4)-prompt-based` | string/null | Type4 text |

**Note:** Type5 fields are present but **completely ignored** during processing.

### Family ID Construction

The pipeline constructs a unique `family_id` for each record:

```
family_id = "<dataset_source>_<idx>"
```

Examples: `"mrpc_0"`, `"paws_42"`, `"hlpc_999"`

---

## Processing Steps

### Step 1: Load Raw Data

- Reads all `.json` and `.jsonl` files from `data/raw/`
- Supports both JSON arrays and JSONL format
- Constructs `family_id` from `dataset_source` and `idx`
- Skips records with missing `idx` or `dataset_source`

### Step 2: Materialize Rows

For each family, creates up to 4 individual rows (one per type):

```python
# For each non-null type field:
row = {
    "id": f"{family_id}__{label}",  # e.g., "mrpc_0__T1"
    "family_id": family_id,
    "source": dataset_source,
    "text": normalized_text,
    "label": label,  # T1, T2, T3, or T4
    "label_id": label_id,  # 0, 1, 2, or 3
    "text_len_char": len(text),
    "text_len_word": len(text.split())
}
```

**Drop conditions:**
- Empty text after normalization
- Text length > 4000 characters
- Missing `family_id`

### Step 3: Text Normalization

Applied to all text fields:

1. **Unicode NFKC normalization**
   ```python
   text = unicodedata.normalize('NFKC', text)
   ```

2. **Whitespace cleanup**
   - Strip leading/trailing whitespace
   - Replace runs of whitespace with single space
   ```python
   text = re.sub(r'\s+', ' ', text.strip())
   ```

3. **Control character removal**
   - Keep printable characters and standard whitespace only
   - Remove null bytes, other control characters

**What is NOT changed:**
- ❌ Word content (no stemming/lemmatization)
- ❌ Casing (original case preserved)
- ❌ Punctuation (kept as-is)
- ❌ Grammar (no corrections)

### Step 4: Family-Aware Splitting

**Critical for preventing data leakage:**

```python
# Group rows by family_id
families = get_unique_families(rows)

# Shuffle families deterministically (seed=42)
np.random.RandomState(42).shuffle(families)

# Split families
train_families = families[:70%]
val_families = families[70%:85%]
test_families = families[85%:]

# Assign all rows from a family to the same split
```

**Split ratios:**
- Training: 70% of families
- Validation: 15% of families
- Test: 15% of families

**Guarantee:** No `family_id` appears in multiple splits.

### Step 5: Deduplication

Within each split, removes exact duplicate rows:

- **Duplicate definition:** Same `text` and `label`
- **Action:** Keep first occurrence, drop subsequent
- **Note:** Different labels with same text are kept (meaningful)

### Step 6: Validation

**Critical checks (fail if violated):**
1. No family leakage (no `family_id` in multiple splits)
2. All `label` values in {"T1", "T2", "T3", "T4"}
3. All `label_id` values match `LABEL2ID` mapping
4. All `source` values in {"mrpc", "paws", "hlpc"}
5. All texts are non-empty

**Warning checks (log but continue):**
- Missing labels in validation or test splits

### Step 7: Output Writing

**JSONL files:**
- One JSON object per line
- UTF-8 encoding
- Consistent field order

**Manifest JSON:**
- Dataset statistics
- Split information
- SHA1 hash of family IDs (for verification)

---

## Output Data Format

### Processed JSONL Files

**Location:** `data/processed/`

**Files:**
- `train_4class.jsonl` — Training split
- `val_4class.jsonl` — Validation split
- `test_4class.jsonl` — Test split
- `manifest.json` — Statistics and metadata

### JSONL Row Schema

```json
{
  "id": "mrpc_0__T1",
  "family_id": "mrpc_0",
  "source": "mrpc",
  "text": "Normalized sentence text.",
  "label": "T1",
  "label_id": 0,
  "text_len_char": 24,
  "text_len_word": 3
}
```

### Manifest Schema

```json
{
  "created_at_utc": "2025-11-13T22:39:11Z",
  "seed": 42,
  "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
  "total_rows": 19959,
  "total_families": 5000,
  "rows_per_split": {
    "train": 13966,
    "val": 2996,
    "test": 2997
  },
  "rows_per_label_per_split": {
    "train": {"T1": 3500, "T2": 3498, "T3": 3468, "T4": 3500},
    "val": {"T1": 750, "T2": 749, "T3": 747, "T4": 750},
    "test": {"T1": 750, "T2": 750, "T3": 747, "T4": 750}
  },
  "avg_len_char_per_split": {
    "train": 145.19,
    "val": 140.49,
    "test": 149.42
  },
  "avg_len_word_per_split": {
    "train": 23.5,
    "val": 22.8,
    "test": 24.1
  },
  "num_dropped_rows": {
    "missing_family_id": 0,
    "empty_text": 0,
    "too_long": 3
  },
  "families_hash_sha1": "a1b2c3d4e5f6..."
}
```

---

## Running the Pipeline

### Prerequisites

```bash
# Install required packages
pip3 install --break-system-packages pyyaml numpy

# Or use virtual environment
python3 -m venv venv
source venv/bin/activate
pip install pyyaml numpy
```

### Execution

```bash
# Navigate to project root
cd /path/to/nlp-multitype-proj

# Run preprocessing
python3 -m src.data_prep
```

### Configuration

Edit `configs/data_config.yaml` to customize:

```yaml
raw_dir: data/raw
processed_dir: data/processed

split:
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
  seed: 42  # Change for different splits
```

### Expected Runtime

- **Small dataset** (5K families): ~5-10 seconds
- **Large dataset** (50K families): ~30-60 seconds

---

## Validation and Quality Checks

### Automatic Validation

The pipeline performs these checks automatically:

#### 1. Family Leakage Check

```python
train_families = set(train_df['family_id'])
val_families = set(val_df['family_id'])
test_families = set(test_df['family_id'])

assert len(train_families & val_families) == 0
assert len(train_families & test_families) == 0
assert len(val_families & test_families) == 0
```

**Result:** ✓ or ✗ with explicit error

#### 2. Label Validation

```python
for row in all_rows:
    assert row['label'] in ['T1', 'T2', 'T3', 'T4']
    assert row['label_id'] == LABEL2ID[row['label']]
```

#### 3. Source Validation

```python
assert row['source'] in ['mrpc', 'paws', 'hlpc']
```

### Manual Verification

After processing, verify output:

```bash
# Check file sizes
ls -lh data/processed/

# Count rows per split
wc -l data/processed/*.jsonl

# View sample rows
head -n 5 data/processed/train_4class.jsonl | python3 -m json.tool

# Check manifest
cat data/processed/manifest.json | python3 -m json.tool

# Verify no family leakage (should return 0)
python3 << EOF
import json
train = [json.loads(line) for line in open('data/processed/train_4class.jsonl')]
val = [json.loads(line) for line in open('data/processed/val_4class.jsonl')]
test = [json.loads(line) for line in open('data/processed/test_4class.jsonl')]

train_fams = set(r['family_id'] for r in train)
val_fams = set(r['family_id'] for r in val)
test_fams = set(r['family_id'] for r in test)

overlap = len(train_fams & val_fams) + len(train_fams & test_fams) + len(val_fams & test_fams)
print(f"Family overlap count: {overlap}")
EOF
```

---

## Troubleshooting

### Common Issues

#### Issue: `ModuleNotFoundError: No module named 'yaml'`

**Solution:**
```bash
pip3 install --break-system-packages pyyaml numpy
```

#### Issue: `No JSON/JSONL files found in data/raw`

**Cause:** Raw data files not present or wrong location

**Solution:**
```bash
# Verify raw data exists
ls -la data/raw/

# Check file extensions
file data/raw/*
```

#### Issue: Family leakage detected

**Symptom:**
```
✗ FAIL: Family leakage detected!
  Train-Val overlap: 5 families
```

**Cause:** Bug in splitting logic or duplicate family IDs in raw data

**Solution:**
1. Check for duplicate `idx` within same `dataset_source`
2. Verify `family_id` construction is correct
3. Re-run with different seed

#### Issue: All samples from one source in one split

**Symptom:** Imbalanced source distribution across splits

**Explanation:** This is expected with family-aware splitting. Families are shuffled, so sources naturally distribute across splits.

**Verification:**
```bash
python3 << EOF
import json
for split in ['train', 'val', 'test']:
    rows = [json.loads(line) for line in open(f'data/processed/{split}_4class.jsonl')]
    sources = [r['source'] for r in rows]
    print(f"{split}: mrpc={sources.count('mrpc')}, paws={sources.count('paws')}, hlpc={sources.count('hlpc')}")
EOF
```

#### Issue: Missing labels in validation split

**Symptom:**
```
⚠ WARNING: Val split missing labels: {'T3'}
```

**Explanation:** With 15% split and small datasets, some rare labels may not appear

**Solution:**
- If T3 is very rare: acceptable (just a warning)
- If T3 is common: check raw data for missing Type3 fields
- Consider using stratified splitting (future enhancement)

### Debug Mode

For detailed debugging, modify the script to print intermediate results:

```python
# In data_prep.py, after materialization:
print(f"Sample rows: {rows[:5]}")

# After splitting:
print(f"Train families sample: {list(train_families)[:10]}")
```

---

## Performance Characteristics

### Processing Speed

| Dataset Size | Families | Rows | Processing Time |
|--------------|----------|------|-----------------|
| Small | 1K | 4K | ~2 seconds |
| Medium | 5K | 20K | ~8 seconds |
| Large | 50K | 200K | ~60 seconds |
| Very Large | 500K | 2M | ~10 minutes |

### Memory Usage

- **Small datasets** (< 10K families): < 100 MB
- **Medium datasets** (< 100K families): < 500 MB
- **Large datasets** (< 1M families): < 2 GB

### Scalability

The pipeline is designed for datasets up to ~1M families. For larger datasets:
- Process in batches
- Use streaming JSONL processing
- Consider distributed processing (Spark, Dask)

---

## Best Practices

### 1. Always Use Family-Aware Splitting

❌ **Don't:**
```python
# Random split - CAUSES LEAKAGE
train, val, test = random_split(rows, [0.7, 0.15, 0.15])
```

✅ **Do:**
```python
# Group by family first
train, val, test = family_aware_split(rows, families)
```

### 2. Verify No Leakage

Always check the manifest and validation output:
```
✓ PASS: No family leakage detected
```

### 3. Monitor Dropped Rows

Check `num_dropped_rows` in manifest:
```json
"num_dropped_rows": {
  "missing_family_id": 0,
  "empty_text": 23,
  "too_long": 3
}
```

If > 5% of rows are dropped, investigate the cause.

### 4. Document Your Seed

Always record the random seed used:
```yaml
split:
  seed: 42  # Document in paper/README
```

### 5. Version Your Processed Data

```bash
# Tag processed data with hash
HASH=$(cat data/processed/manifest.json | grep families_hash | cut -d'"' -f4)
echo "Processed data hash: $HASH" > data/processed/VERSION
```

---

## Related Documentation

- **Data Contract:** [`DATA_CONTRACT.md`](DATA_CONTRACT.md) — Complete schema specifications
- **Local Runbook:** [`RUNBOOK_LOCAL.md`](RUNBOOK_LOCAL.md) — Step-by-step execution guide
- **Data README:** [`../data/README.md`](../data/README.md) — Data directory overview

---

*Last updated: 2025-11-13*

