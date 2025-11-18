# Data Contract (JSONL Format)

**NLP Multi-Type Classification Project**

This document defines the formal schemas, validation rules, and data contracts for all stages of the data pipeline using JSONL format.

---

## Table of Contents

1. [Raw Data Schema](#raw-data-schema)
2. [Processed Data Schema](#processed-data-schema)
3. [Label Mapping](#label-mapping)
4. [Split Configuration](#split-configuration)
5. [Validation Rules](#validation-rules)
6. [File Formats and Naming](#file-formats-and-naming)
7. [Text Normalization](#text-normalization)
8. [Manifest Schema](#manifest-schema)

---

## Raw Data Schema

### Family-Based Input Format

Raw data comes from three sources: **MRPC**, **PAWS**, and **HLPC**. Each record represents a "family" with multiple paraphrase variants.

#### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `idx` | integer | Unique index within the source dataset |
| `dataset_source` | string | One of: "mrpc", "paws", "hlpc" |
| `human_original_text(type1)` | string or null | Human-written original text |
| `llm_generated_text(type2)` | string or null | LLM-generated text |
| `human_paraphrased_text(type3)` | string or null | Human paraphrase of original |
| `llm_paraphrased_original_text(type4)-prompt-based` | string or null | LLM paraphrase of original |

#### Ignored Fields

Type5 fields are present in raw data but **MUST BE IGNORED**:
- `llm_paraphrased_generated_text(type5)-1st`
- `llm_paraphrased_generated_text(type5)-3rd`

#### Family ID Construction

`family_id` is constructed as: `"<dataset_source>_<idx>"`

Examples:
- `mrpc_42`
- `paws_1337`
- `hlpc_999`

#### Example Raw Record

```json
{
  "idx": 0,
  "dataset_source": "mrpc",
  "human_original_text(type1)": "The weather today is sunny and warm.",
  "llm_generated_text(type2)": "Today's weather is characterized by sunshine and warmth.",
  "human_paraphrased_text(type3)": "It's a sunny, warm day outside.",
  "llm_paraphrased_original_text(type4)-prompt-based": "The current meteorological conditions feature abundant sunshine.",
  "llm_paraphrased_generated_text(type5)-1st": "...",
  "llm_paraphrased_generated_text(type5)-3rd": "..."
}
```

### Expected File Formats

- **JSON:** Array of objects
- **JSONL:** One JSON object per line (preferred for large datasets)

---

## Processed Data Schema

### Single-Sentence Classification Format (JSONL)

After data preparation, each sentence becomes an individual classification sample stored in JSONL format.

#### Required Fields

| Field | Type | Allowed Values | Description |
|-------|------|----------------|-------------|
| `id` | string | `"<family_id>__<label>"` | Globally unique row identifier |
| `family_id` | string | Non-empty string | Links to original family (format: `<source>_<idx>`) |
| `source` | string | `"mrpc"`, `"paws"`, `"hlpc"` | Data source |
| `text` | string | Non-empty, normalized | The sentence to classify |
| `label` | string | `"T1"`, `"T2"`, `"T3"`, `"T4"` | Class label |
| `label_id` | integer | `0`, `1`, `2`, `3` | Numeric label mapping |
| `text_len_char` | integer | Positive integer | Character count |
| `text_len_word` | integer | Positive integer | Word count (whitespace split) |

#### Example Processed Rows (JSONL)

```jsonl
{"id": "mrpc_0__T1", "family_id": "mrpc_0", "source": "mrpc", "text": "The weather today is sunny and warm.", "label": "T1", "label_id": 0, "text_len_char": 39, "text_len_word": 7}
{"id": "mrpc_0__T2", "family_id": "mrpc_0", "source": "mrpc", "text": "Today's weather is characterized by sunshine and warmth.", "label": "T2", "label_id": 1, "text_len_char": 58, "text_len_word": 8}
{"id": "mrpc_0__T3", "family_id": "mrpc_0", "source": "mrpc", "text": "It's a sunny, warm day outside.", "label": "T3", "label_id": 2, "text_len_char": 33, "text_len_word": 6}
{"id": "mrpc_0__T4", "family_id": "mrpc_0", "source": "mrpc", "text": "The current meteorological conditions feature abundant sunshine.", "label": "T4", "label_id": 3, "text_len_char": 65, "text_len_word": 9}
```

---

## Label Mapping

### Canonical Label Definition

| String Label | Integer Label | Description | Raw Field Name |
|--------------|---------------|-------------|----------------|
| **T1** | **0** | Human Original | `human_original_text(type1)` |
| **T2** | **1** | LLM Generated | `llm_generated_text(type2)` |
| **T3** | **2** | Human Paraphrased | `human_paraphrased_text(type3)` |
| **T4** | **3** | LLM Paraphrased | `llm_paraphrased_original_text(type4)-prompt-based` |

### Consistency Requirements

- **All code must use this mapping**
- No alternative mappings are allowed
- Label conversion uses `constants.LABEL2ID` and `constants.ID2LABEL`
- Labels are stored as strings (`"T1"`, `"T2"`, `"T3"`, `"T4"`) in JSONL
- `label_id` field provides the numeric mapping for training

---

## Split Configuration

### Family-Aware Splitting

**Critical Rule:** All samples from the same `family_id` must reside in the **same split** (train, val, or test).

#### Split Ratios

| Split | Ratio | Purpose |
|-------|-------|---------|
| **Train** | 70% | Model training |
| **Validation** | 15% | Hyperparameter tuning, early stopping |
| **Test** | 15% | Final evaluation (held out) |

#### Configuration Parameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `seed` | 42 | Random seed for reproducibility |
| `train_ratio` | 0.70 | Proportion of families for training |
| `val_ratio` | 0.15 | Proportion of families for validation |
| `test_ratio` | 0.15 | Proportion of families for test |

#### Split Determinism

- **Fixed seed** (default: 42) ensures reproducibility
- **Same split generated** across multiple runs with same config
- **Family grouping** prevents leakage

---

## Validation Rules

### Critical Validation Checks (Fail Build)

These checks **must pass** or the data preparation step fails:

1. **Label Validity:**
   - All `label` values must be in `{"T1", "T2", "T3", "T4"}`
   - All `label_id` values must match `LABEL2ID` mapping
   - No missing labels

2. **Text Non-Empty:**
   - All `text` values must be non-empty after normalization
   - No null or missing text values

3. **Family ID Non-Empty:**
   - All `family_id` values must be non-empty strings
   - Format: `"<source>_<idx>"`

4. **No Cross-Split Leakage:**
   - Each `family_id` must appear in **exactly one** split
   - No family can have samples in multiple splits

5. **Maximum Text Length:**
   - All texts must be ≤ 4000 characters
   - Texts exceeding this are dropped

6. **Valid Source:**
   - All `source` values must be in `{"mrpc", "paws", "hlpc"}`

### Warning-Level Checks (Log but Continue)

These checks trigger **warnings** but do not fail the build:

1. **Missing Labels in Splits:**
   - Warn if validation or test split is missing any label
   - Training split must have all labels (error if not)

---

## File Formats and Naming

### Processed JSONL Files

#### File Names

| File | Location | Description |
|------|----------|-------------|
| `train_4class.jsonl` | `data/processed/` | Training split |
| `val_4class.jsonl` | `data/processed/` | Validation split |
| `test_4class.jsonl` | `data/processed/` | Test split |
| `manifest.json` | `data/processed/` | Dataset statistics and metadata |

#### JSONL Format Requirements

- **Encoding:** UTF-8
- **Format:** One JSON object per line
- **No trailing commas**
- **Consistent field order:** id, family_id, source, text, label, label_id, text_len_char, text_len_word

#### Example JSONL File Structure

```jsonl
{"id": "mrpc_0__T1", "family_id": "mrpc_0", "source": "mrpc", "text": "Example text.", "label": "T1", "label_id": 0, "text_len_char": 13, "text_len_word": 2}
{"id": "mrpc_0__T2", "family_id": "mrpc_0", "source": "mrpc", "text": "Another example.", "label": "T2", "label_id": 1, "text_len_char": 16, "text_len_word": 2}
{"id": "paws_1__T3", "family_id": "paws_1", "source": "paws", "text": "Third example.", "label": "T3", "label_id": 2, "text_len_char": 14, "text_len_word": 2}
```

---

## Text Normalization

### Normalization Pipeline

The following preprocessing steps are applied to all text fields:

1. **Unicode NFKC Normalization:**
   - Apply `unicodedata.normalize('NFKC', text)`
   - Ensures consistent representation of characters

2. **Whitespace Stripping:**
   - Remove leading and trailing whitespace
   - Replace runs of whitespace (spaces, tabs, newlines) with single space

3. **Control Character Removal:**
   - Remove control characters except standard whitespace
   - Keep printable characters only

4. **Final Cleanup:**
   - Strip again and normalize internal whitespace

### What Is NOT Changed

- **Words:** No stemming, lemmatization
- **Casing:** Original case preserved
- **Punctuation:** Preserved as-is
- **Content:** No semantic modifications

### Example Normalization

```python
# Before:
"  The   weather\ttoday\n is sunny.  "

# After:
"The weather today is sunny."
```

---

## Manifest Schema

### Manifest JSON Structure

The `manifest.json` file contains dataset statistics and metadata.

#### Required Fields

```json
{
  "created_at_utc": "2025-11-13T12:00:00Z",
  "seed": 42,
  "split_ratios": {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
  },
  "total_rows": 15000,
  "total_families": 4500,
  "rows_per_split": {
    "train": 10500,
    "val": 2250,
    "test": 2250
  },
  "rows_per_label_per_split": {
    "train": {
      "T1": 2800,
      "T2": 2700,
      "T3": 2500,
      "T4": 2500
    },
    "val": {
      "T1": 600,
      "T2": 580,
      "T3": 535,
      "T4": 535
    },
    "test": {
      "T1": 600,
      "T2": 580,
      "T3": 535,
      "T4": 535
    }
  },
  "avg_len_char_per_split": {
    "train": 87.5,
    "val": 86.2,
    "test": 88.1
  },
  "avg_len_word_per_split": {
    "train": 14.3,
    "val": 14.1,
    "test": 14.5
  },
  "num_dropped_rows": {
    "missing_family_id": 0,
    "empty_text": 12,
    "too_long": 3
  },
  "families_hash_sha1": "a1b2c3d4e5f6..."
}
```

#### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `created_at_utc` | string | ISO 8601 timestamp of manifest creation |
| `seed` | integer | Random seed used for splitting |
| `split_ratios` | object | Train/val/test ratios used |
| `total_rows` | integer | Total number of rows across all splits |
| `total_families` | integer | Total number of unique families |
| `rows_per_split` | object | Row counts per split |
| `rows_per_label_per_split` | object | Label distribution per split |
| `avg_len_char_per_split` | object | Average character length per split |
| `avg_len_word_per_split` | object | Average word count per split |
| `num_dropped_rows` | object | Counts of rows dropped by reason |
| `families_hash_sha1` | string | SHA1 hash of sorted family IDs (for verification) |

---

## Data Quality Standards

### Deduplication Policy

**Within-Split Deduplication:**
- Remove exact duplicate rows (same text and label) within each split
- Keep first occurrence

**Cross-Split Handling:**
- Family-aware splitting prevents cross-split duplicates by design

### Text Length Constraints

| Constraint | Value | Action |
|------------|-------|--------|
| Minimum length | 1 character | Drop if violated |
| Hard maximum | 4000 characters | Drop if exceeded |

---

*Last updated: 2025-11-13*

