# Data Contract

**NLP Multi-Type Classification Project**

This document defines the formal schemas, validation rules, and data contracts for all stages of the data pipeline.

---

## Table of Contents

1. [Raw Data Schema](#raw-data-schema)
2. [Processed Data Schema](#processed-data-schema)
3. [Label Mapping](#label-mapping)
4. [Split Configuration](#split-configuration)
5. [Validation Rules](#validation-rules)
6. [File Formats and Naming](#file-formats-and-naming)
7. [Data Quality Standards](#data-quality-standards)

---

## Raw Data Schema

### Family-Based Input Format

Raw data is organized by **families**, where each family represents variants of the same base content across different generation/paraphrase types.

#### Required Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `family_id` | string | **Yes** | Unique identifier for grouping related samples |
| `type1` | string or null | No | Human original text |
| `type2` | string or null | No | LLM generated text |
| `type3` | string or null | No | Human paraphrased text |
| `type4` | string or null | No | LLM paraphrased text |

#### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `source` | string | Source of the data (e.g., dataset name, domain) |
| `domain` | string | Content domain (e.g., news, scientific, creative) |
| `language` | string | Language code (default: en) |
| `metadata` | object | Additional metadata |

#### Example Raw Row

```json
{
  "family_id": "fam_001",
  "type1": "The weather today is sunny and warm.",
  "type2": "Today's weather is characterized by sunshine and warmth.",
  "type3": "It's a sunny, warm day outside.",
  "type4": "The current meteorological conditions feature abundant sunshine and elevated temperatures.",
  "source": "weather_dataset",
  "domain": "weather"
}
```

### Expected File Formats

- **JSON:** List of objects, one per family
- **CSV:** One row per family, with columns for each field
- **JSONL:** One JSON object per line

---

## Processed Data Schema

### Single-Sentence Classification Format

After data preparation, each sentence becomes an individual classification sample.

#### Required Columns

| Column | Type | Required | Allowed Values | Description |
|--------|------|----------|----------------|-------------|
| `family_id` | string | **Yes** | Non-empty string | Links to original family (for leakage checks) |
| `text` | string | **Yes** | Non-empty, stripped | The sentence to classify |
| `label` | string | **Yes** | {T1, T2, T3, T4} | Class label |

**Alternative:** `label` can be integer in {0, 1, 2, 3} instead of string. Choose one scheme and apply consistently.

#### Optional Columns

| Column | Type | Description |
|--------|------|-------------|
| `text_len_char` | int | Character count (for analysis) |
| `text_len_word` | int | Word count (for analysis) |
| `split` | string | Split assignment (train/val/test), useful for merged manifests |

#### Example Processed Rows

```csv
family_id,text,label,text_len_char,text_len_word
fam_001,"The weather today is sunny and warm.",T1,39,7
fam_001,"Today's weather is characterized by sunshine and warmth.",T2,58,8
fam_001,"It's a sunny, warm day outside.",T3,33,6
fam_001,"The current meteorological conditions feature abundant sunshine and elevated temperatures.",T4,91,11
```

### Column Naming Conventions

- **All lowercase** with underscores (snake_case)
- **No spaces** in column names
- **Deterministic order:** Always use the same column order across files
- **Recommended order:** `family_id`, `text`, `label`, `text_len_char`, `text_len_word`, `split`

---

## Label Mapping

### Canonical Label Definition

| String Label | Integer Label | Description |
|--------------|---------------|-------------|
| **T1** | **0** | Human Original |
| **T2** | **1** | LLM Generated |
| **T3** | **2** | Human Paraphrased |
| **T4** | **3** | LLM Paraphrased |

### Consistency Requirements

- **All code must use this mapping**
- No alternative mappings are allowed
- Label conversion functions must use `constants.LABEL2ID` and `constants.ID2LABEL`

### Label Format Choice

Choose **one** format and use consistently:

- **Option A:** Store as strings (T1, T2, T3, T4) in CSVs
- **Option B:** Store as integers (0, 1, 2, 3) in CSVs

**Recommendation:** Use strings (T1-T4) for human readability in CSVs; convert to integers during model training.

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

| Parameter | Value | Description |
|-----------|-------|-------------|
| `train_ratio` | 0.70 | Proportion of families for training |
| `val_ratio` | 0.15 | Proportion of families for validation |
| `test_ratio` | 0.15 | Proportion of families for test |
| `seed` | 42 | Random seed for reproducibility |
| `group_key` | `family_id` | Column used for grouping |

#### Split Determinism

- **Must use fixed seed** (default: 42) for reproducibility
- **Same split must be generated** across multiple runs with same config
- **Document split seed** in all result files and reports

---

## Validation Rules

### Critical Validation Checks (Fail Build)

These checks **must pass** or the data preparation step should fail:

1. **Label Validity:**
   - All `label` values must be in {T1, T2, T3, T4} or {0, 1, 2, 3}
   - No missing labels

2. **Text Non-Empty:**
   - All `text` values must be non-empty after stripping whitespace
   - No null or missing text values

3. **Family ID Non-Empty:**
   - All `family_id` values must be non-empty strings
   - No null or missing family IDs

4. **No Cross-Split Leakage:**
   - Each `family_id` must appear in **exactly one** split
   - No family can have samples in both train and val, train and test, or val and test

5. **Maximum Text Length:**
   - All texts must be ≤ `max_len_chars` (default: 4000 characters)
   - Texts exceeding this should be flagged and either truncated or removed

### Warning-Level Checks (Log but Continue)

These checks trigger **warnings** but do not fail the build:

1. **Class Imbalance:**
   - Warn if any class represents < 10% of total samples
   - Print class distribution for manual review

2. **Text Length Outliers:**
   - Warn if any text is < 10 characters or > 2000 characters
   - Print outliers for manual inspection

3. **Duplicate Texts:**
   - Warn if exact duplicate texts are found
   - Count and report duplicates within and across splits

---

## File Formats and Naming

### Processed CSV Files

#### File Names

| File | Location | Description |
|------|----------|-------------|
| `train_4class.csv` | `data/processed/` | Training split |
| `val_4class.csv` | `data/processed/` | Validation split |
| `test_4class.csv` | `data/processed/` | Test split |

#### CSV Format Requirements

- **Encoding:** UTF-8
- **Delimiter:** Comma (`,`)
- **Quote Character:** Double quote (`"`)
- **Header:** First row must contain column names
- **Index Column:** Do not include row index column

#### Example CSV Structure

```csv
family_id,text,label
fam_001,"The weather is sunny.",T1
fam_001,"Today features sunshine.",T2
fam_002,"Machine learning is complex.",T1
fam_002,"ML algorithms can be intricate.",T3
```

---

## Data Quality Standards

### Preprocessing Requirements

The following preprocessing steps **must be applied**:

1. **Whitespace Stripping:**
   - Remove leading and trailing whitespace from `text`
   - Normalize internal whitespace (replace multiple spaces with single space)

2. **Unicode Normalization:**
   - Apply NFC (Canonical Decomposition, followed by Canonical Composition)
   - Ensures consistent representation of accented characters

3. **Empty Text Removal:**
   - Drop any samples where `text` is empty after preprocessing

4. **Deduplication:**
   - Remove exact duplicate texts within the dataset
   - Flag cross-split duplicates as potential leakage

### Allowed Character Set

- **Primary:** ASCII printable characters (0x20-0x7E)
- **Extended:** Unicode characters for non-English text
- **Prohibited:** Control characters (except newline in rare cases), null bytes

### Text Length Constraints

| Constraint | Value | Action |
|------------|-------|--------|
| Minimum length | 1 character | Fail validation if violated |
| Recommended minimum | 10 characters | Warn if violated |
| Recommended maximum | 2000 characters | Warn if violated |
| Hard maximum | 4000 characters | Fail validation if violated |

---

## Deduplication Policy

### Within-Split Deduplication

**Policy:** Remove exact duplicate texts within the same split.

**Rationale:** Duplicates do not add information and can bias metrics.

**Implementation:**
- After splitting, deduplicate each split independently
- Keep the first occurrence of each duplicate
- Log the number of duplicates removed

### Cross-Split Deduplication

**Policy:** Flag and investigate, but do not automatically remove.

**Rationale:** Cross-split duplicates indicate potential data leakage.

**Implementation:**
- Check if any text appears in multiple splits
- If found, log as **ERROR** and fail validation
- Require re-splitting with deduplication before splitting

**Recommendation:** Deduplicate the entire dataset **before** family-aware splitting to avoid cross-split duplicates.

---

## Schema Version and Changes

### Current Version

**Version:** 1.0.0  
**Date:** 2025-11-13

### Schema Stability

- **No breaking changes** without version increment
- **Backward compatibility** maintained within major version
- **Deprecation warnings** given at least one minor version before removal

### Change Log

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-11-13 | Initial data contract definition |

---

## Compliance Checklist

Use this checklist to verify data contract compliance:

- [ ] Raw data includes `family_id` for all rows
- [ ] Processed CSVs have required columns: `family_id`, `text`, `label`
- [ ] All labels are in {T1, T2, T3, T4} or {0, 1, 2, 3}
- [ ] All texts are non-empty and stripped
- [ ] Text length ≤ 4000 characters
- [ ] No `family_id` appears in multiple splits
- [ ] Split ratios are 70/15/15 (train/val/test)
- [ ] Fixed seed (42) used for splitting
- [ ] CSV files use UTF-8 encoding
- [ ] Column names follow snake_case convention
- [ ] Preprocessing (strip, normalize, deduplicate) applied
- [ ] Validation checks pass without errors

---

*Last updated: 2025-11-13*

