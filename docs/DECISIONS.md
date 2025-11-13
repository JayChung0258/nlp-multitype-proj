# Project Decisions and Task Redefinition

**NLP Multi-Type Classification Project**

This document records key architectural decisions, the rationale for removing the pair-based approach, and the new task definition.

---

## Table of Contents

1. [Background and Motivation](#background-and-motivation)
2. [Removal of Pair-Based Approach](#removal-of-pair-based-approach)
3. [New Task Definition](#new-task-definition)
4. [Data Pipeline Changes](#data-pipeline-changes)
5. [Evaluation Metric Selection](#evaluation-metric-selection)
6. [Model Selection Rationale](#model-selection-rationale)
7. [Future Considerations](#future-considerations)

---

## Background and Motivation

### Original Project Goals

The original goal of this project was to detect differences between human-written and machine-generated text, including paraphrased variants. The task was framed around understanding authorship and generation patterns in text data.

### Initial Approach: Pair-Based Classification

The initial implementation attempted a **pair-based detection** approach:

- **Method**: Concatenate Type1 (Human Original) and Type3 (Human Paraphrased) sentences, forming artificial "T1+T3" pairs.
- **Task**: Train a binary classifier to distinguish these concatenated pairs from other pairs or types.
- **Rationale**: The hypothesis was that comparing pairs would reveal subtle differences in generation/authorship patterns.

---

## Removal of Pair-Based Approach

### Why Pair-Based Classification Was Invalid

After careful analysis, the pair-based approach was **discarded entirely** for the following reasons:

#### 1. **Semantic Invalidity**

Concatenating two unrelated sentences (e.g., T1 + T3) creates artificial text with no semantic coherence. Real-world applications require classifying **individual sentences**, not arbitrary concatenations.

**Example of the problem:**
- T1: "The weather today is sunny."
- T3: "Machine learning algorithms can be complex."
- Concatenated: "The weather today is sunny. Machine learning algorithms can be complex."

This concatenation has no meaningful semantic relationship and does not reflect any realistic use case.

#### 2. **Task Distortion**

The fundamental question we want to answer is:

> "Given a single sentence, can we determine if it is:
> - Human original (T1)
> - LLM generated (T2)
> - Human paraphrased (T3)
> - LLM paraphrased (T4)"

The pair-based approach does not answer this question. Instead, it creates a different, less useful task that doesn't generalize to real-world scenarios.

#### 3. **Loss of Granularity**

Pair-based classification conflates multiple classes and loses fine-grained distinctions:
- We cannot distinguish T1 from T3 in isolation
- We cannot distinguish T2 from T4 in isolation
- Performance metrics become difficult to interpret

#### 4. **Deployment Infeasibility**

In production:
- Users provide **single sentences** for classification
- There is no meaningful way to generate pairs
- The pair-based model would be unusable

### Decision: Complete Removal

**Action taken:**
- All pair-based code, data generation scripts, and results have been removed
- No references to pair-based methods remain (except in this decision document for context)
- The codebase has been restructured for single-sentence classification

---

## New Task Definition

### Single-Sentence Multi-Class Classification

**Task:** Given a single sentence, predict one of four classes:

| Class | Label | Description |
|-------|-------|-------------|
| **T1** | 0 | Human Original |
| **T2** | 1 | LLM Generated |
| **T3** | 2 | Human Paraphrased |
| **T4** | 3 | LLM Paraphrased |

### Input/Output Specification

- **Input:** One sentence (string)
- **Output:** One class label from {T1, T2, T3, T4} or integer {0, 1, 2, 3}
- **Constraints:** Sentence length should be reasonable (< 4000 characters as guardrail)

### Real-World Applicability

This task is directly applicable to:
- **Content moderation:** Detecting machine-generated text in user submissions
- **Academic integrity:** Identifying AI-assisted writing
- **Data quality assurance:** Filtering synthetic text from human-authored datasets
- **Paraphrase detection:** Distinguishing original from paraphrased content

---

## Data Pipeline Changes

### Family-Based Organization

Data is organized by **families**, where each family contains variants of the same base content:

```
Family ID: f001
  - type1: "The quick brown fox jumps." (Human Original)
  - type2: "A swift brown fox leaps." (LLM Generated)
  - type3: "The fast fox jumped quickly." (Human Paraphrased)
  - type4: "The brown fox made a quick jump." (LLM Paraphrased)
```

### Critical: Family-Aware Splitting

**Problem:** If samples from the same family appear in multiple splits (train/val/test), the model can memorize content rather than learning generalizable patterns → **data leakage**.

**Solution:** All samples from the same family must reside in the same split.

**Implementation:**
- Use `family_id` as the grouping key
- Apply group-aware splitting (e.g., `sklearn.model_selection.GroupShuffleSplit`)
- Split ratios: 70% train / 15% val / 15% test
- Fixed seed (42) for reproducibility

**Validation:**
- Verify no `family_id` appears in multiple splits
- Fail the build if leakage is detected

### Transformation: Family → Single-Sentence Rows

Each family is "unpacked" into individual classification samples:

| family_id | text | label |
|-----------|------|-------|
| f001 | "The quick brown fox jumps." | T1 |
| f001 | "A swift brown fox leaps." | T2 |
| f001 | "The fast fox jumped quickly." | T3 |
| f001 | "The brown fox made a quick jump." | T4 |

All four rows must go to the **same split**.

---

## Evaluation Metric Selection

### Primary Metric: Macro-F1

**Rationale:**
- **Balanced evaluation:** Treats all classes equally (important for potentially imbalanced classes)
- **Standard in multi-class NLP tasks:** Widely used and interpretable
- **Robust to class imbalance:** Unlike accuracy, Macro-F1 is not dominated by majority classes

### Secondary Metrics

To provide a comprehensive evaluation, we also compute:

1. **Accuracy:** Overall correctness (useful baseline reference)
2. **Per-class Precision/Recall/F1:** Detailed performance for each class
3. **AUROC (one-vs-rest):** Calibration and probabilistic quality
4. **Confusion Matrix:** Visual representation of class confusions

### Efficiency Metrics

For practical deployment consideration:
- **Training time** (seconds per epoch)
- **Inference latency** (milliseconds per sample)
- **Parameter count** (model size)
- **Peak VRAM** (GPU memory usage, if available)

These metrics help assess the tradeoff between performance and computational cost.

---

## Model Selection Rationale

### Model Lineup

We benchmark **4-6 models** across two categories:

#### Classical Baselines (TF-IDF + ML)
1. **Logistic Regression:** Simple, interpretable baseline
2. **Linear SVM:** Strong text classification baseline
3. **Random Forest:** Non-linear, ensemble method
4. **XGBoost:** State-of-the-art gradient boosting

**Rationale:** These provide fast, interpretable baselines and can be surprisingly competitive on text tasks.

#### Transformer Models (Fine-tuning)
1. **DistilBERT-base:** Lightweight, fast, good starting point
2. **BERT-base-uncased:** Standard transformer baseline
3. **RoBERTa-base:** Robust BERT variant with improved training
4. **DeBERTa-v3-base:** State-of-the-art architecture
5. **ELECTRA-base** (optional): Efficient discriminative pre-training

**Rationale:** Transformers capture contextual semantics and have proven effective on authorship and generation detection tasks.

### Why Not Novel Architectures?

**Decision:** Focus on benchmarking established models rather than novel research contributions.

**Rationale:**
- **Reproducibility:** Standard models are well-documented and stable
- **Practical value:** Existing models are sufficient for this task
- **AWS portability:** Standard models integrate easily with SageMaker
- **Baseline establishment:** Need solid baselines before exploring novel methods

---

## Future Considerations

### Potential Extensions (Out of Scope for Initial Phase)

1. **Multi-seed robustness testing:** Run experiments with 3+ seeds to measure stability
2. **Perturbation analysis:** Test model robustness to lowercase, punctuation removal, etc.
3. **Length stratification:** Analyze performance across short/medium/long text buckets
4. **Error analysis:** Deep dive into T3 ↔ T4 confusions (expected to be hardest)
5. **Active learning:** Identify which samples are most informative for labeling
6. **Domain adaptation:** Test generalization to different text domains

### AWS Migration

Once local benchmarking is complete, migrate to AWS for:
- Larger-scale experiments
- Hyperparameter tuning (SageMaker Tuner)
- Distributed training (multi-GPU)
- Cost optimization (Spot instances)

See `RUNBOOK_AWS.md` for migration plan.

---

## Summary

| Aspect | Old Approach (Discarded) | New Approach (Current) |
|--------|--------------------------|------------------------|
| **Task** | Binary classification of sentence pairs | 4-class classification of single sentences |
| **Input** | Concatenated sentences (T1+T3, etc.) | Single sentence |
| **Output** | Binary label | One of {T1, T2, T3, T4} |
| **Real-world applicability** | None (artificial task) | Direct deployment for content detection |
| **Data leakage risk** | High (unclear grouping) | Mitigated (family-aware splits) |
| **Evaluation metric** | Unclear interpretation | Macro-F1 with comprehensive metrics |
| **Model selection** | Ad-hoc | Systematic baseline + transformer comparison |

**Conclusion:** The new single-sentence, multi-class approach is theoretically sound, practically useful, and ready for rigorous benchmarking.

---

*Last updated: 2025-11-13*

