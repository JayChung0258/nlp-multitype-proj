# NLP Multi-Type Text Classification

A systematic benchmarking project for **4-class single-sentence classification** of text origins and generation types.

---

## Overview

This project classifies individual sentences into one of four categories:

| Class | Label | Description |
|-------|-------|-------------|
| **T1** | 0 | Human Original |
| **T2** | 1 | LLM Generated |
| **T3** | 2 | Human Paraphrased |
| **T4** | 3 | LLM Paraphrased |

**Key Features:**
- ✅ **Single-sentence classification** (not pair-based)
- ✅ **Family-aware data splitting** (no leakage)
- ✅ **Rigorous evaluation** (Macro-F1 primary metric)
- ✅ **Baseline + Transformer models** (6 models compared)
- ✅ **AWS-ready** (portable to SageMaker/EC2)

---

## Project Structure

```
nlp-multitype-proj/
├── data/
│   ├── raw/                      # Place raw data here
│   ├── processed/                # Generated train/val/test CSVs
│   └── README.md
│
├── src/
│   ├── constants.py              # Label maps, split ratios, column names
│   ├── schema.py                 # Data contracts and validation schemas
│   ├── data_prep.py              # Data ingestion and family-aware splitting
│   ├── eval_utils.py             # Metric computation and result formatting
│   └── viz_utils.py              # Plotting functions for EDA and results
│
├── notebooks/
│   ├── 00_eda.ipynb              # Exploratory data analysis
│   └── 01_sanity_check.ipynb     # Data validation and config checks
│
├── configs/
│   ├── data_config.yaml          # Paths, splits, preprocessing
│   ├── models_baseline.yaml      # Classical ML model configs
│   ├── models_transformer.yaml   # Transformer model configs
│   └── project.yaml              # Project-level settings
│
├── results/                      # Model outputs (metrics, predictions)
│   ├── baseline/
│   ├── transformer/
│   └── robustness/
│
├── figures/                      # Generated plots and visualizations
│   ├── data/
│   └── results/
│
├── docs/
│   ├── DECISIONS.md              # Why pair-based was removed; task definition
│   ├── DATA_CONTRACT.md          # Formal schemas and validation rules
│   ├── RUNBOOK_LOCAL.md          # How to run locally
│   └── RUNBOOK_AWS.md            # How to deploy on AWS
│
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Quick Start

### 1. Clone and Setup

```bash
# Clone repository
git clone <repository-url>
cd nlp-multitype-proj

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
# Place raw data in data/raw/
cp /path/to/raw_data.json data/raw/

# Run data preparation
python src/data_prep.py --config configs/data_config.yaml
```

**Output:** Creates `train_4class.csv`, `val_4class.csv`, `test_4class.csv` in `data/processed/`

### 3. Exploratory Data Analysis

```bash
# Open EDA notebook
jupyter notebook notebooks/00_eda.ipynb

# Run all cells to:
# - Verify data integrity
# - Analyze class distributions
# - Check for leakage
# - Visualize text length patterns
```

### 4. Train Models (Future)

**Note:** Training scripts are not yet implemented in this phase.

**Planned workflow:**

```bash
# Train baseline models
python src/train_baseline.py --config configs/models_baseline.yaml

# Train transformers
python src/train_transformer.py --model-name bert-base-uncased
```

---

## Architecture Diagram

```
┌──────────────┐
│  Raw Data    │  Family-based (type1, type2, type3, type4)
│  (JSON/CSV)  │
└──────┬───────┘
       │
       ↓
┌──────────────────────────────────┐
│     data_prep.py                 │
│  • Parse families                │
│  • Build single-sentence samples │
│  • Family-aware split (70/15/15)│
│  • Validate (no leakage)         │
└──────┬───────────────────────────┘
       │
       ↓
┌──────────────────────────────────┐
│  Processed CSVs                  │
│  • train_4class.csv              │
│  • val_4class.csv                │
│  • test_4class.csv               │
└──────┬───────────────────────────┘
       │
       ├─────────────┬──────────────┐
       ↓             ↓              ↓
┌─────────────┐ ┌──────────┐ ┌────────────┐
│  TF-IDF +   │ │ Trans-   │ │ Robustness │
│  Classical  │ │ formers  │ │ Analysis   │
│  ML Models  │ │ (BERT,   │ │ (Future)   │
│             │ │ RoBERTa) │ │            │
└─────────────┘ └──────────┘ └────────────┘
       │             │              │
       └──────┬──────┴──────────────┘
              ↓
       ┌─────────────────┐
       │  Evaluation     │
       │  • Macro-F1     │
       │  • Accuracy     │
       │  • Per-class F1 │
       │  • Confusion    │
       └─────────────────┘
```

---

## Key Design Decisions

### No Pair-Based Inputs

**Previous approach (discarded):** Concatenating T1+T3 sentences for binary classification.

**Why removed:**
- Semantically invalid (no meaningful concatenation)
- Not applicable to real-world use cases
- Loses fine-grained class distinctions

**New approach:** Single-sentence, 4-class classification (T1/T2/T3/T4).

See [`docs/DECISIONS.md`](docs/DECISIONS.md) for full rationale.

### Family-Aware Splitting

**Problem:** If samples from the same family appear in multiple splits, the model can memorize content → **data leakage**.

**Solution:** All samples from the same `family_id` must go to the **same split**.

**Implementation:**
- Use `sklearn.model_selection.GroupShuffleSplit`
- Fixed seed (42) for reproducibility
- Validation check fails if leakage detected

See [`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md) for schema details.

### Evaluation Metrics

**Primary metric:** Macro-F1 (treats all classes equally)

**Secondary metrics:**
- Accuracy
- Per-class Precision/Recall/F1
- AUROC (one-vs-rest)
- Confusion matrix

**Efficiency metrics:**
- Training time (seconds)
- Inference latency (ms/sample)
- Parameter count
- Peak VRAM (if available)

---

## Model Lineup

### Classical Baselines (TF-IDF Features)

1. **Logistic Regression** — Simple, interpretable baseline
2. **Linear SVM** — Strong text classification baseline
3. **Random Forest** — Non-linear ensemble method
4. **XGBoost** — Gradient boosting (state-of-the-art classical ML)

### Transformer Models (Fine-tuning)

1. **DistilBERT-base** — Lightweight, fast
2. **BERT-base-uncased** — Standard transformer baseline
3. **RoBERTa-base** — Robust BERT variant
4. **DeBERTa-v3-base** — State-of-the-art architecture
5. **ELECTRA-base** (optional) — Efficient discriminative pre-training

---

## Documentation

| Document | Description |
|----------|-------------|
| [`DECISIONS.md`](docs/DECISIONS.md) | Why pair-based was removed; task definition |
| [`DATA_CONTRACT.md`](docs/DATA_CONTRACT.md) | Formal schemas, validation rules, allowed values |
| [`RUNBOOK_LOCAL.md`](docs/RUNBOOK_LOCAL.md) | Step-by-step local execution guide |
| [`RUNBOOK_AWS.md`](docs/RUNBOOK_AWS.md) | AWS migration (SageMaker, EC2, S3) |

---

## Configuration Files

| File | Purpose |
|------|---------|
| [`data_config.yaml`](configs/data_config.yaml) | Data paths, label maps, split ratios |
| [`models_baseline.yaml`](configs/models_baseline.yaml) | TF-IDF and classical ML hyperparameters |
| [`models_transformer.yaml`](configs/models_transformer.yaml) | Transformer training hyperparameters |
| [`project.yaml`](configs/project.yaml) | Output directories, reporting options, runtime settings |

---

## Data Contract Summary

### Raw Data (Family-Based)

Each row represents a **family** with up to 4 variants:

```json
{
  "family_id": "f001",
  "type1": "Human original text.",
  "type2": "LLM generated text.",
  "type3": "Human paraphrased text.",
  "type4": "LLM paraphrased text."
}
```

### Processed Data (Single-Sentence)

Each row is a **single classification sample**:

| family_id | text | label |
|-----------|------|-------|
| f001 | "Human original text." | T1 |
| f001 | "LLM generated text." | T2 |
| f001 | "Human paraphrased text." | T3 |
| f001 | "LLM paraphrased text." | T4 |

**Critical:** All rows with the same `family_id` must be in the **same split**.

---

## Validation Checks

### Critical (Fail Build)

✅ All labels in {T1, T2, T3, T4} or {0, 1, 2, 3}  
✅ All texts non-empty and ≤ 4000 characters  
✅ No `family_id` overlap between splits (leakage check)  
✅ All required columns present  

### Warning (Log Only)

⚠️ Class imbalance (any class < 10%)  
⚠️ Text length outliers (< 10 or > 2000 chars)  
⚠️ Duplicate texts detected  

---

## Dependencies

### Core Libraries

- **pandas** — Data manipulation
- **numpy** — Numerical operations
- **scikit-learn** — Classical ML models and metrics
- **transformers** — Hugging Face transformers
- **torch** — PyTorch (for transformers)
- **pyyaml** — Config file parsing
- **matplotlib** — Plotting
- **seaborn** — Statistical visualizations

See [`requirements.txt`](requirements.txt) for full list.

---

## AWS Deployment

### S3 Layout

```
s3://my-nlp-bucket/
├── data/processed/       # Train/val/test CSVs
├── results/              # Metrics and predictions
├── models/               # Trained models and checkpoints
├── configs/              # YAML configs
└── code/                 # Source files
```

### SageMaker Training

```python
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point='train_transformer.py',
    role='arn:aws:iam::...:role/SageMakerRole',
    instance_type='ml.p3.2xlarge',
    use_spot_instances=True,
    ...
)

estimator.fit({'train': 's3://my-nlp-bucket/data/processed/train_4class.csv'})
```

See [`docs/RUNBOOK_AWS.md`](docs/RUNBOOK_AWS.md) for complete AWS setup.

---

## Contributing

### Development Workflow

1. Create a feature branch: `git checkout -b feature/new-feature`
2. Make changes and test locally
3. Run data validation: `python src/data_prep.py`
4. Run EDA: `jupyter notebook notebooks/00_eda.ipynb`
5. Commit with clear message: `git commit -m "Add feature X"`
6. Push and create pull request

### Code Standards

- Follow PEP 8 for Python
- Use type hints where reasonable
- Add docstrings to public functions
- Update tests if applicable

---

## Troubleshooting

### Common Issues

**Issue:** `ModuleNotFoundError: No module named 'src'`  
**Solution:** Run from project root: `python src/data_prep.py`

**Issue:** Family leakage detected  
**Solution:** Check raw data for duplicate `family_id` values; verify `group_key: family_id` in config

**Issue:** Class imbalance warning  
**Solution:** Normal if some types are rare; consider class weighting during training

See [`docs/RUNBOOK_LOCAL.md#troubleshooting`](docs/RUNBOOK_LOCAL.md#troubleshooting) for more.

---

## License

[Specify license here]

---

## Contact

[Add contact information or team details]

---

## Changelog

### Version 1.0.0 (2025-11-13)

- Initial project skeleton
- Data preparation pipeline (placeholder)
- Config files and schemas defined
- Documentation complete
- EDA and sanity check notebooks created
- **No training code yet** (to be implemented in next phase)

---

**Status:** 🚧 **Phase 1 Complete** — Skeleton and planning artifacts ready. Training implementation pending.


