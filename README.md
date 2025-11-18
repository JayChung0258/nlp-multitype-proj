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
- ✅ **JSONL format** (efficient, portable)
- ✅ **Rigorous evaluation** (Macro-F1 primary metric)
- ✅ **Baseline + Transformer models** implemented and tested
- ✅ **AWS EC2-ready** (automated scripts for GPU training)

---

## Project Structure

```
nlp-multitype-proj/
├── data/
│   ├── raw/                      # Place raw JSON/JSONL data here
│   ├── processed/                # Generated train/val/test JSONL files
│   └── README.md
│
├── src/
│   ├── constants.py              # Label maps, split ratios, field names
│   ├── schema.py                 # Data contracts and validation schemas
│   ├── data_prep.py              # ✅ Data preprocessing pipeline (JSONL)
│   ├── train_baseline.py         # ✅ TF-IDF + Logistic Regression training
│   ├── train_transformer.py      # ✅ Transformer training (multi-model)
│   ├── aws_utils.py              # AWS helper functions (S3, EC2 detection)
│   ├── eval_utils.py             # Metric computation (stubs)
│   └── viz_utils.py              # Plotting functions (stubs)
│
├── notebooks/
│   └── 00_eda.ipynb              # ✅ Complete EDA + validation + report generation
│
├── configs/
│   ├── data_config.yaml          # Paths, splits, preprocessing
│   ├── models_baseline.yaml      # Classical ML model configs
│   ├── models_transformer.yaml   # Transformer model configs
│   ├── project.yaml              # Project-level settings
│   └── aws_config.yaml           # AWS S3/EC2 configuration
│
├── scripts/                      # ✅ Helper scripts
│   ├── train_all_transformers.sh # Train all transformer models
│   ├── aws_ec2_setup.sh          # Automated EC2 environment setup
│   ├── aws_sync_results.sh       # Download results from EC2
│   └── README.md
│
├── results/                      # Model outputs
│   ├── baseline/                 # TF-IDF + LogReg results
│   ├── transformer/              # Transformer model results (per model)
│   └── robustness/
│
├── reports/                      # Generated reports
│   └── eda_report.md             # Auto-generated from EDA notebook
│
├── figures/                      # Generated plots and visualizations
│   ├── data/
│   └── results/
│
├── docs/
│   ├── DECISIONS.md              # Project decisions and task definition
│   ├── DATA_CONTRACT.md          # Data schemas and validation rules
│   ├── DATA_PREPROCESSING.md     # Preprocessing pipeline documentation
│   ├── RUNBOOK_LOCAL.md          # Local development guide
│   └── RUNBOOK_AWS_EC2.md        # AWS EC2 workflow guide
│
├── requirements.txt              # Python dependencies
├── .gitignore                    # Excludes models, data, credentials
├── .gitattributes                # Binary file handling
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
python -m src.data_prep
```

**Output:** Creates `train_4class.jsonl`, `val_4class.jsonl`, `test_4class.jsonl`, and `manifest.json` in `data/processed/`

### 3. Exploratory Data Analysis

```bash
# Open EDA notebook
jupyter notebook notebooks/00_eda.ipynb

# Run all cells to:
# - Load JSONL datasets
# - Verify data integrity
# - Analyze class distributions
# - Check for leakage (family, text, similarity)
# - Validate manifest consistency
# - Generate EDA report (saved to reports/eda_report.md)
```

### 4. Train Models

**Baseline Model (TF-IDF + Logistic Regression):**

```bash
python -m src.train_baseline
```

**Output:** `results/baseline/` with metrics, confusion matrix, and model

**All Transformer Models (Recommended):**

```bash
# Train all 4 models at once
./scripts/train_all_transformers.sh
```

**Individual Transformer Models:**

```bash
# Train specific model
python -m src.train_transformer --model_name distilbert-base-uncased
python -m src.train_transformer --model_name bert-base-uncased
python -m src.train_transformer --model_name roberta-base
python -m src.train_transformer --model_name microsoft/deberta-v3-base
```

**Output:** `results/transformer/<model-name>/` with metrics, confusion matrix, report, and full model

---

## Pipeline Overview

### 1. Data Preprocessing (`src/data_prep.py`)

**Input:** Raw JSON/JSONL files from MRPC, PAWS, HLPC datasets  
**Process:**
- Parse family-based records (each family has type1-type4 variants)
- Normalize text (Unicode NFKC, whitespace cleanup)
- Build single-sentence classification samples
- Perform family-aware split (70% train / 15% val / 15% test)
- Validate data integrity (no leakage, valid labels)

**Output:** 
- `data/processed/train_4class.jsonl` (13,966 samples)
- `data/processed/val_4class.jsonl` (2,996 samples)
- `data/processed/test_4class.jsonl` (2,997 samples)
- `data/processed/manifest.json` (statistics and metadata)

### 2. Model Training

**Baseline (TF-IDF + Logistic Regression):**
- Command: `python -m src.train_baseline`
- Features: TF-IDF with 1-2 grams, 20K max features
- Model: Multinomial logistic regression
- Performance: 51% accuracy, 51% Macro-F1
- Training time: <1 second

**Transformers (BERT family):**
- Command: `python -m src.train_transformer --model_name <model>`
- Supported: DistilBERT, BERT, RoBERTa, DeBERTa, ELECTRA
- Configuration: 3 epochs, 256 max seq length, batch size 16
- Performance: 58-70% Macro-F1 (model dependent)
- Training time: 20-40 minutes per model on GPU

**Batch training:**
- Command: `./scripts/train_all_transformers.sh`
- Trains all 4 transformer models sequentially
- Total time: ~2 hours on GPU, ~6-8 hours on CPU

### 3. Evaluation

**Metrics computed for each model:**
- Accuracy and Macro-F1 (primary)
- Per-class F1 scores (T1, T2, T3, T4)
- Confusion matrix (4x4)
- Training time and model size
- Throughput (samples/sec)

**Output artifacts per model:**
- `metrics.json` — Structured metrics
- `report.txt` — Human-readable summary
- `confusion_matrix.png` — Visualization
- `model/` — Full HuggingFace model (for transformers)

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
| [`DATA_CONTRACT.md`](docs/DATA_CONTRACT.md) | JSONL schemas, validation rules, label mappings |
| [`DATA_PREPROCESSING.md`](docs/DATA_PREPROCESSING.md) | Preprocessing pipeline documentation |
| [`RUNBOOK_LOCAL.md`](docs/RUNBOOK_LOCAL.md) | Step-by-step local execution guide |
| [`RUNBOOK_AWS_EC2.md`](docs/RUNBOOK_AWS_EC2.md) | Complete AWS EC2 workflow guide |

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

Data from three sources: **MRPC**, **PAWS**, **HLPC**. Each record is a family:

```json
{
  "idx": 0,
  "dataset_source": "mrpc",
  "human_original_text(type1)": "Human original text.",
  "llm_generated_text(type2)": "LLM generated text.",
  "human_paraphrased_text(type3)": "Human paraphrased text.",
  "llm_paraphrased_original_text(type4)-prompt-based": "LLM paraphrased text."
}
```

### Processed Data (JSONL Format)

Each line is a **single classification sample**:

```jsonl
{"id": "mrpc_0__T1", "family_id": "mrpc_0", "source": "mrpc", "text": "Human original text.", "label": "T1", "label_id": 0, "text_len_char": 20, "text_len_word": 3}
{"id": "mrpc_0__T2", "family_id": "mrpc_0", "source": "mrpc", "text": "LLM generated text.", "label": "T2", "label_id": 1, "text_len_char": 19, "text_len_word": 3}
```

**Critical:** All rows with the same `family_id` must be in the **same split** (prevents leakage).

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

## AWS EC2 Deployment

This project is **AWS EC2-ready** for GPU-accelerated training.

### Quick Start (EC2)

```bash
# 1. Launch g4dn.xlarge GPU instance (AWS Console)
# 2. SSH into instance
ssh -i ~/.ssh/my-key.pem ubuntu@<EC2_IP>

# 3. Clone and setup
git clone https://github.com/<USER>/nlp-multitype-proj.git
cd nlp-multitype-proj
./scripts/aws_ec2_setup.sh

# 4. Train all models
./scripts/train_all_transformers.sh

# 5. Download results (from local machine)
./scripts/aws_sync_results.sh ubuntu@<EC2_IP> ~/.ssh/my-key.pem

# 6. Stop instance
aws ec2 stop-instances --instance-ids i-xxxxx
```

### Available Scripts

- **`scripts/train_all_transformers.sh`** — Train all 4 transformer models
- **`scripts/aws_ec2_setup.sh`** — Automated EC2 environment setup  
- **`scripts/aws_sync_results.sh`** — Download results from EC2

### Cost Estimate

**Training all 4 transformer models on g4dn.xlarge:**
- Time: ~2 hours
- Cost (Spot): **~$0.32**
- Cost (On-Demand): **~$1.05**

See [`docs/RUNBOOK_AWS_EC2.md`](docs/RUNBOOK_AWS_EC2.md) for complete EC2 workflow guide.


---

## Results

### Baseline Model (TF-IDF + Logistic Regression)

| Metric | Value |
|--------|-------|
| Accuracy | 51.22% |
| Macro-F1 | 50.97% |
| Training Time | 0.51 seconds |
| Model Size | 1.4 MB |

### Transformer Models (DistilBERT - 3 epochs)

| Metric | Value |
|--------|-------|
| Accuracy | 59.23% |
| Macro-F1 | 57.86% |
| Training Time | 55 minutes (CPU/MPS) |
| Model Size | 256 MB |
| Parameters | 67M |

**F1 Scores by Class:**
- T1 (Human Original): 0.49
- T2 (LLM Generated): **0.77** ⭐
- T3 (Human Paraphrased): 0.35
- T4 (LLM Paraphrased): **0.70** ⭐

**Key Finding:** Models perform better at detecting LLM-generated text (T2, T4) vs human-written text (T1, T3).

---

## License

MIT

---




