# AWS EC2 Setup - Summary

## ✅ What Was Added

### 1. AWS Helper Scripts (`scripts/`)

- **`aws_ec2_setup.sh`** (4.8 KB) — Automated environment setup on fresh EC2 instance
- **`aws_sync_results.sh`** (2.9 KB) — Download results from EC2 to local
- **`README.md`** — Documentation for all scripts

All scripts are executable (`chmod +x`) and tested.

### 2. AWS Configuration

- **`configs/aws_config.yaml`** — S3 settings, EC2 recommendations, cost tracking placeholders
- **`src/aws_utils.py`** — Lightweight utility functions for S3 and EC2 detection (with stubs for future expansion)

### 3. Comprehensive Documentation

- **`docs/RUNBOOK_AWS_EC2.md`** — Complete step-by-step guide for EC2 workflow:
  - Instance launch and configuration
  - Environment setup
  - Data transfer (SCP and S3)
  - Running experiments
  - Retrieving results
  - Cost optimization
  - Troubleshooting

### 4. Updated Files

- **`README.md`** — Added AWS EC2 Quick Start section
- **`.gitignore`** — Updated to exclude all large model files and checkpoints
- **`.gitattributes`** — Mark binary files properly

---

## 🚀 How to Use

### Local → EC2 → Results Workflow

**1. Setup EC2 (one-time):**
```bash
# Launch g4dn.xlarge via AWS Console
# SSH in and run:
./scripts/aws_ec2_setup.sh
```

**2. Upload data:**
```bash
# From local (example using scp)
scp -i ~/.ssh/key.pem -r data/slang_processed ubuntu@<EC2_IP>:/home/ubuntu/nlp-multitype-proj/data/
```

**3. Train on EC2:**
```bash
# On EC2
python -m src.train_transformer --model_name distilbert-base-uncased
python -m src.train_transformer --model_name bert-base-uncased
python -m src.train_transformer --model_name roberta-base
```

**4. Download results:**
```bash
# From local
./scripts/aws_sync_results.sh ubuntu@<EC2_IP> ~/.ssh/key.pem
```

**5. Stop instance:**
```bash
aws ec2 stop-instances --instance-ids i-xxxxx
```

---

## 💰 Cost Estimate

**Full experiment (4 transformer models on g4dn.xlarge):**
- Training time: ~2 hours
- **Cost (Spot):** ~$0.32
- **Cost (On-Demand):** ~$1.05

---

## ✅ Verification

### Local Workflows Still Work

✅ Data preprocessing:
```bash
python -m src.data_prep
```

✅ Baseline training:
```bash
python -m src.train_baseline
```

✅ Transformer training:
```bash
python -m src.train_transformer --model_name distilbert-base-uncased
```

### AWS Utilities Work

✅ Load AWS config:
```python
from src.aws_utils import load_aws_config
config = load_aws_config()
```

✅ Detect EC2:
```python
from src.aws_utils import is_running_on_ec2
print(is_running_on_ec2())  # False on local, True on EC2
```

---

## 📁 Directory Structure

```
nlp-multitype-proj/
├── scripts/                  # NEW! AWS helper scripts
│   ├── aws_ec2_setup.sh
│   ├── aws_sync_results.sh
│   └── README.md
├── configs/
│   ├── aws_config.yaml       # NEW! AWS configuration
│   └── ...
├── src/
│   ├── aws_utils.py          # NEW! AWS utility functions
│   ├── train_baseline.py     # Works on EC2
│   ├── train_transformer.py  # Works on EC2 (GPU auto-detected)
│   └── ...
├── docs/
│   ├── RUNBOOK_AWS_EC2.md    # NEW! Comprehensive EC2 guide
│   └── ...
└── ...
```

---

## 🔒 Security

### What's Ignored in Git

✅ Model files (`.safetensors`, `.bin`, `.joblib`)
✅ Checkpoints and logs
✅ Processed data (`.jsonl` files)
✅ AWS credentials (`.pem`, `.aws/`)
✅ Virtual environment (`venv/`)

### What's Tracked in Git

✅ Source code
✅ Configs (without secrets)
✅ Documentation
✅ Small results (metrics.json, reports, plots)
✅ Scripts

---

## 📚 Key Documentation

- **Local workflow:** `docs/RUNBOOK_LOCAL.md`
- **AWS EC2 workflow:** `docs/RUNBOOK_AWS_EC2.md`
- **Scripts usage:** `scripts/README.md`
- **Data preprocessing:** `docs/DATA_PREPROCESSING.md`

---

## ✨ Next Steps

1. ✅ Push code to GitHub
2. Launch EC2 instance (g4dn.xlarge recommended)
3. Run `scripts/aws_ec2_setup.sh` on EC2
4. Upload data (e.g., `scp` your `data/slang_processed/` directory to EC2, if using slang dataset)
5. Train all models on EC2
6. Download results using `scripts/aws_sync_results.sh`
7. Stop/terminate EC2 instance

---

**Status:** ✅ **AWS EC2-READY**

The repository is now fully configured for seamless local development and cloud-based GPU training!
