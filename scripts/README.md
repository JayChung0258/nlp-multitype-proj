# Scripts Directory

Helper scripts for local and AWS EC2 workflows.

---

## Available Scripts

### 1. `train_all_transformers.sh` ⭐

**Purpose:** Train all transformer models sequentially with unified configuration

**Usage:**
```bash
# From project root (local or EC2)
./scripts/train_all_transformers.sh
```

**Trains these models:**
1. DistilBERT-base-uncased (~20 min on GPU)
2. BERT-base-uncased (~30 min on GPU)
3. RoBERTa-base (~30 min on GPU)
4. DeBERTa-v3-base (~40 min on GPU)

**Default configuration:**
- Max sequence length: 256
- Training epochs: 3
- Train batch size: 16
- Eval batch size: 32
- Learning rate: 2e-5
- Weight decay: 0.01
- Warmup ratio: 0.1
- Random seed: 42

**Customize via environment variables:**
```bash
# Faster training (fewer epochs, smaller sequences)
MAX_SEQ_LENGTH=128 NUM_EPOCHS=2 ./scripts/train_all_transformers.sh

# Smaller batch size for limited memory
TRAIN_BATCH_SIZE=8 EVAL_BATCH_SIZE=16 ./scripts/train_all_transformers.sh
```

**Estimated time:**
- **On CPU/MPS:** ~6-8 hours total
- **On GPU (g4dn.xlarge):** ~1.5-2 hours total
- **On GPU (g5.xlarge):** ~1 hour total

**Output:** Each model's results saved to `results/transformer/<model-slug>/`

**Features:**
- Auto-activates virtual environment
- Shows progress for each model
- Reports total time at completion
- Safe to run unattended (exits on error)

---

### 2. `aws_ec2_setup.sh`

**Purpose:** Automated environment setup on fresh EC2 instance

**Usage:**
```bash
# Method 1: Run directly on EC2
cd ~
git clone https://github.com/<YOUR_USER>/nlp-multitype-proj.git
cd nlp-multitype-proj
chmod +x scripts/aws_ec2_setup.sh
./scripts/aws_ec2_setup.sh

# Method 2: Download and run
wget https://raw.githubusercontent.com/<USER>/nlp-multitype-proj/main/scripts/aws_ec2_setup.sh
chmod +x aws_ec2_setup.sh
./aws_ec2_setup.sh
```

**What it does:**
1. Updates system packages (`apt-get update`)
2. Installs git, python3-venv, python3-pip
3. Creates project directory (`~/projects/`)
4. Clones GitHub repository
5. Creates Python virtual environment
6. Installs all dependencies from `requirements.txt`
7. Verifies CUDA availability

**Requirements:**
- Ubuntu 20.04+ or AWS Deep Learning AMI
- Internet connectivity
- **Update `<GITHUB_REPO_URL>` placeholder** in script before running

**Output:**
- Repository cloned to `~/projects/nlp-multitype-proj/`
- Virtual environment ready at `~/projects/nlp-multitype-proj/venv/`
- All Python packages installed

---

### 3. `aws_sync_results.sh`

**Purpose:** Download experiment results from EC2 to local machine

**Usage:**
```bash
# From local machine (project root)
./scripts/aws_sync_results.sh ubuntu@<EC2_PUBLIC_IP> ~/.ssh/my-key.pem
```

**Arguments:**
1. EC2 host: `ubuntu@<EC2_PUBLIC_IP>` (replace with your instance IP)
2. SSH key: Path to `.pem` file

**What it downloads:**
- `results/baseline/` — Baseline model results
- `results/transformer/` — All transformer model results
- `reports/` — Generated EDA reports (if any)

**Output location:** `./results_from_ec2/`

**Example:**
```bash
./scripts/aws_sync_results.sh ubuntu@54.123.45.67 ~/.ssh/my-ec2-key.pem

# Results will be in:
# results_from_ec2/results/transformer/bert-base-uncased/
# results_from_ec2/results/transformer/roberta-base/
# etc.
```

**Note:** 
- Run from your **local machine**, not on EC2
- Automatically sets correct SSH key permissions (`chmod 400`)

---

## Complete Workflow

### Local Training (No AWS)

```bash
# 1. Setup
git clone <repo-url>
cd nlp-multitype-proj
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Prepare data (if needed)
python -m src.data_prep

# 3. Run EDA
jupyter notebook notebooks/00_eda.ipynb

# 4. Train baseline
python -m src.train_baseline

# 5. Train all transformers
./scripts/train_all_transformers.sh
```

---

### AWS EC2 GPU Training

#### On Local Machine (Preparation)

```bash
# 1. Launch EC2 instance via AWS Console
#    - Type: g4dn.xlarge (GPU)
#    - AMI: Deep Learning AMI (Ubuntu)
#    - Storage: 100GB
#    - Security group: Allow SSH from your IP
#    - Download .pem key

# 2. Set SSH key permissions
chmod 400 ~/.ssh/my-ec2-key.pem

# 3. Update GitHub repo URL in scripts/aws_ec2_setup.sh
# (Replace <GITHUB_REPO_URL> with your actual repo)
```

#### On EC2 Instance

```bash
# 1. SSH into EC2
ssh -i ~/.ssh/my-ec2-key.pem ubuntu@<EC2_PUBLIC_IP>

# 2. Clone and setup (if not using setup script)
git clone https://github.com/<YOUR_USER>/nlp-multitype-proj.git
cd nlp-multitype-proj

# OR run automated setup
./scripts/aws_ec2_setup.sh

# 3. Activate environment
source venv/bin/activate

# 4. Verify CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
# Should print: CUDA: True

# 5. Optional: Use tmux to avoid disconnection
tmux new -s training

# 6. Train all models
./scripts/train_all_transformers.sh

# 7. Detach from tmux (Ctrl+B, then D) and disconnect
# Training continues in background
```

#### Back on Local Machine

```bash
# 1. Download results
./scripts/aws_sync_results.sh ubuntu@<EC2_PUBLIC_IP> ~/.ssh/my-ec2-key.pem

# 2. Results are in: ./results_from_ec2/results/

# 3. Stop or terminate EC2 instance
aws ec2 stop-instances --instance-ids i-xxxxxxxxxxxxx
# or
aws ec2 terminate-instances --instance-ids i-xxxxxxxxxxxxx
```

---

## Script Features

### Safety
- ✅ Exits on error (`set -e`)
- ✅ Idempotent (safe to run multiple times)
- ✅ Clear error messages
- ✅ Validates prerequisites

### Usability
- ✅ Detailed progress messages
- ✅ Time tracking per model and total
- ✅ Environment variable configuration
- ✅ Auto-activates virtual environment
- ✅ Comprehensive usage instructions in comments

### Portability
- ✅ Works on Ubuntu, macOS, WSL
- ✅ No hardcoded paths or IPs
- ✅ Uses placeholders for user-specific values

---

## Configuration via Environment Variables

All scripts support customization without editing:

**Training configuration:**
```bash
MAX_SEQ_LENGTH=128        # Default: 256
NUM_EPOCHS=5              # Default: 3
TRAIN_BATCH_SIZE=8        # Default: 16
EVAL_BATCH_SIZE=16        # Default: 32
LEARNING_RATE=3e-5        # Default: 2e-5
WEIGHT_DECAY=0.02         # Default: 0.01
WARMUP_RATIO=0.15         # Default: 0.1
SEED=123                  # Default: 42

# Example: Quick test run
MAX_SEQ_LENGTH=128 NUM_EPOCHS=1 ./scripts/train_all_transformers.sh
```

---

## Prerequisites

### Local Machine
- Bash shell
- SSH client (for EC2)
- Git
- Python 3.8+

### EC2 Instance (for GPU training)
- Instance type: `g4dn.xlarge` or better
- AMI: Deep Learning AMI (Ubuntu) or Ubuntu 20.04+
- Storage: 100GB minimum
- Security group: Allow SSH (port 22) from your IP
- SSH key pair (`.pem` file)

---

## Troubleshooting

### Script permission denied
```bash
chmod +x scripts/*.sh
```

### Virtual environment not found (train_all_transformers.sh)
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Missing dependencies for DeBERTa
```bash
pip install protobuf sentencepiece
```
*Already included in `requirements.txt` v2.0+*

### Out of memory during training
```bash
# Reduce batch size
TRAIN_BATCH_SIZE=8 EVAL_BATCH_SIZE=16 ./scripts/train_all_transformers.sh

# Or reduce sequence length
MAX_SEQ_LENGTH=128 ./scripts/train_all_transformers.sh
```

### SSH connection to EC2 fails
```bash
# Check security group allows SSH from your IP
# Verify key permissions
chmod 400 ~/.ssh/my-ec2-key.pem

# Test connection
ssh -i ~/.ssh/my-ec2-key.pem ubuntu@<EC2_IP>
```

---

## Notes

**Processed data is in the repo:**
- Data sync scripts are not needed
- Simply clone the repo to get everything
- Approximately 5.7 MB of processed JSONL files

**Model weights NOT in repo:**
- Large model files (`.safetensors`, `.bin`) are `.gitignore`d
- Only metrics, reports, and plots are tracked
- Download trained models from EC2 using `aws_sync_results.sh`

**For very large datasets:**
- Consider using S3 instead of including data in repo
- See `src/aws_utils.py` for S3 helper stubs
- Uncomment S3-related lines in `requirements.txt` if needed

---

*Last updated: 2025-11-13*
