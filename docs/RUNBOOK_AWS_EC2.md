# AWS EC2 Runbook

**NLP Multi-Type Classification Project**

Complete step-by-step guide for running GPU-accelerated transformer training on AWS EC2.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [EC2 Instance Setup](#ec2-instance-setup)
4. [Environment Setup](#environment-setup)
5. [Running Experiments](#running-experiments)
6. [Retrieving Results](#retrieving-results)
7. [Cost Management](#cost-management)
8. [Troubleshooting](#troubleshooting)

---

## Overview

### Why Use AWS EC2?

- **Speed:** GPU training is 10-20x faster than CPU
- **Cost-effective:** Pay only for compute time (~$0.32 for all models on Spot)
- **Scalability:** Run multiple experiments in parallel
- **Flexibility:** Full control over environment

### Workflow Summary

```
Local Machine                    AWS EC2 (GPU)                Local Machine
─────────────                    ─────────────                ─────────────
                                                              
1. Push code     ───────────────> 2. Clone repo
   to GitHub                         Setup environment
                                     
                                  3. Train models
                                     (1.5-2 hours)
                                     
4. Download      <─────────────── 5. Results ready
   results                           
   
5. Analyze                        6. Stop/terminate
   Compare                           instance
```

---

## Prerequisites

### Local Machine

- ✅ AWS account with EC2 access
- ✅ GitHub repository (code pushed)
- ✅ SSH client
- ✅ AWS CLI (optional but recommended)

### Knowledge

- Basic AWS EC2 concepts
- SSH and terminal usage
- Git basics

---

## EC2 Instance Setup

### Recommended Instance Types

| Instance | GPU | vCPUs | RAM | Cost/hr (On-Demand) | Cost/hr (Spot) | Use Case |
|----------|-----|-------|-----|---------------------|----------------|----------|
| **g4dn.xlarge** | 1x T4 | 4 | 16GB | $0.526 | ~$0.16 | **Recommended** |
| g4dn.2xlarge | 1x T4 | 8 | 32GB | $0.752 | ~$0.23 | Larger batch sizes |
| g5.xlarge | 1x A10G | 4 | 16GB | $1.006 | ~$0.30 | Faster training |
| g5.2xlarge | 1x A10G | 8 | 32GB | $1.212 | ~$0.36 | Production workloads |

**Recommendation:** Start with **g4dn.xlarge** (best cost/performance ratio)

### Step-by-Step Instance Launch

#### 1. Open EC2 Console

Go to: AWS Console → EC2 → **Launch Instance**

#### 2. Configure Instance

**Name and tags:**
- Name: `nlp-multitype-training`

**Application and OS Images (AMI):**
- Click **Browse more AMIs**
- Search for: "Deep Learning AMI GPU PyTorch"
- Select: **Deep Learning AMI GPU PyTorch 2.0 (Ubuntu 20.04)**
- This AMI includes CUDA, PyTorch, and NVIDIA drivers pre-installed

**Instance type:**
- Select: **g4dn.xlarge**

**Key pair:**
- Create new or select existing key pair
- **Download the `.pem` file** if creating new
- Save it securely (you'll need it for SSH)

**Network settings:**
- Click **Edit**
- **Create security group** or select existing
- Ensure **SSH (port 22)** is allowed
- **Source:** Select **"My IP"** (for security)

**Configure storage:**
- Size: **100 GiB** (sufficient for models and data)
- Volume type: **gp3** (better performance than gp2)
- **Do not** delete on termination (if you want to preserve data)

**Advanced details (optional but recommended):**
- Check **"Request Spot instances"** for 70-90% cost savings
- Leave max price at default (on-demand price)

#### 3. Launch

- Review settings
- Click **Launch instance**
- Wait for state to become **Running** (~1-2 minutes)
- **Copy the Public IPv4 address**

---

## Environment Setup

### Option A: Automated Setup (Recommended)

#### Step 1: Update Script with Your GitHub URL

Before running, update `scripts/aws_ec2_setup.sh`:

```bash
# Open the file and replace:
# GITHUB_REPO_URL="<GITHUB_REPO_URL>"
# 
# With your actual URL:
# GITHUB_REPO_URL="https://github.com/yourusername/nlp-multitype-proj.git"
```

#### Step 2: Secure SSH Key

```bash
# On local machine
mv ~/Downloads/my-ec2-key.pem ~/.ssh/
chmod 400 ~/.ssh/my-ec2-key.pem
```

#### Step 3: Connect and Run Setup

```bash
# SSH into EC2
ssh -i ~/.ssh/my-ec2-key.pem ubuntu@<EC2_PUBLIC_IP>

# Download and run setup script
wget https://raw.githubusercontent.com/<YOUR_USER>/nlp-multitype-proj/main/scripts/aws_ec2_setup.sh
chmod +x aws_ec2_setup.sh
./aws_ec2_setup.sh

# OR if you already cloned the repo manually:
cd ~/projects/nlp-multitype-proj
./scripts/aws_ec2_setup.sh
```

**Expected output:**
```
======================================================================
Setup Complete!
======================================================================
Project location: /home/ubuntu/projects/nlp-multitype-proj
...
CUDA available: True
```

### Option B: Manual Setup

```bash
# SSH into EC2
ssh -i ~/.ssh/my-ec2-key.pem ubuntu@<EC2_PUBLIC_IP>

# Update system
sudo apt-get update
sudo apt-get install -y git python3-venv python3-pip

# Clone repository
mkdir -p ~/projects && cd ~/projects
git clone https://github.com/<YOUR_USER>/nlp-multitype-proj.git
cd nlp-multitype-proj

# Setup Python environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Verify CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
# Should print: CUDA: True

# Verify data
ls -lh data/processed/
# Should show train/val/test JSONL files
```

---

## Running Experiments

### Verify Environment

```bash
cd ~/projects/nlp-multitype-proj
source venv/bin/activate

# Check data
ls data/processed/
# Should see: train_4class.jsonl, val_4class.jsonl, test_4class.jsonl, manifest.json

# Check GPU
nvidia-smi
# Should show GPU information
```

### Train All Models (Recommended)

Use tmux to avoid disconnection issues:

```bash
# Start tmux session
tmux new -s training

# Activate environment
source venv/bin/activate

# Train all transformers
./scripts/train_all_transformers.sh

# Expected output:
# ======================================================================
# [1/4] Training: distilbert-base-uncased
# ======================================================================
# ...
# [2/4] Training: bert-base-uncased
# ...
# etc.

# Detach from tmux: Press Ctrl+B, then D
# You can now safely disconnect from SSH

# To reattach later:
# tmux attach -t training
```

**Estimated time on g4dn.xlarge:**
- DistilBERT: ~20 minutes
- BERT: ~30 minutes
- RoBERTa: ~30 minutes
- DeBERTa: ~40 minutes
- **Total: ~2 hours**

### Train Individual Models

```bash
source venv/bin/activate

# Train specific model
python -m src.train_transformer --model_name distilbert-base-uncased

# With custom settings
python -m src.train_transformer \
    --model_name bert-base-uncased \
    --max_seq_length 128 \
    --train_batch_size 8 \
    --num_train_epochs 5
```

### Monitor Training

**Option 1: Attach to tmux**
```bash
tmux attach -t training
```

**Option 2: Check logs**
```bash
# If using nohup
tail -f training.log

# Check GPU usage
watch -n 1 nvidia-smi
```

**Option 3: Check output files**
```bash
# List completed models
ls -d results/transformer/*/

# View latest metrics
cat results/transformer/*/metrics.json | python3 -m json.tool | grep -A 2 "macro_f1_test"
```

---

## Retrieving Results

### Download from EC2 to Local

**On your local machine:**

```bash
cd /path/to/nlp-multitype-proj

# Download all results
./scripts/aws_sync_results.sh ubuntu@<EC2_PUBLIC_IP> ~/.ssh/my-ec2-key.pem
```

**Results location:** `./results_from_ec2/results/`

### View Results Locally

```bash
# Navigate to downloaded results
cd results_from_ec2/results

# View metrics for all models
cat transformer/*/metrics.json | python3 -m json.tool | grep -E "(model_name|accuracy_test|macro_f1_test)" | head -20

# View detailed reports
cat transformer/distilbert-base-uncased/report.txt
cat transformer/bert-base-uncased/report.txt
cat transformer/roberta-base/report.txt
cat transformer/microsoft-deberta-v3-base/report.txt

# View confusion matrices
open transformer/*/confusion_matrix.png  # macOS
xdg-open transformer/*/confusion_matrix.png  # Linux
```

---

## Cost Management

### Instance Costs (November 2024)

| Instance | On-Demand | Spot (avg) | Savings |
|----------|-----------|------------|---------|
| g4dn.xlarge | $0.526/hr | $0.16/hr | 70% |
| g4dn.2xlarge | $0.752/hr | $0.23/hr | 69% |
| g5.xlarge | $1.006/hr | $0.30/hr | 70% |

### Training Cost Calculator

**Scenario:** Train 4 transformer models (3 epochs each)

| Instance | Time | On-Demand Cost | Spot Cost |
|----------|------|----------------|-----------|
| g4dn.xlarge | 2 hours | $1.05 | $0.32 |
| g5.xlarge | 1 hour | $1.01 | $0.30 |

### Cost Optimization Tips

**1. Use Spot Instances (70-90% savings)**

When launching:
- Check **"Request Spot instances"**
- Set max price to on-demand price
- For 2-hour jobs, interruption risk is minimal

**2. Stop vs Terminate**

**Stop** (pause instance):
```bash
aws ec2 stop-instances --instance-ids i-xxxxxxxxxxxxx
```
- Keeps EBS volume (~$10/month for 100GB)
- Can resume later
- Good for: Multi-day experiments with breaks

**Terminate** (delete instance):
```bash
aws ec2 terminate-instances --instance-ids i-xxxxxxxxxxxxx
```
- Deletes everything (EBS volume too)
- **Cost: $0** ongoing
- Good for: One-time experiments

**3. Set Billing Alerts**

- Go to AWS Billing Dashboard
- Create budget alert
- Set threshold: $50/month (or your limit)
- Get email when approaching limit

**4. Run Multiple Models in One Session**

✅ **Good:** Launch → Train all 4 models → Download results → Terminate  
❌ **Bad:** Launch → Train 1 model → Terminate → Repeat 4 times

### Monitor Costs

```bash
# Check current month's costs
aws ce get-cost-and-usage \
    --time-period Start=2025-11-01,End=2025-11-30 \
    --granularity MONTHLY \
    --metrics BlendedCost
```

---

## Troubleshooting

### CUDA Not Available

**Symptoms:**
```python
import torch
print(torch.cuda.is_available())  # Returns False
```

**Solutions:**

1. **Verify GPU instance type:**
   ```bash
   # Check instance metadata
   curl http://169.254.169.254/latest/meta-data/instance-type
   # Should return: g4dn.xlarge (or other GPU type)
   ```

2. **Check NVIDIA driver:**
   ```bash
   nvidia-smi
   # Should show GPU information
   ```

3. **If nvidia-smi fails:**
   ```bash
   # Reinstall NVIDIA drivers
   sudo apt-get update
   sudo apt-get install -y nvidia-driver-525
   sudo reboot
   ```

### Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions:**

1. **Reduce batch size:**
   ```bash
   TRAIN_BATCH_SIZE=8 EVAL_BATCH_SIZE=16 ./scripts/train_all_transformers.sh
   ```

2. **Reduce sequence length:**
   ```bash
   MAX_SEQ_LENGTH=128 ./scripts/train_all_transformers.sh
   ```

3. **Use larger instance:**
   - Upgrade to `g4dn.2xlarge` (32GB RAM)

### Training Stopped Unexpectedly

**Cause:** SSH connection dropped or Spot instance interrupted

**Prevention:**
- Use `tmux` or `screen` to keep session alive
- Checkpoints are saved every epoch (can resume manually)

**Recovery:**
```bash
# SSH back in
ssh -i ~/.ssh/my-ec2-key.pem ubuntu@<EC2_IP>

# Reattach to tmux
tmux attach -t training

# Check what completed
ls -d results/transformer/*/
```

### Slow Training

**Issue:** Training taking much longer than expected

**Checks:**

1. **Verify GPU is being used:**
   ```bash
   nvidia-smi
   # Should show python process using GPU memory
   ```

2. **Check device in training output:**
   ```
   Device: cuda  # Good!
   Device: cpu   # Bad - not using GPU
   ```

3. **Monitor GPU utilization:**
   ```bash
   watch -n 1 nvidia-smi
   # GPU utilization should be 90-100% during training
   ```

### Permissions Error on Scripts

```bash
chmod +x scripts/*.sh
```

### Module Not Found Errors

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

---

## Advanced Usage

### Custom Training Configuration

```bash
# Shorter sequences for faster training
MAX_SEQ_LENGTH=128 ./scripts/train_all_transformers.sh

# More epochs for better performance
NUM_EPOCHS=5 ./scripts/train_all_transformers.sh

# Lower learning rate
LEARNING_RATE=1e-5 ./scripts/train_all_transformers.sh

# Combine multiple settings
MAX_SEQ_LENGTH=128 NUM_EPOCHS=2 TRAIN_BATCH_SIZE=32 ./scripts/train_all_transformers.sh
```

### Train Specific Models Only

Edit `scripts/train_all_transformers.sh` to comment out unwanted models:

```bash
MODELS=(
    "distilbert-base-uncased"
    "bert-base-uncased"
    # "roberta-base"  # Skip this one
    "microsoft/deberta-v3-base"
)
```

### Run Baseline Model

```bash
# Baseline is fast (~1 second) so run separately
python -m src.train_baseline
```

### Upload Results to S3 (Optional)

```bash
# On EC2: Configure AWS CLI
aws configure
# Enter your credentials

# Upload results
aws s3 sync results/ s3://my-nlp-bucket/nlp-multitype/results/

# Download on local
aws s3 sync s3://my-nlp-bucket/nlp-multitype/results/ ./results/
```

---

## Performance Benchmarks

### Training Time Comparison

**Dataset:** 13,966 training samples, 3 epochs

| Model | Parameters | Local (MPS) | g4dn.xlarge (T4) | g5.xlarge (A10G) |
|-------|------------|-------------|------------------|------------------|
| DistilBERT | 67M | 55 min | ~20 min | ~12 min |
| BERT-base | 110M | 90 min | ~30 min | ~18 min |
| RoBERTa-base | 125M | 145 min | ~30 min | ~18 min |
| DeBERTa-v3 | 183M | 180 min | ~40 min | ~25 min |

### Cost per Model (g4dn.xlarge)

| Model | Time | On-Demand | Spot |
|-------|------|-----------|------|
| DistilBERT | 20 min | $0.18 | $0.05 |
| BERT | 30 min | $0.26 | $0.08 |
| RoBERTa | 30 min | $0.26 | $0.08 |
| DeBERTa | 40 min | $0.35 | $0.11 |
| **All 4** | **2 hours** | **$1.05** | **$0.32** |

---

## Complete End-to-End Example

### Day 1: Setup (15 minutes)

```bash
# === On Local Machine ===

# 1. Ensure code is pushed to GitHub
cd /path/to/nlp-multitype-proj
git add .
git commit -m "Ready for EC2 training"
git push

# 2. Launch EC2 via AWS Console
#    - Instance: g4dn.xlarge
#    - AMI: Deep Learning AMI GPU PyTorch
#    - Storage: 100GB gp3
#    - Security: Allow SSH from My IP
#    - Download .pem key

# 3. Configure SSH key
mv ~/Downloads/my-ec2-key.pem ~/.ssh/
chmod 400 ~/.ssh/my-ec2-key.pem

# 4. Connect to EC2
ssh -i ~/.ssh/my-ec2-key.pem ubuntu@54.123.45.67

# === Now on EC2 ===

# 5. Clone and setup
git clone https://github.com/yourusername/nlp-multitype-proj.git
cd nlp-multitype-proj
./scripts/aws_ec2_setup.sh

# 6. Verify
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
# Output: CUDA: True

ls data/processed/
# Should show JSONL files (already in repo)

# 7. Disconnect
exit
```

### Day 2: Training (2 hours on GPU)

```bash
# === On Local Machine ===

# SSH back into EC2
ssh -i ~/.ssh/my-ec2-key.pem ubuntu@54.123.45.67

# === Now on EC2 ===

cd ~/projects/nlp-multitype-proj
source venv/bin/activate

# Use tmux to prevent disconnection issues
tmux new -s training

# Run all models
./scripts/train_all_transformers.sh

# Detach from tmux: Ctrl+B, then D
# Disconnect from SSH (training continues)
exit
```

### Day 3: Retrieve Results (5 minutes)

```bash
# === On Local Machine ===

# Download results
cd /path/to/nlp-multitype-proj
./scripts/aws_sync_results.sh ubuntu@54.123.45.67 ~/.ssh/my-ec2-key.pem

# View results
cd results_from_ec2/results
cat transformer/bert-base-uncased/metrics.json | python3 -m json.tool

# Compare models
for model in transformer/*/; do
    echo "$model:"
    cat "$model/metrics.json" | python3 -m json.tool | grep -E "(accuracy_test|macro_f1_test)"
done

# Terminate EC2 instance (or stop if you'll use again soon)
aws ec2 terminate-instances --instance-ids i-xxxxxxxxxxxxx

# Done!
```

---

## Quick Reference Commands

### EC2 Management

```bash
# List instances
aws ec2 describe-instances --query 'Reservations[*].Instances[*].[InstanceId,State.Name,PublicIpAddress,InstanceType]' --output table

# Stop instance
aws ec2 stop-instances --instance-ids i-xxxxxxxxxxxxx

# Start stopped instance
aws ec2 start-instances --instance-ids i-xxxxxxxxxxxxx

# Terminate instance
aws ec2 terminate-instances --instance-ids i-xxxxxxxxxxxxx

# Get public IP
aws ec2 describe-instances --instance-ids i-xxxxxxxxxxxxx --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
```

### Training Commands

```bash
# Single model
python -m src.train_transformer --model_name bert-base-uncased

# All models
./scripts/train_all_transformers.sh

# Baseline
python -m src.train_baseline
```

### Monitoring

```bash
# GPU usage
nvidia-smi
watch -n 1 nvidia-smi

# Disk space
df -h

# Memory
free -h

# Running processes
htop
```

---

## Best Practices

### 1. Always Use tmux or screen

**Why:** Prevents training interruption if SSH connection drops

```bash
# Start session
tmux new -s training

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t training
```

### 2. Verify CUDA Before Training

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
```

### 3. Download Results Immediately

Don't leave results only on EC2:
```bash
./scripts/aws_sync_results.sh ubuntu@<EC2_IP> ~/.ssh/my-ec2-key.pem
```

### 4. Stop/Terminate When Done

**Check costs before leaving:**
```bash
# This instance costs $0.526/hour (On-Demand)
# If you forget to stop it for a week: $88.37!
```

**Always stop or terminate:**
```bash
aws ec2 stop-instances --instance-ids i-xxxxx
```

### 5. Set Up Cost Alerts

- AWS Console → Billing → Budgets
- Create monthly budget ($50 or your threshold)
- Set alert at 80%

---

## Checklist

### Before Training

- [ ] EC2 instance launched (g4dn.xlarge recommended)
- [ ] SSH key configured (`chmod 400`)
- [ ] Environment setup complete (`aws_ec2_setup.sh` run)
- [ ] CUDA verified (prints `True`)
- [ ] Data present (`ls data/processed/` shows JSONL files)
- [ ] Virtual environment activated

### During Training

- [ ] Using tmux/screen (to survive disconnections)
- [ ] GPU is being used (`nvidia-smi` shows python process)
- [ ] Monitoring progress (`tmux attach` or check logs)

### After Training

- [ ] Results downloaded to local (`aws_sync_results.sh`)
- [ ] Results backed up (git commit or S3 upload)
- [ ] EC2 instance stopped or terminated
- [ ] Billing dashboard checked (no unexpected charges)

---

## Security Notes

- **Never commit `.pem` files** (already in `.gitignore`)
- **Set restrictive permissions:** `chmod 400 ~/.ssh/my-key.pem`
- **Limit SSH access:** Security group should allow SSH only from your IP
- **No hardcoded credentials:** Use environment variables or AWS IAM roles
- **Rotate keys:** Change SSH keys periodically

---

*Last updated: 2025-11-13*
