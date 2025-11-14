# AWS EC2 Runbook

**NLP Multi-Type Classification Project**

This runbook provides step-by-step instructions for running experiments on AWS EC2 GPU instances.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [EC2 Instance Setup](#ec2-instance-setup)
3. [First-Time Environment Setup](#first-time-environment-setup)
4. [Data Transfer](#data-transfer)
5. [Running Experiments](#running-experiments)
6. [Retrieving Results](#retrieving-results)
7. [Cost Management](#cost-management)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Local Machine

- ✅ AWS account with EC2 access
- ✅ AWS CLI installed (optional but recommended)
- ✅ SSH client
- ✅ GitHub repository access
- ✅ Processed data ready in `data/processed/`

### Knowledge Required

- Basic AWS EC2 concepts
- SSH and terminal usage
- Git basics

---

## EC2 Instance Setup

### Step 1: Launch EC2 Instance

**Recommended Instance Type:**
- **For DistilBERT, BERT:** `g4dn.xlarge` (1x T4 GPU, 4 vCPUs, 16GB RAM) - ~$0.50/hour
- **For larger models:** `g4dn.2xlarge` (1x T4 GPU, 8 vCPUs, 32GB RAM) - ~$0.75/hour
- **For multiple parallel runs:** `g5.xlarge` (1x A10G GPU) - ~$1.00/hour

**Launch via AWS Console:**

1. Go to EC2 Console → **Launch Instance**

2. **Name:** `nlp-multitype-training`

3. **AMI:** Select **Deep Learning AMI (Ubuntu 20.04)**
   - Search for: "Deep Learning AMI GPU PyTorch"
   - This comes with CUDA, PyTorch, and other ML tools pre-installed

4. **Instance type:** Select `g4dn.xlarge` or `g4dn.2xlarge`

5. **Key pair:**
   - Create new or select existing
   - **Download the `.pem` file** and save it securely (you'll need this for SSH)

6. **Network settings:**
   - Create or select security group with:
     - **SSH (port 22)** allowed from **"My IP"** (for security)

7. **Configure storage:**
   - **Size:** 100 GB (sufficient for models and data)
   - **Type:** gp3 (better performance than gp2)

8. Click **Launch Instance**

9. Wait for instance state to become **Running** (~1-2 minutes)

10. **Note the Public IPv4 address** (you'll need this for SSH)

### Step 2: Configure SSH Key

On your local machine:

```bash
# Move key to secure location
mv ~/Downloads/my-ec2-key.pem ~/.ssh/

# Set correct permissions (required for SSH)
chmod 400 ~/.ssh/my-ec2-key.pem
```

### Step 3: Connect to EC2

```bash
# SSH into your instance
ssh -i ~/.ssh/my-ec2-key.pem ubuntu@<EC2_PUBLIC_IP>

# Example:
ssh -i ~/.ssh/my-ec2-key.pem ubuntu@54.123.45.67
```

**Note:** If using Deep Learning AMI, the username is `ubuntu`. For Amazon Linux, use `ec2-user`.

---

## First-Time Environment Setup

### Option A: Automated Setup (Recommended)

After SSH'ing into EC2, run the setup script:

```bash
# Download and run setup script
wget https://raw.githubusercontent.com/<YOUR_USERNAME>/nlp-multitype-proj/main/scripts/aws_ec2_setup.sh

chmod +x aws_ec2_setup.sh
./aws_ec2_setup.sh
```

This will:
1. Install system dependencies
2. Clone the repository
3. Create virtual environment
4. Install Python packages
5. Verify CUDA availability

### Option B: Manual Setup

```bash
# Update system
sudo apt-get update
sudo apt-get install -y git python3-venv python3-pip

# Create project directory
mkdir -p ~/projects
cd ~/projects

# Clone repository (replace with your GitHub URL)
git clone https://github.com/<YOUR_USERNAME>/nlp-multitype-proj.git
cd nlp-multitype-proj

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Verify CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Expected output:**
```
CUDA available: True
```

---

## Data Transfer

You have three options for getting data to EC2:

### Option 1: SCP from Local (Recommended for Small Data)

**On your local machine:**

```bash
# Navigate to project root
cd /path/to/nlp-multitype-proj

# Upload processed data to EC2
./scripts/aws_sync_data.sh upload ubuntu@<EC2_PUBLIC_IP> ~/.ssh/my-ec2-key.pem
```

**Expected time:** ~30 seconds to 2 minutes (depends on data size)

### Option 2: Regenerate on EC2

If you have raw data in the repo or can download it:

```bash
# On EC2
cd ~/projects/nlp-multitype-proj
source venv/bin/activate

# Run data preprocessing
python -m src.data_prep

# Verify output
ls -lh data/processed/
```

### Option 3: Download from S3 (If Using S3)

```bash
# On EC2
aws s3 sync s3://my-nlp-bucket/data/processed data/processed/
```

**Note:** Requires AWS CLI configured with credentials.

---

## Running Experiments

### Activate Environment

```bash
cd ~/projects/nlp-multitype-proj
source venv/bin/activate
```

### Run Baseline Model

```bash
python -m src.train_baseline
```

**Expected time:** ~1 second (CPU)

### Run Transformer Models

**DistilBERT (fastest):**
```bash
python -m src.train_transformer --model_name distilbert-base-uncased
```

**Expected time:** ~15-20 minutes on g4dn.xlarge

**BERT-base:**
```bash
python -m src.train_transformer --model_name bert-base-uncased
```

**Expected time:** ~25-30 minutes on g4dn.xlarge

**RoBERTa-base:**
```bash
python -m src.train_transformer --model_name roberta-base
```

**Expected time:** ~25-30 minutes on g4dn.xlarge

**DeBERTa-v3-base:**
```bash
python -m src.train_transformer --model_name microsoft/deberta-v3-base
```

**Expected time:** ~30-40 minutes on g4dn.xlarge

**ELECTRA-base (optional):**
```bash
python -m src.train_transformer --model_name google/electra-base-discriminator
```

### Run Multiple Models Sequentially

Create a simple bash script on EC2:

```bash
# Create run_all_models.sh
cat > run_all_models.sh << 'EOF'
#!/bin/bash
set -e

source venv/bin/activate

echo "Running all transformer models..."

python -m src.train_transformer --model_name distilbert-base-uncased
python -m src.train_transformer --model_name bert-base-uncased
python -m src.train_transformer --model_name roberta-base
python -m src.train_transformer --model_name microsoft/deberta-v3-base

echo "All models complete!"
EOF

chmod +x run_all_models.sh
./run_all_models.sh
```

### Monitor Training

**Option 1: Use tmux (recommended for long runs)**

```bash
# Install tmux
sudo apt-get install -y tmux

# Start tmux session
tmux new -s training

# Run training inside tmux
python -m src.train_transformer --model_name bert-base-uncased

# Detach from session: Ctrl+B, then D
# Reattach later: tmux attach -t training
```

**Option 2: Use nohup**

```bash
nohup python -m src.train_transformer --model_name bert-base-uncased > training.log 2>&1 &

# Check progress
tail -f training.log
```

**Option 3: Use screen**

```bash
screen -S training
python -m src.train_transformer --model_name bert-base-uncased
# Detach: Ctrl+A, then D
# Reattach: screen -r training
```

---

## Retrieving Results

### Download Results to Local Machine

**On your local machine:**

```bash
# Navigate to project root
cd /path/to/nlp-multitype-proj

# Download all results
./scripts/aws_sync_results.sh ubuntu@<EC2_PUBLIC_IP> ~/.ssh/my-ec2-key.pem
```

**Results will be in:** `./results_from_ec2/results/`

### Upload Results to S3 (Optional)

**On EC2:**

```bash
# Upload all results
aws s3 sync results/ s3://my-nlp-bucket/nlp-multitype/results/

# Upload specific model
aws s3 sync results/transformer/bert-base-uncased/ \
    s3://my-nlp-bucket/nlp-multitype/results/transformer/bert-base-uncased/
```

---

## Cost Management

### Instance Costs (as of 2024)

| Instance Type | GPU | vCPUs | RAM | On-Demand | Spot (avg) |
|---------------|-----|-------|-----|-----------|------------|
| g4dn.xlarge | 1x T4 | 4 | 16GB | $0.526/hr | ~$0.16/hr |
| g4dn.2xlarge | 1x T4 | 8 | 32GB | $0.752/hr | ~$0.23/hr |
| g5.xlarge | 1x A10G | 4 | 16GB | $1.006/hr | ~$0.30/hr |

### Expected Costs for Full Training

**Scenario:** Train 4 transformer models (DistilBERT, BERT, RoBERTa, DeBERTa)

- **Time:** ~2 hours total on g4dn.xlarge
- **Cost (On-Demand):** ~$1.05
- **Cost (Spot):** ~$0.32

### Best Practices

1. **Use Spot Instances** for 70-90% savings (see Spot section below)

2. **Stop instance when not in use:**
   ```bash
   # From local
   aws ec2 stop-instances --instance-ids <INSTANCE_ID>
   ```

3. **Terminate when done:**
   ```bash
   aws ec2 terminate-instances --instance-ids <INSTANCE_ID>
   ```

4. **Set up billing alerts:**
   - Go to AWS Billing Dashboard
   - Create budget alert for $50/month (or your threshold)

### Stop vs Terminate

- **Stop:** Hibernates instance, keeps EBS volume, can restart later
  - Cost: ~$10/month for 100GB storage
  - Use when: You'll resume experiments soon

- **Terminate:** Permanently deletes instance and attached volumes
  - Cost: $0
  - Use when: Experiments are complete and results are downloaded

### Using Spot Instances

**Launch Spot Instance** (70-90% cheaper):

1. In EC2 Console → Launch Instance
2. Check **"Request Spot instances"**
3. Set **max price** to on-demand price or leave default
4. Continue with normal setup

**Note:** Spot instances can be interrupted, but this is rare for short training runs (< 3 hours).

---

## Troubleshooting

### Issue: SSH Connection Refused

**Cause:** Security group not configured correctly

**Solution:**
```bash
# Check security group allows SSH from your IP
aws ec2 describe-security-groups --group-ids <SG_ID>

# Update security group to allow your IP
aws ec2 authorize-security-group-ingress \
    --group-id <SG_ID> \
    --protocol tcp \
    --port 22 \
    --cidr <YOUR_IP>/32
```

### Issue: CUDA Not Available

**Symptom:**
```
CUDA available: False
```

**Solution:**
1. Verify you selected a GPU instance type (g4dn.*, g5.*, p3.*, p4.*)
2. Check NVIDIA driver:
   ```bash
   nvidia-smi
   ```
3. If `nvidia-smi` fails, reinstall CUDA drivers:
   ```bash
   sudo apt-get install -y nvidia-driver-525
   sudo reboot
   ```

### Issue: Out of Memory (OOM)

**Symptom:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
```bash
# Reduce batch size
python -m src.train_transformer \
    --model_name bert-base-uncased \
    --train_batch_size 8 \  # Reduced from 16
    --eval_batch_size 16     # Reduced from 32

# Or reduce sequence length
python -m src.train_transformer \
    --model_name bert-base-uncased \
    --max_seq_length 128  # Reduced from 256
```

### Issue: Instance Stops Unexpectedly (Spot)

**Cause:** Spot instance interrupted by AWS

**Solution:**
- Training will stop mid-way
- Checkpoints are saved every epoch in `results/transformer/*/checkpoints/`
- Can resume from checkpoint (requires code modification)
- Or use On-Demand instances for critical runs

### Issue: Slow Training

**Possible causes:**

1. **Using CPU instead of GPU:**
   ```bash
   # Verify GPU is being used
   nvidia-smi
   # Should show python process using GPU memory
   ```

2. **Not using mixed precision:**
   - Already enabled automatically if CUDA is available (`fp16=True`)

3. **Swap memory thrashing:**
   ```bash
   # Check memory usage
   free -h
   htop
   ```

---

## Quick Reference

### Common Commands

**Connect to EC2:**
```bash
ssh -i ~/.ssh/my-ec2-key.pem ubuntu@<EC2_IP>
```

**Check GPU:**
```bash
nvidia-smi
watch -n 1 nvidia-smi  # Monitor in real-time
```

**Monitor Training:**
```bash
# Using tmux
tmux attach -t training

# Using tail
tail -f training.log
```

**Download results:**
```bash
# From local machine
cd /path/to/nlp-multitype-proj
./scripts/aws_sync_results.sh ubuntu@<EC2_IP> ~/.ssh/my-ec2-key.pem
```

**Stop instance:**
```bash
aws ec2 stop-instances --instance-ids i-xxxxxxxxxxxxx
```

**Terminate instance:**
```bash
aws ec2 terminate-instances --instance-ids i-xxxxxxxxxxxxx
```

---

## Cost Optimization Tips

### 1. Use Spot Instances

**Savings:** 70-90%

**How:**
- Check "Request Spot instances" when launching
- For 2-hour training jobs, interruption risk is minimal

### 2. Stop (Don't Leave Running)

**Bad:**
- Launch instance → Run experiments → Forget about it
- **Cost:** $0.526/hr × 720 hr/month = **$378/month**

**Good:**
- Launch → Train → Download results → **STOP**
- **Cost:** $0.526/hr × 5 hours = **$2.63**

### 3. Terminate When Completely Done

- **Stop:** Keeps EBS volume (~$10/month)
- **Terminate:** Deletes everything (ensures $0 ongoing cost)

### 4. Run Multiple Models in One Session

Instead of:
- Launch → Train DistilBERT → Stop → Launch → Train BERT → Stop (inefficient)

Do:
- Launch → Train all models sequentially → Stop (efficient)

---

## Example End-to-End Workflow

### Day 1: Setup (One-time)

```bash
# 1. Launch EC2 instance via AWS Console
#    - Type: g4dn.xlarge
#    - AMI: Deep Learning AMI
#    - Storage: 100GB
#    - Download .pem key

# 2. On local: Configure SSH key
mv ~/Downloads/my-ec2-key.pem ~/.ssh/
chmod 400 ~/.ssh/my-ec2-key.pem

# 3. Connect to EC2
ssh -i ~/.ssh/my-ec2-key.pem ubuntu@54.123.45.67

# 4. On EC2: Run setup (inside EC2)
cd ~
git clone https://github.com/<YOUR_USER>/nlp-multitype-proj.git
cd nlp-multitype-proj
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 5. Verify
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
# Should print: CUDA: True

# 6. Exit EC2
exit
```

### Day 2: Upload Data and Train

```bash
# 1. On local: Upload data
cd /path/to/nlp-multitype-proj
./scripts/aws_sync_data.sh upload ubuntu@54.123.45.67 ~/.ssh/my-ec2-key.pem

# 2. SSH back into EC2
ssh -i ~/.ssh/my-ec2-key.pem ubuntu@54.123.45.67

# 3. On EC2: Start training session
cd ~/projects/nlp-multitype-proj
source venv/bin/activate

# Use tmux to avoid connection drops
tmux new -s training

# 4. Run all models
python -m src.train_transformer --model_name distilbert-base-uncased
python -m src.train_transformer --model_name bert-base-uncased
python -m src.train_transformer --model_name roberta-base
python -m src.train_transformer --model_name microsoft/deberta-v3-base

# 5. Detach from tmux (Ctrl+B, then D)
# Close SSH (it's safe now, training continues)
exit
```

### Day 3: Retrieve Results

```bash
# 1. On local: Download results
cd /path/to/nlp-multitype-proj
./scripts/aws_sync_results.sh ubuntu@54.123.45.67 ~/.ssh/my-ec2-key.pem

# 2. Results are in: ./results_from_ec2/results/

# 3. View metrics
cat results_from_ec2/results/transformer/bert-base-uncased/metrics.json
cat results_from_ec2/results/transformer/bert-base-uncased/report.txt

# 4. Stop or terminate EC2
aws ec2 stop-instances --instance-ids i-xxxxxxxxxxxxx
# or
aws ec2 terminate-instances --instance-ids i-xxxxxxxxxxxxx
```

---

## S3 Integration (Optional)

### Setup

```bash
# On EC2: Configure AWS CLI
aws configure
# Enter:
#   AWS Access Key ID: <YOUR_KEY>
#   AWS Secret Access Key: <YOUR_SECRET>
#   Default region: us-west-2
#   Default output format: json
```

### Upload Results to S3

```bash
# After training
aws s3 sync results/ s3://my-nlp-bucket/nlp-multitype/results/

# Upload specific model
aws s3 sync results/transformer/bert-base-uncased/ \
    s3://my-nlp-bucket/nlp-multitype/results/bert-base-uncased/
```

### Download from S3 to Local

```bash
# On local
aws s3 sync s3://my-nlp-bucket/nlp-multitype/results/ ./results/
```

---

## Performance Benchmarks

### Training Time on g4dn.xlarge (1x T4 GPU)

| Model | Epochs | Time | Cost (On-Demand) | Cost (Spot) |
|-------|--------|------|------------------|-------------|
| DistilBERT | 3 | ~15 min | $0.13 | $0.04 |
| BERT-base | 3 | ~25 min | $0.22 | $0.07 |
| RoBERTa-base | 3 | ~25 min | $0.22 | $0.07 |
| DeBERTa-v3 | 3 | ~35 min | $0.31 | $0.09 |

**Total for all 4 models:** ~1.7 hours = **$0.88** (Spot) or **$2.95** (On-Demand)

### Instance Comparison

| Metric | Local (MacBook) | EC2 g4dn.xlarge | EC2 g5.xlarge |
|--------|-----------------|-----------------|---------------|
| GPU | None/MPS | NVIDIA T4 | NVIDIA A10G |
| BERT training | ~2 hours | ~25 min | ~15 min |
| Cost per hour | $0 | $0.526 | $1.006 |

---

## Security Best Practices

1. **SSH Key Security:**
   - Never commit `.pem` files to Git
   - Use `chmod 400` on key files
   - Store in `~/.ssh/` only

2. **Security Groups:**
   - Allow SSH only from your IP (not 0.0.0.0/0)
   - No need to open other ports for this project

3. **AWS Credentials:**
   - Never hardcode in scripts
   - Use AWS CLI configuration or IAM roles
   - Add `.aws/` to `.gitignore` (already done)

4. **Data Privacy:**
   - If data is sensitive, use S3 with encryption
   - Consider VPC for additional isolation

---

## Next Steps

After completing EC2 experiments:

1. **Download results** using sync script
2. **Analyze and compare** model performance locally
3. **Generate comparison plots** (see `src/viz_utils.py`)
4. **Write up findings** in final report
5. **Clean up AWS resources:**
   - Terminate EC2 instance
   - Delete S3 objects (if temporary)
   - Remove unused EBS volumes

---

*Last updated: 2025-11-13*

