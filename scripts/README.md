# Scripts Directory

This directory contains helper scripts for AWS EC2 workflows.

---

## Available Scripts

### 1. `aws_ec2_setup.sh`

**Purpose:** Automated environment setup on a fresh EC2 instance

**Usage:**
```bash
# On EC2 instance
chmod +x aws_ec2_setup.sh
./aws_ec2_setup.sh
```

**What it does:**
- Installs system dependencies (git, python3-venv, pip)
- Creates project directory (`~/projects/`)
- Clones GitHub repository
- Creates Python virtual environment
- Installs all Python packages
- Verifies CUDA availability

**Requirements:**
- Fresh Ubuntu or Deep Learning AMI instance
- Internet connectivity
- Git repository URL (update placeholder in script)

---

### 2. `aws_sync_data.sh`

**Purpose:** Sync processed data between local machine and EC2

**Usage:**

Upload (local → EC2):
```bash
./scripts/aws_sync_data.sh upload ubuntu@54.123.45.67 ~/.ssh/my-key.pem
```

Download (EC2 → local):
```bash
./scripts/aws_sync_data.sh download ubuntu@54.123.45.67 ~/.ssh/my-key.pem
```

**Arguments:**
1. Direction: `upload` or `download`
2. EC2 host: `ubuntu@<PUBLIC_IP>`
3. SSH key path: Path to `.pem` file

**What it syncs:**
- `data/processed/` directory (train/val/test JSONL files + manifest)

**Note:** Run this from your **local machine**, not on EC2.

---

### 3. `aws_sync_results.sh`

**Purpose:** Download experiment results from EC2 to local machine

**Usage:**
```bash
./scripts/aws_sync_results.sh ubuntu@54.123.45.67 ~/.ssh/my-key.pem
```

**Arguments:**
1. EC2 host: `ubuntu@<PUBLIC_IP>`
2. SSH key path: Path to `.pem` file

**What it downloads:**
- `results/baseline/` — Baseline model results
- `results/transformer/` — Transformer model results
- `reports/` — Generated reports (if any)

**Output location:** `./results_from_ec2/`

**Note:** Run this from your **local machine**, not on EC2.

---

## Quick Start: EC2 Workflow

### 1. Setup EC2 (One-time)

```bash
# Launch g4dn.xlarge instance via AWS Console
# SSH into instance
ssh -i ~/.ssh/my-key.pem ubuntu@<EC2_IP>

# Run setup script
wget https://raw.githubusercontent.com/<USER>/nlp-multitype-proj/main/scripts/aws_ec2_setup.sh
chmod +x aws_ec2_setup.sh
./aws_ec2_setup.sh
```

### 2. Upload Data

```bash
# On local machine
cd /path/to/nlp-multitype-proj
./scripts/aws_sync_data.sh upload ubuntu@<EC2_IP> ~/.ssh/my-key.pem
```

### 3. Run Training

```bash
# On EC2
cd ~/projects/nlp-multitype-proj
source venv/bin/activate
python -m src.train_transformer --model_name distilbert-base-uncased
```

### 4. Download Results

```bash
# On local machine
./scripts/aws_sync_results.sh ubuntu@<EC2_IP> ~/.ssh/my-key.pem
```

### 5. Stop Instance

```bash
# Via AWS CLI
aws ec2 stop-instances --instance-ids i-xxxxx

# Or via Console
# EC2 → Instances → Select instance → Instance state → Stop
```

---

## Prerequisites

### Local Machine

- Bash shell (Linux, macOS, WSL on Windows)
- SSH client
- Git
- Processed data in `data/processed/`

### AWS

- EC2 instance (GPU recommended: g4dn.xlarge)
- SSH key pair (.pem file)
- Security group allowing SSH from your IP

---

## Security Notes

- **Never commit `.pem` files** to Git (already in `.gitignore`)
- **Set correct permissions:** `chmod 400 ~/.ssh/my-key.pem`
- **Replace placeholders:** Update `<GITHUB_REPO_URL>` in `aws_ec2_setup.sh`
- **Restrict SSH:** Configure security group to allow SSH only from your IP

---

## Troubleshooting

**Script fails with "Permission denied":**
```bash
chmod +x scripts/*.sh
```

**SCP fails with "Permission denied (publickey)":**
- Check key permissions: `chmod 400 ~/.ssh/my-key.pem`
- Verify correct username (usually `ubuntu` for Ubuntu, `ec2-user` for Amazon Linux)
- Check EC2 security group allows SSH from your IP

**Git clone fails with authentication error:**
- For public repos: Use HTTPS URL
- For private repos: Set up SSH key or personal access token

---

## Advanced: S3 Integration

For very large datasets or multi-user access, use S3 instead of SCP:

**Upload to S3 (on local):**
```bash
aws s3 sync data/processed/ s3://my-bucket/data/processed/
```

**Download on EC2:**
```bash
aws s3 sync s3://my-bucket/data/processed/ data/processed/
```

**Upload results to S3 (on EC2):**
```bash
aws s3 sync results/ s3://my-bucket/results/
```

See `configs/aws_config.yaml` for S3 configuration.

---

*Last updated: 2025-11-13*

