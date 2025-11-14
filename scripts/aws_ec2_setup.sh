#!/usr/bin/env bash
#
# AWS EC2 Setup Script for NLP Multi-Type Classification Project
#
# This script sets up the development environment on a fresh EC2 instance.
# Tested on: Ubuntu 20.04+ and AWS Deep Learning AMI
#
# Usage:
#   1. SSH into your EC2 instance
#   2. wget or copy this script to the instance
#   3. chmod +x aws_ec2_setup.sh
#   4. ./aws_ec2_setup.sh
#
# Or run directly:
#   bash aws_ec2_setup.sh

set -e  # Exit on error

echo "========================================================================"
echo "NLP Multi-Type Classification: EC2 Environment Setup"
echo "========================================================================"

# ============================================================
# Step 1: Update system and install basic tools
# ============================================================
echo ""
echo "[1/6] Updating system packages..."
sudo apt-get update
sudo apt-get install -y git python3-venv python3-pip curl wget

echo "✓ System packages updated"

# ============================================================
# Step 2: Create project directory
# ============================================================
echo ""
echo "[2/6] Setting up project directory..."

PROJECT_ROOT="${HOME}/projects"
PROJECT_DIR="${PROJECT_ROOT}/nlp-multitype-proj"

mkdir -p "${PROJECT_ROOT}"
cd "${PROJECT_ROOT}"

echo "  Project root: ${PROJECT_ROOT}"

# ============================================================
# Step 3: Clone repository
# ============================================================
echo ""
echo "[3/6] Cloning repository..."

# TODO: Replace <GITHUB_REPO_URL> with your actual GitHub repository URL
GITHUB_REPO_URL="<GITHUB_REPO_URL>"

if [ -d "${PROJECT_DIR}" ]; then
    echo "  Repository already exists at ${PROJECT_DIR}"
    echo "  Pulling latest changes..."
    cd "${PROJECT_DIR}"
    git pull
else
    echo "  Cloning from ${GITHUB_REPO_URL}..."
    git clone "${GITHUB_REPO_URL}" nlp-multitype-proj
    cd "${PROJECT_DIR}"
fi

echo "✓ Repository ready at ${PROJECT_DIR}"

# ============================================================
# Step 4: Create Python virtual environment
# ============================================================
echo ""
echo "[4/6] Creating Python virtual environment..."

if [ -d "venv" ]; then
    echo "  Virtual environment already exists"
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
source venv/bin/activate

echo "✓ Virtual environment activated"

# ============================================================
# Step 5: Install Python dependencies
# ============================================================
echo ""
echo "[5/6] Installing Python dependencies..."

pip install --upgrade pip
pip install -r requirements.txt

echo "✓ Python dependencies installed"

# ============================================================
# Step 6: Verify installation
# ============================================================
echo ""
echo "[6/6] Verifying installation..."

echo "  Python version:"
python --version

echo ""
echo "  PyTorch CUDA availability:"
python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'  Device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"

echo ""
echo "  Installed key packages:"
pip list | grep -E "(torch|transformers|pandas|scikit-learn)"

# ============================================================
# Setup complete
# ============================================================
echo ""
echo "========================================================================"
echo "Setup Complete!"
echo "========================================================================"
echo ""
echo "Project location: ${PROJECT_DIR}"
echo ""
echo "Next steps:"
echo "  1. Sync data from local or S3:"
echo "     - From local: Use aws_sync_data.sh (run on your local machine)"
echo "     - From S3: aws s3 sync s3://your-bucket/data/processed data/processed/"
echo ""
echo "  2. Activate virtual environment (if not already active):"
echo "     cd ${PROJECT_DIR}"
echo "     source venv/bin/activate"
echo ""
echo "  3. Run training:"
echo "     python -m src.train_transformer --model_name distilbert-base-uncased"
echo "     python -m src.train_transformer --model_name bert-base-uncased"
echo ""
echo "  4. After experiments, sync results back:"
echo "     - From local: Use aws_sync_results.sh"
echo "     - To S3: aws s3 sync results/ s3://your-bucket/results/"
echo ""
echo "  5. Remember to STOP or TERMINATE the instance when done!"
echo ""
echo "========================================================================"

