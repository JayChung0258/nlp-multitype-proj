#!/usr/bin/env bash
#
# Download results from AWS EC2 instance to local machine
#
# Usage:
#   ./scripts/aws_sync_results.sh ubuntu@<EC2_PUBLIC_IP> ~/.ssh/my-key.pem
#
# Arguments:
#   $1 - EC2 host (format: ubuntu@EC2_PUBLIC_IP)
#   $2 - Path to SSH private key (.pem file)
#
# Example:
#   ./scripts/aws_sync_results.sh ubuntu@54.123.45.67 ~/.ssh/my-ec2-key.pem
#
# This script downloads:
#   - results/baseline/
#   - results/transformer/
#   - reports/ (if exists)
#
# Downloaded files are saved to: ./results_from_ec2/

set -e  # Exit on error

# ============================================================
# Parse arguments
# ============================================================
EC2_HOST="$1"
SSH_KEY="$2"

if [ -z "$EC2_HOST" ] || [ -z "$SSH_KEY" ]; then
    echo "Error: Missing required arguments"
    echo ""
    echo "Usage:"
    echo "  $0 ubuntu@<EC2_IP> ~/.ssh/key.pem"
    exit 1
fi

# Validate SSH key exists
if [ ! -f "$SSH_KEY" ]; then
    echo "Error: SSH key not found: $SSH_KEY"
    exit 1
fi

# Ensure SSH key has correct permissions
chmod 400 "$SSH_KEY"

echo "========================================================================"
echo "AWS Results Sync: Download from EC2"
echo "========================================================================"
echo "  EC2 Host: $EC2_HOST"
echo "  SSH Key:  $SSH_KEY"
echo ""

# ============================================================
# Download results
# ============================================================

EC2_PROJECT_PATH="~/projects/nlp-multitype-proj"
LOCAL_DEST="./results_from_ec2"

echo "Downloading results from EC2..."
echo "  Source: $EC2_HOST:$EC2_PROJECT_PATH/results/"
echo "  Destination: $LOCAL_DEST/"
echo ""

# Create local directory
mkdir -p "$LOCAL_DEST"

# Download results directory
echo "Syncing results/..."
scp -i "$SSH_KEY" -r "$EC2_HOST:$EC2_PROJECT_PATH/results" "$LOCAL_DEST/" || echo "  (results/ may not exist yet)"

# Download reports directory (if exists)
echo ""
echo "Syncing reports/..."
scp -i "$SSH_KEY" -r "$EC2_HOST:$EC2_PROJECT_PATH/reports" "$LOCAL_DEST/" 2>/dev/null || echo "  (reports/ may not exist yet)"

echo ""
echo "========================================================================"
echo "Download Complete!"
echo "========================================================================"
echo ""
echo "Results saved to: $LOCAL_DEST/"
echo ""
echo "Contents:"
ls -lh "$LOCAL_DEST/" 2>/dev/null || echo "  (directory created but may be empty)"

echo ""
echo "View metrics:"
echo "  cat $LOCAL_DEST/results/baseline/logreg_metrics.json"
echo "  cat $LOCAL_DEST/results/transformer/*/metrics.json"
echo ""
echo "View reports:"
echo "  cat $LOCAL_DEST/results/baseline/logreg_report.txt"
echo "  cat $LOCAL_DEST/results/transformer/*/report.txt"
echo ""
echo "========================================================================"

