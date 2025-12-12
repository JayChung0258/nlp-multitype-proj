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

LOCAL_DEST="./results_from_ec2"

echo "Downloading results from EC2..."
echo "  Destination: $LOCAL_DEST/"
echo ""

# Create local directory
mkdir -p "$LOCAL_DEST"

#
# Project paths on EC2 (we try multiple to support both Ubuntu + Amazon Linux)
# - Amazon Linux default user: ec2-user -> /home/ec2-user/nlp-multitype-proj
# - Our Ubuntu user-data script: /home/ubuntu/nlp-multitype-proj
# - Older docs used: ~/projects/nlp-multitype-proj
#
EC2_PROJECT_PATH_CANDIDATES=(
    "/home/ec2-user/nlp-multitype-proj"
    "/home/ubuntu/nlp-multitype-proj"
    "~/projects/nlp-multitype-proj"
)

download_dir() {
    local remote_subdir="$1"  # e.g., results or reports
    local ok=1

    for base in "${EC2_PROJECT_PATH_CANDIDATES[@]}"; do
        if scp -i "$SSH_KEY" -r "$EC2_HOST:$base/$remote_subdir" "$LOCAL_DEST/" 2>/dev/null; then
            echo "  ✓ Downloaded from: $base/$remote_subdir"
            ok=0
            break
        fi
    done

    return $ok
}

# Download results directory
echo "Syncing results/..."
if ! download_dir "results"; then
    echo "  (results/ not found in any known project path on EC2)"
fi

# Download reports directory (if exists)
echo ""
echo "Syncing reports/..."
if ! download_dir "reports"; then
    echo "  (reports/ not found in any known project path on EC2)"
fi

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
echo "  cat $LOCAL_DEST/results/transformer_*/*/metrics.json"
echo ""
echo "View reports:"
echo "  cat $LOCAL_DEST/results/baseline/logreg_report.txt"
echo "  cat $LOCAL_DEST/results/transformer_*/*/report.txt"
echo ""
echo "========================================================================"

