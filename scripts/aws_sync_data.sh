#!/usr/bin/env bash
#
# Sync processed data between local machine and AWS EC2 instance
#
# Usage:
#   Upload (local → EC2):
#     ./scripts/aws_sync_data.sh upload ubuntu@<EC2_PUBLIC_IP> ~/.ssh/my-key.pem
#
#   Download (EC2 → local):
#     ./scripts/aws_sync_data.sh download ubuntu@<EC2_PUBLIC_IP> ~/.ssh/my-key.pem
#
# Arguments:
#   $1 - Direction: 'upload' or 'download'
#   $2 - EC2 host (format: ubuntu@EC2_PUBLIC_IP)
#   $3 - Path to SSH private key (.pem file)
#
# Example:
#   ./scripts/aws_sync_data.sh upload ubuntu@54.123.45.67 ~/.ssh/my-ec2-key.pem

set -e  # Exit on error

# ============================================================
# Parse arguments
# ============================================================
DIRECTION="$1"
EC2_HOST="$2"
SSH_KEY="$3"

if [ -z "$DIRECTION" ] || [ -z "$EC2_HOST" ] || [ -z "$SSH_KEY" ]; then
    echo "Error: Missing required arguments"
    echo ""
    echo "Usage:"
    echo "  Upload:   $0 upload ubuntu@<EC2_IP> ~/.ssh/key.pem"
    echo "  Download: $0 download ubuntu@<EC2_IP> ~/.ssh/key.pem"
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
echo "AWS Data Sync: $DIRECTION"
echo "========================================================================"
echo "  EC2 Host: $EC2_HOST"
echo "  SSH Key:  $SSH_KEY"
echo ""

# ============================================================
# Sync data based on direction
# ============================================================

EC2_PROJECT_PATH="~/projects/nlp-multitype-proj"

if [ "$DIRECTION" = "upload" ]; then
    # Upload: local → EC2
    echo "Uploading processed data to EC2..."
    echo "  Source: ./data/processed/"
    echo "  Destination: $EC2_HOST:$EC2_PROJECT_PATH/data/"
    echo ""
    
    # Create remote directory if it doesn't exist
    ssh -i "$SSH_KEY" "$EC2_HOST" "mkdir -p $EC2_PROJECT_PATH/data"
    
    # Sync processed data
    scp -i "$SSH_KEY" -r data/processed "$EC2_HOST:$EC2_PROJECT_PATH/data/"
    
    echo ""
    echo "✓ Data uploaded successfully!"
    
elif [ "$DIRECTION" = "download" ]; then
    # Download: EC2 → local
    echo "Downloading processed data from EC2..."
    echo "  Source: $EC2_HOST:$EC2_PROJECT_PATH/data/processed"
    echo "  Destination: ./data_from_ec2/"
    echo ""
    
    # Create local directory
    mkdir -p data_from_ec2
    
    # Sync processed data
    scp -i "$SSH_KEY" -r "$EC2_HOST:$EC2_PROJECT_PATH/data/processed" ./data_from_ec2/
    
    echo ""
    echo "✓ Data downloaded successfully to ./data_from_ec2/processed/"
    
else
    echo "Error: Invalid direction '$DIRECTION'"
    echo "  Must be 'upload' or 'download'"
    exit 1
fi

echo ""
echo "========================================================================"
echo "Sync Complete!"
echo "========================================================================"

