"""
AWS utilities for NLP multi-type classification project.

This module provides helper functions for S3 integration and AWS-specific
functionality. These are lightweight stubs that can be extended as needed.

For now, most AWS operations can be done via AWS CLI or boto3 directly.
"""

import os
from pathlib import Path
from typing import Optional
import yaml


# ============================================================
# CONFIGURATION
# ============================================================

def load_aws_config(config_path: str = "configs/aws_config.yaml") -> dict:
    """
    Load AWS configuration from YAML file.
    
    Args:
        config_path: Path to AWS config file
        
    Returns:
        Configuration dictionary
    """
    if not Path(config_path).exists():
        return {"aws": {"use_s3": False}}
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_data_root() -> str:
    """
    Get data root directory, with optional environment variable override.
    
    Returns:
        Base data directory path
    """
    # Check for environment variable override
    data_root = os.environ.get('DATA_ROOT')
    
    if data_root:
        print(f"Using DATA_ROOT from environment: {data_root}")
        return data_root
    
    # Default: current directory
    return "."


# ============================================================
# S3 INTEGRATION (STUBS)
# ============================================================

def upload_results_to_s3(
    local_path: str,
    bucket: str,
    prefix: str,
    region: str = "us-west-2"
) -> bool:
    """
    Upload results to S3 bucket.
    
    TODO: Implement using boto3 when S3 integration is needed.
    
    For now, use AWS CLI directly:
        aws s3 sync <local_path> s3://<bucket>/<prefix>/
    
    Args:
        local_path: Local directory to upload
        bucket: S3 bucket name
        prefix: S3 prefix (folder path)
        region: AWS region
        
    Returns:
        True if successful, False otherwise
    """
    print(f"TODO: Upload {local_path} to s3://{bucket}/{prefix}/")
    print(f"For now, use: aws s3 sync {local_path} s3://{bucket}/{prefix}/ --region {region}")
    return False


def download_data_from_s3(
    bucket: str,
    prefix: str,
    local_path: str,
    region: str = "us-west-2"
) -> bool:
    """
    Download data from S3 bucket.
    
    TODO: Implement using boto3 when S3 integration is needed.
    
    For now, use AWS CLI directly:
        aws s3 sync s3://<bucket>/<prefix>/ <local_path>
    
    Args:
        bucket: S3 bucket name
        prefix: S3 prefix (folder path)
        local_path: Local directory to download to
        region: AWS region
        
    Returns:
        True if successful, False otherwise
    """
    print(f"TODO: Download s3://{bucket}/{prefix}/ to {local_path}")
    print(f"For now, use: aws s3 sync s3://{bucket}/{prefix}/ {local_path} --region {region}")
    return False


def upload_file_to_s3(
    local_file: str,
    bucket: str,
    s3_key: str,
    region: str = "us-west-2"
) -> bool:
    """
    Upload a single file to S3.
    
    TODO: Implement using boto3 when needed.
    
    Args:
        local_file: Local file path
        bucket: S3 bucket name
        s3_key: S3 object key (path in bucket)
        region: AWS region
        
    Returns:
        True if successful, False otherwise
    """
    print(f"TODO: Upload {local_file} to s3://{bucket}/{s3_key}")
    print(f"For now, use: aws s3 cp {local_file} s3://{bucket}/{s3_key} --region {region}")
    return False


# ============================================================
# EC2 UTILITIES
# ============================================================

def is_running_on_ec2() -> bool:
    """
    Detect if code is running on an EC2 instance.
    
    Returns:
        True if running on EC2, False otherwise
    """
    # Check for EC2 metadata service
    # EC2 instances have instance metadata available at this endpoint
    try:
        import requests
        response = requests.get(
            "http://169.254.169.254/latest/meta-data/instance-id",
            timeout=0.5
        )
        return response.status_code == 200
    except:
        return False


def get_ec2_instance_type() -> Optional[str]:
    """
    Get EC2 instance type if running on EC2.
    
    Returns:
        Instance type string (e.g., "g4dn.xlarge") or None
    """
    if not is_running_on_ec2():
        return None
    
    try:
        import requests
        response = requests.get(
            "http://169.254.169.254/latest/meta-data/instance-type",
            timeout=0.5
        )
        if response.status_code == 200:
            return response.text
    except:
        pass
    
    return None


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    print("AWS Utilities - Example Usage")
    print("="*70)
    
    # Load config
    config = load_aws_config()
    print(f"\nAWS Config loaded:")
    print(f"  S3 enabled: {config.get('aws', {}).get('use_s3', False)}")
    print(f"  S3 bucket: {config.get('aws', {}).get('s3_bucket', 'Not configured')}")
    
    # Check environment
    print(f"\nData root: {get_data_root()}")
    print(f"Running on EC2: {is_running_on_ec2()}")
    
    if is_running_on_ec2():
        print(f"Instance type: {get_ec2_instance_type()}")
    
    print("\n" + "="*70)

