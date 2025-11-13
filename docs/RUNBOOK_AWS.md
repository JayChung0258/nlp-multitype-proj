# AWS Runbook

**NLP Multi-Type Classification Project**

This runbook outlines how to migrate the local project to AWS for scalable training and experimentation.

---

## Table of Contents

1. [Overview](#overview)
2. [AWS Architecture](#aws-architecture)
3. [S3 Data Organization](#s3-data-organization)
4. [SageMaker Training](#sagemaker-training)
5. [EC2 Training (Alternative)](#ec2-training-alternative)
6. [Cost Optimization](#cost-optimization)
7. [IAM Permissions](#iam-permissions)
8. [Migration Checklist](#migration-checklist)

---

## Overview

### Why Migrate to AWS?

- **Scalability:** Train multiple models in parallel
- **GPU Access:** Use powerful GPU instances (P3, P4, G5)
- **Cost Efficiency:** Spot instances reduce costs by 70-90%
- **Reproducibility:** Managed environments with version control
- **Collaboration:** Shared S3 storage and results

### AWS Services Used

1. **S3:** Data storage (raw, processed, results, models)
2. **SageMaker:** Managed training jobs with MLOps features
3. **EC2 (optional):** Direct instance control for custom workflows
4. **CloudWatch:** Logging and monitoring
5. **IAM:** Access control and permissions

---

## AWS Architecture

### High-Level Architecture

```
┌─────────────────┐
│   Local Machine │
│   (Development) │
└────────┬────────┘
         │
         │ aws s3 sync
         ↓
┌─────────────────────────────────────────────┐
│               AWS S3 Bucket                 │
│  s3://my-nlp-bucket/                        │
│  ├── data/                                  │
│  │   ├── raw/                               │
│  │   └── processed/                         │
│  ├── results/                               │
│  │   ├── baseline/                          │
│  │   └── transformer/                       │
│  ├── models/                                │
│  │   ├── checkpoints/                       │
│  │   └── final/                             │
│  ├── configs/                               │
│  └── code/                                  │
└────────┬───────────────────┬────────────────┘
         │                   │
         │                   │
         ↓                   ↓
┌────────────────┐  ┌────────────────┐
│   SageMaker    │  │   EC2 Instance │
│ Training Jobs  │  │  (Alternative) │
└────────┬───────┘  └────────┬───────┘
         │                   │
         └───────┬───────────┘
                 │
                 ↓
┌─────────────────────────────┐
│      CloudWatch Logs        │
└─────────────────────────────┘
```

---

## S3 Data Organization

### Bucket Structure

```
s3://my-nlp-bucket/
├── data/
│   ├── raw/
│   │   └── raw_data.json
│   └── processed/
│       ├── train_4class.csv
│       ├── val_4class.csv
│       └── test_4class.csv
│
├── results/
│   ├── baseline/
│   │   ├── logreg_metrics.json
│   │   ├── linear_svm_metrics.json
│   │   └── baseline_summary.csv
│   └── transformer/
│       ├── bert-base-uncased_metrics.json
│       ├── roberta-base_metrics.json
│       └── transformer_summary.csv
│
├── models/
│   ├── checkpoints/
│   │   ├── bert-base-uncased-epoch1/
│   │   ├── bert-base-uncased-epoch2/
│   │   └── roberta-base-best/
│   └── final/
│       ├── bert-base-uncased/
│       └── roberta-base/
│
├── configs/
│   ├── data_config.yaml
│   ├── models_baseline.yaml
│   ├── models_transformer.yaml
│   └── project.yaml
│
├── code/
│   ├── src/
│   │   ├── data_prep.py
│   │   ├── train_baseline.py
│   │   ├── train_transformer.py
│   │   └── ...
│   └── requirements.txt
│
└── figures/
    ├── data/
    └── results/
```

### Uploading Data to S3

**Initial upload:**

```bash
# Set your bucket name
export S3_BUCKET=s3://my-nlp-bucket

# Upload processed data
aws s3 sync data/processed/ $S3_BUCKET/data/processed/

# Upload configs
aws s3 sync configs/ $S3_BUCKET/configs/

# Upload code
aws s3 sync src/ $S3_BUCKET/code/src/
aws s3 cp requirements.txt $S3_BUCKET/code/
```

**Verify upload:**

```bash
aws s3 ls $S3_BUCKET/data/processed/
```

### Downloading Results from S3

```bash
# Download all results
aws s3 sync $S3_BUCKET/results/ results/

# Download specific model results
aws s3 cp $S3_BUCKET/results/transformer/bert-base-uncased_metrics.json results/
```

---

## SageMaker Training

### Overview

SageMaker provides managed training with:
- Automatic instance provisioning
- Hyperparameter tuning
- Spot instance support
- Experiment tracking
- Model registry

### Training Job Configuration

**Example: SageMaker training script entry point**

```python
# src/train_sagemaker.py

import sagemaker
from sagemaker.pytorch import PyTorch
import os

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
role = os.environ['SAGEMAKER_ROLE']  # IAM role ARN
bucket = os.environ['S3_BUCKET']

# Define training job
estimator = PyTorch(
    entry_point='train_transformer.py',
    source_dir='src/',
    role=role,
    framework_version='2.0.0',
    py_version='py310',
    instance_type='ml.p3.2xlarge',
    instance_count=1,
    output_path=f's3://{bucket}/models/',
    code_location=f's3://{bucket}/code/',
    hyperparameters={
        'model-name': 'bert-base-uncased',
        'num-epochs': 3,
        'learning-rate': 2e-5,
        'batch-size': 16,
    },
    use_spot_instances=True,
    max_wait=86400,  # 24 hours
    max_run=43200,   # 12 hours
)

# Start training
estimator.fit({
    'train': f's3://{bucket}/data/processed/train_4class.csv',
    'val': f's3://{bucket}/data/processed/val_4class.csv',
})
```

### Launching Training Job

**From local machine:**

```bash
# Set environment variables
export SAGEMAKER_ROLE=arn:aws:iam::123456789012:role/SageMakerExecutionRole
export S3_BUCKET=my-nlp-bucket

# Run training script
python src/train_sagemaker.py
```

**Expected output:**

```
Starting training job: nlp-multitype-bert-2025-11-13-10-30-00
...
Training job completed successfully
Model artifacts saved to: s3://my-nlp-bucket/models/nlp-multitype-bert-2025-11-13-10-30-00/output/model.tar.gz
```

### Instance Type Selection

| Instance Type | vCPUs | GPU | RAM | Cost/hr (on-demand) | Cost/hr (spot) | Recommended Use |
|---------------|-------|-----|-----|---------------------|----------------|-----------------|
| ml.p3.2xlarge | 8 | 1x V100 | 61 GB | $3.06 | ~$0.90 | Single transformer model |
| ml.p3.8xlarge | 32 | 4x V100 | 244 GB | $12.24 | ~$3.67 | Parallel experiments |
| ml.g5.xlarge | 4 | 1x A10G | 16 GB | $1.01 | ~$0.30 | Small models, baselines |
| ml.g5.2xlarge | 8 | 1x A10G | 32 GB | $1.21 | ~$0.36 | Medium transformer models |

**Recommendation:** Start with `ml.p3.2xlarge` with Spot instances for cost-effectiveness.

### Monitoring Training

**View logs in CloudWatch:**

```bash
# Get training job name
aws sagemaker list-training-jobs --sort-by CreationTime --sort-order Descending --max-results 1

# Stream logs
aws logs tail /aws/sagemaker/TrainingJobs --follow --log-stream-name-prefix <job-name>
```

**Check job status:**

```bash
aws sagemaker describe-training-job --training-job-name <job-name>
```

---

## EC2 Training (Alternative)

### When to Use EC2 Instead of SageMaker

Use EC2 if you need:
- Full control over environment
- Custom software dependencies
- Interactive development (Jupyter, SSH)
- Long-running experiments (> 5 days)

### EC2 Setup

**Step 1: Launch Instance**

```bash
# Use Deep Learning AMI (Ubuntu)
aws ec2 run-instances \
    --image-id ami-0c2b0d3fb02824d92 \
    --instance-type p3.2xlarge \
    --key-name my-key-pair \
    --security-group-ids sg-xxxxxxxx \
    --iam-instance-profile Name=EC2-S3-Access-Role \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100}}]'
```

**Step 2: Connect to Instance**

```bash
# Get instance public IP
aws ec2 describe-instances --instance-ids i-xxxxxxxxx --query 'Reservations[0].Instances[0].PublicIpAddress'

# SSH into instance
ssh -i my-key-pair.pem ubuntu@<public-ip>
```

**Step 3: Setup Environment**

```bash
# Clone repo
git clone <repo-url>
cd nlp-multitype-proj

# Install dependencies
pip install -r requirements.txt

# Download data from S3
aws s3 sync s3://my-nlp-bucket/data/processed/ data/processed/
```

**Step 4: Run Training**

```bash
# Train baseline models
python src/train_baseline.py --config configs/data_config.yaml

# Train transformers
python src/train_transformer.py --model-name bert-base-uncased

# Upload results
aws s3 sync results/ s3://my-nlp-bucket/results/
```

**Step 5: Terminate Instance (When Done)**

```bash
aws ec2 terminate-instances --instance-ids i-xxxxxxxxx
```

---

## Cost Optimization

### Spot Instances

**Savings:** 70-90% compared to on-demand pricing

**Recommendation:** Always use Spot for training jobs (they can be interrupted but will checkpoint and resume).

**SageMaker Spot configuration:**

```python
estimator = PyTorch(
    ...
    use_spot_instances=True,
    max_wait=86400,  # Maximum time to wait for Spot (24 hours)
    max_run=43200,   # Maximum training time (12 hours)
)
```

**EC2 Spot request:**

```bash
aws ec2 request-spot-instances \
    --spot-price "1.00" \
    --instance-count 1 \
    --type "one-time" \
    --launch-specification file://spot-spec.json
```

### Checkpointing

**Critical for Spot instances:** Save checkpoints frequently to resume if interrupted.

```python
# In training script
def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    # Upload to S3
    s3_client.upload_file(path, bucket, f'checkpoints/{path}')
```

### Multi-Model Training

Train multiple models in parallel to maximize utilization:

```bash
# Launch 4 models in parallel on ml.p3.8xlarge (4 GPUs)
python src/parallel_train.py \
    --models bert-base-uncased roberta-base distilbert-base-uncased deberta-v3-base \
    --instance-type ml.p3.8xlarge
```

### Cost Monitoring

```bash
# Estimate training cost
# ml.p3.2xlarge Spot: ~$0.90/hr
# Expected training time per model: ~2 hours
# Total cost for 4 models: ~$7.20
```

**Set budget alerts in AWS Budgets** to avoid surprises.

---

## IAM Permissions

### SageMaker Execution Role

Required permissions for SageMaker training jobs:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::my-nlp-bucket",
        "arn:aws:s3:::my-nlp-bucket/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchGetImage",
        "ecr:GetDownloadUrlForLayer"
      ],
      "Resource": "*"
    }
  ]
}
```

### EC2 Instance Profile

For EC2 instances to access S3:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::my-nlp-bucket",
        "arn:aws:s3:::my-nlp-bucket/*"
      ]
    }
  ]
}
```

### Creating Roles

**Create SageMaker role:**

```bash
aws iam create-role \
    --role-name SageMakerExecutionRole \
    --assume-role-policy-document file://trust-policy.json

aws iam attach-role-policy \
    --role-name SageMakerExecutionRole \
    --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

aws iam put-role-policy \
    --role-name SageMakerExecutionRole \
    --policy-name S3Access \
    --policy-document file://s3-access-policy.json
```

---

## Migration Checklist

### Pre-Migration

- [ ] Local training and evaluation working correctly
- [ ] All data processed and validated (no leakage)
- [ ] Configs finalized and tested
- [ ] Code pushed to Git repository
- [ ] AWS account setup with billing alerts

### S3 Setup

- [ ] Create S3 bucket: `aws s3 mb s3://my-nlp-bucket`
- [ ] Upload processed data: `aws s3 sync data/processed/ s3://my-nlp-bucket/data/processed/`
- [ ] Upload configs: `aws s3 sync configs/ s3://my-nlp-bucket/configs/`
- [ ] Upload code: `aws s3 sync src/ s3://my-nlp-bucket/code/src/`
- [ ] Verify uploads: `aws s3 ls s3://my-nlp-bucket/ --recursive`

### IAM Setup

- [ ] Create SageMaker execution role
- [ ] Attach S3 access policy
- [ ] Attach CloudWatch logs policy
- [ ] Verify role ARN: `aws iam get-role --role-name SageMakerExecutionRole`

### Training Job Setup

- [ ] Modify training scripts to read from S3 paths
- [ ] Add checkpointing to S3
- [ ] Test with small model on cheap instance (ml.g5.xlarge)
- [ ] Verify metrics are uploaded to S3

### Production Training

- [ ] Launch baseline models (cheap, fast)
- [ ] Launch transformer models with Spot instances
- [ ] Monitor CloudWatch logs
- [ ] Download results and analyze

### Post-Training

- [ ] Download all results: `aws s3 sync s3://my-nlp-bucket/results/ results/`
- [ ] Generate comparison plots
- [ ] Document findings
- [ ] Terminate any running instances
- [ ] Archive S3 data if needed

---

## Quick Reference

### Common AWS CLI Commands

```bash
# List S3 contents
aws s3 ls s3://my-nlp-bucket/ --recursive

# Upload file
aws s3 cp local-file.csv s3://my-nlp-bucket/data/

# Download file
aws s3 cp s3://my-nlp-bucket/results/metrics.json ./

# Sync directories
aws s3 sync local-dir/ s3://my-nlp-bucket/remote-dir/

# List SageMaker jobs
aws sagemaker list-training-jobs --sort-by CreationTime --sort-order Descending

# Describe training job
aws sagemaker describe-training-job --training-job-name <job-name>

# Stop training job
aws sagemaker stop-training-job --training-job-name <job-name>

# List EC2 instances
aws ec2 describe-instances --filters "Name=instance-state-name,Values=running"
```

---

*Last updated: 2025-11-13*

