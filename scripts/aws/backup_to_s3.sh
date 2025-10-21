#!/bin/bash
# Backup Critical Training Data to AWS S3
# Run this when AWS services recover

set -e

echo "========================================="
echo "FaB Detector Phase 3 - AWS Backup Script"
echo "========================================="
echo ""

# Configuration
S3_BUCKET="s3://fabcode-backup-1760719829"
REGION="us-east-2"

echo "⚠️  WARNING: This will upload ~730GB of data to S3"
echo "Estimated time: 2-4 hours depending on bandwidth"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    exit 1
fi

echo ""
echo "Starting backup..."
echo ""

# 1. Backup training data (largest)
echo "[1/5] Uploading training data (728GB)..."
aws s3 sync data/synthetic/ ${S3_BUCKET}/phase3_training_data/ \
    --region ${REGION} \
    --exclude "*.pyc" \
    --exclude "__pycache__/*" \
    --storage-class INTELLIGENT_TIERING

# 2. Backup all model checkpoints
echo "[2/5] Uploading model checkpoints..."
aws s3 sync runs/train/yolo11x_2641classes_20251018_073329/weights/ \
    ${S3_BUCKET}/phase3_checkpoints/ \
    --region ${REGION}

# 3. Backup complete runs directory (logs, metrics)
echo "[3/5] Uploading training logs and metrics..."
aws s3 sync runs/ ${S3_BUCKET}/phase3_runs/ \
    --region ${REGION} \
    --exclude "*/weights/*.pt"  # Already backed up separately

# 4. Backup card metadata
echo "[4/5] Uploading card metadata..."
aws s3 cp data/card.json ${S3_BUCKET}/phase3_metadata/card.json --region ${REGION}
aws s3 cp data/card_weights_all_printings.json ${S3_BUCKET}/phase3_metadata/ --region ${REGION}

# 5. Backup scripts and configurations
echo "[5/5] Uploading scripts and configs..."
aws s3 sync scripts/ ${S3_BUCKET}/phase3_scripts/ --region ${REGION}

echo ""
echo "========================================="
echo "✅ Backup Complete!"
echo "========================================="
echo ""
echo "Backed up to: ${S3_BUCKET}"
echo ""
echo "Files backed up:"
echo "  - Training data: phase3_training_data/ (728GB)"
echo "  - Checkpoints: phase3_checkpoints/ (1.4GB)"
echo "  - Training logs: phase3_runs/ (minimal)"
echo "  - Metadata: phase3_metadata/ (20MB)"
echo "  - Scripts: phase3_scripts/"
echo ""
echo "To restore on new instance:"
echo "  aws s3 sync ${S3_BUCKET}/phase3_training_data/ data/synthetic/"
echo "  aws s3 sync ${S3_BUCKET}/phase3_checkpoints/ runs/train/yolo11x_2641classes_20251018_073329/weights/"
echo ""
