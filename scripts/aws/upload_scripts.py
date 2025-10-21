#!/usr/bin/env python3
"""Upload new scripts to S3"""
import boto3
from pathlib import Path

BUCKET = 'fabcode-backup-1760719829'
s3 = boto3.client('s3')
FILES = [
    'train_full_yolo11x.py',
    'monitor_training.sh',
    'smart_aws_downloader.py',
    'disk_cache_test.py',
    'calculate_batch_size.py',
]
print("Uploading new scripts to S3...")
for f in FILES:
    path = Path('/workspace/FaBCode') / f
    if path.exists():
        print(f"  {f}...", end='')
        s3.upload_file(str(path), BUCKET, f)
        print(" ✓")
    else:
        print(f"  {f}... SKIP (not found)")
print("\n✓ Done! Scripts uploaded to S3.")
