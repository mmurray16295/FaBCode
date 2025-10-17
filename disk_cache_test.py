#!/usr/bin/env python3
"""Quick smoke test with disk caching and controlled dataset size."""

import torch
from ultralytics import YOLO

# Quick validation with small subset
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print("\n" + "="*60)
print("Starting disk cache test (1% train, no validation)")
print("="*60 + "\n")

model = YOLO('/workspace/FaBCode/yolo11x.pt')

results = model.train(
    data='/workspace/FaBCode/data/synthetic/data.yaml',
    epochs=1,
    batch=16,
    imgsz=640,
    device=0,
    workers=8,
    project='/workspace/FaBCode/runs/disk_test',
    name='smoke_disk',
    cache='/workspace/FaBCode/cache/dataset_cache',  # Disk cache
    fraction=0.01,  # Only 1% of training data (~875 images)
    val=False,  # DISABLE VALIDATION
    split='train',  # Only use train split, ignore test/val
    patience=50,
    verbose=True,
    plots=True,
)

print("\n" + "="*60)
print("Disk cache test completed!")
print("="*60)
print(f"Results saved to: {results.save_dir}")
