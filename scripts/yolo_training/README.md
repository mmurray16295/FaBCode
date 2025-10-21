# YOLO Training Scripts

Scripts for training YOLO models on FaB card detection dataset.

## Directory Structure

```
yolo_training/
├── README.md                   # This file
├── download_yolo11x.py        # Download pretrained YOLO11x weights
├── train_yolo.py              # Generic YOLO training script
├── train_yolo11x.py           # Production training (2641 classes)
├── train_sanity_check.py      # Quick validation test (100 classes)
├── monitor_training.py        # Python training monitor
└── monitor_training.sh        # Shell training monitor
```

## Scripts Overview

### Model Preparation

**download_yolo11x.py**
- Downloads pretrained YOLO11x weights (~110MB)
- Required before first training
- Downloads from Ultralytics servers
- Usage: `python download_yolo11x.py`

### Training Scripts

**train_yolo.py**
- Generic YOLO training script with full control
- Flexible configuration for any YOLO model variant
- Usage:
  ```bash
  python train_yolo.py --data ../../data/synthetic/data.yaml \
                       --model yolo11n.pt \
                       --epochs 100 \
                       --batch 16 \
                       --device 0
  ```

**train_yolo11x.py**
- Production training script for all 2641 FaB card classes
- Optimized for maximum accuracy with YOLO11x model
- Runs for ~48 hours on RTX 5090 (~$35 on RunPod)
- Features:
  - Automatic checkpoint resumption
  - Progress tracking with alerts
  - Detailed metrics logging
  - Auto-saves every epoch
- Usage:
  ```bash
  # New training
  python train_yolo11x.py --epochs 400 --batch 16 --device 0
  
  # Resume from checkpoint
  python train_yolo11x.py --resume --resume-checkpoint runs/detect/train/weights/last.pt
  ```

**train_sanity_check.py**
- Quick validation test with subset of classes
- Tests 100 random classes for ~2 hours (~$1-2 on RunPod)
- Validates synthetic data quality before full training
- Usage:
  ```bash
  python train_sanity_check.py --num-classes 100 --epochs 50 --batch 32
  ```

### Monitoring Scripts

**monitor_training.py**
- Real-time Python-based training monitor
- Features:
  - Progress tracking from results.csv
  - Alert if training stalls
  - Email notifications (optional)
  - Metric summaries
- Usage:
  ```bash
  python monitor_training.py --results-dir runs/detect/fab_2641_yolo11x \
                             --check-interval 60 \
                             --alert-threshold-minutes 30
  ```

**monitor_training.sh**
- Shell-based training monitor
- Displays latest metrics and GPU usage
- Refreshes automatically
- Usage: `./monitor_training.sh`

## Typical Workflow

### 1. Initial Setup
```bash
# Download pretrained weights
python yolo_training/download_yolo11x.py
```

### 2. Sanity Check (Optional but Recommended)
```bash
# Quick test with 100 classes to validate data quality
python yolo_training/train_sanity_check.py --num-classes 100 --epochs 50
```

### 3. Full Production Training
```bash
# Start training all 2641 classes
python yolo_training/train_yolo11x.py --epochs 400 --batch 16 --device 0

# In another terminal, monitor progress
python yolo_training/monitor_training.py --results-dir runs/detect/fab_2641_yolo11x
```

### 4. Resume Training (if interrupted)
```bash
python yolo_training/train_yolo11x.py --resume --resume-checkpoint runs/detect/fab_2641_yolo11x/weights/last.pt
```

## Training Configuration

### Hardware Requirements

**Minimum:**
- GPU: 8GB VRAM (RTX 3060/4060)
- RAM: 16GB
- Storage: 50GB

**Recommended (for production):**
- GPU: 24GB VRAM (RTX 4090/5090)
- RAM: 32GB+
- Storage: 100GB+

### Batch Size Guidelines

| GPU VRAM | Recommended Batch Size |
|----------|------------------------|
| 8GB      | 4-8                    |
| 12GB     | 8-12                   |
| 16GB     | 12-16                  |
| 24GB     | 16-32                  |

### Training Duration Estimates

**On RTX 5090 (24GB):**
- Sanity check (100 classes, 50 epochs): ~2 hours
- Full training (2641 classes, 400 epochs): ~48 hours

**On RTX 4090 (24GB):**
- Sanity check: ~3 hours
- Full training: ~60 hours

**On RTX 3090 (24GB):**
- Sanity check: ~4 hours
- Full training: ~72 hours

## Output Structure

Training results are saved to `runs/detect/`:

```
runs/detect/
└── fab_2641_yolo11x/           # Training run directory
    ├── weights/
    │   ├── best.pt             # Best model weights
    │   └── last.pt             # Latest checkpoint
    ├── results.csv             # Epoch-by-epoch metrics
    ├── results.png             # Training curves
    ├── confusion_matrix.png    # Classification performance
    ├── val_batch0_pred.jpg     # Validation predictions
    └── args.yaml               # Training configuration
```

## Key Metrics

**Training Metrics:**
- `box_loss` - Bounding box regression loss
- `cls_loss` - Classification loss
- `dfl_loss` - Distribution focal loss

**Validation Metrics:**
- `mAP50` - Mean average precision at IoU=0.5
- `mAP50-95` - Mean average precision at IoU=0.5-0.95
- `precision` - Percentage of correct predictions
- `recall` - Percentage of detected cards

**Target Performance:**
- mAP50: >0.95 (excellent)
- mAP50-95: >0.85 (excellent)
- Precision: >0.90
- Recall: >0.90

## Troubleshooting

**Out of memory errors:**
- Reduce batch size: `--batch 8` or `--batch 4`
- Use smaller model: `train_yolo.py --model yolo11n.pt` instead of yolo11x

**Training stalls/crashes:**
- Check GPU temperature and throttling
- Verify data.yaml path is correct
- Ensure dataset has sufficient samples per class
- Check disk space for checkpoint saves

**Poor accuracy:**
- Run sanity check to validate data quality
- Check class balance in dataset
- Increase training epochs
- Verify augmentation settings in data.yaml

**Resume not working:**
- Ensure checkpoint path is correct
- Use `--resume-checkpoint` with full path to last.pt
- Verify checkpoint wasn't corrupted

## Dependencies

These scripts require:
- `ultralytics` - YOLO implementation
- `torch` - PyTorch deep learning framework
- `pandas` - Data analysis (for monitoring)
- `pyyaml` - YAML parsing
- `opencv-python` - Image processing

Install with:
```bash
pip install ultralytics torch pandas pyyaml opencv-python
```

## Notes

- Training is GPU-intensive - use cloud GPU if local GPU unavailable
- Always run sanity check before committing to 48-hour full training
- Monitor GPU memory usage to avoid OOM errors
- Save checkpoints frequently (automatic in train_yolo11x.py)
- Results are deterministic if you set random seed
