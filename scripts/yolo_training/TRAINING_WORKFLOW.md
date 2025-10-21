# YOLO11x Training Workflow

Complete workflow for training YOLO11x on 2641 FaB card classes.

---

## 📋 Quick Start Checklist

### Local Machine (Tonight)
- [ ] Generate 25,000 synthetic images (~7 hours)
- [ ] Compress dataset for upload
- [ ] Review RUNPOD_SETUP.md

### RunPod Setup (Tomorrow Morning)
- [ ] Create RTX 5090 instance
- [ ] Install dependencies
- [ ] Upload dataset
- [ ] Run sanity check (~2 hours, $1.40)

### Full Training (Tomorrow Afternoon)
- [ ] Start YOLO11x training (~48 hours, $35)
- [ ] Monitor progress via TensorBoard
- [ ] Download results when complete

---

## 🎯 Workflow Steps

### Step 1: Generate Dataset (Local - Tonight)

```powershell
cd "c:\VS Code\FaB Code"
python scripts\generate_dataset_25k.py
```

**Expected:**
- Time: ~7 hours
- Output: 25,000 images (22,500 train / 2,500 val)
- Size: ~40-50GB

**While it runs:**
- Review `RUNPOD_SETUP.md`
- Create RunPod account if needed
- Plan RunPod instance timing

### Step 2: Compress Dataset (Local)

```powershell
# After generation completes
cd "c:\VS Code\FaB Code"
tar -czf synthetic_dataset.tar.gz data\synthetic\
```

**Expected:**
- Time: ~10 minutes
- Compressed size: ~20-30GB

### Step 3: Setup RunPod (Cloud - Tomorrow)

Follow detailed instructions in `RUNPOD_SETUP.md`:

1. Create RTX 5090 instance ($0.69/hr)
2. Connect via SSH
3. Install dependencies
4. Upload dataset

### Step 4: Sanity Check (Cloud - 2 hours)

```bash
cd /workspace/FaBCode
python scripts/train_sanity_check.py
```

**Expected:**
- Time: ~2 hours
- Cost: $1.40
- mAP@0.5: >85% (indicates good data quality)

**If mAP < 85%:**
- Check synthetic data augmentations
- Review training plots
- May need to adjust generation parameters

### Step 5: Full Training (Cloud - 48 hours)

```bash
# Start tmux session (important!)
tmux new -s yolo_training

# Run training
cd /workspace/FaBCode
python scripts/train_yolo11x.py

# Detach from tmux: Ctrl+B, then D
# Reattach later: tmux attach -t yolo_training
```

**Expected:**
- Time: ~48 hours (RTX 5090)
- Cost: ~$35
- Final mAP@0.5: 95-97%

### Step 6: Monitor Training (Cloud)

**In another terminal:**

```bash
# Real-time monitoring
python scripts/monitor_training.py

# Or quick status check
python scripts/monitor_training.py --quick-check

# Or TensorBoard
tensorboard --logdir runs/detect --host 0.0.0.0 --port 6006
```

**Access TensorBoard:**
- URL: `https://<pod-id>-6006.proxy.runpod.net`

### Step 7: Download Results (Local)

```powershell
# Download best weights
scp -P <port> root@<pod-id>.runpod.io:/workspace/FaBCode/runs/detect/fab_2641_yolo11x/weights/best.pt .

# Download all results
scp -r -P <port> root@<pod-id>.runpod.io:/workspace/FaBCode/runs/detect/fab_2641_yolo11x .\results\
```

---

## 📊 Expected Timeline

| Day | Task | Time | Cost |
|-----|------|------|------|
| **Day 1 (Tonight)** | Generate 25k images | 7h | $0 |
| **Day 1 (Tonight)** | Compress dataset | 10min | $0 |
| **Day 2 (Morning)** | Setup RunPod | 30min | $0.35 |
| **Day 2 (Morning)** | Upload dataset | 1h | $0.69 |
| **Day 2 (Morning)** | Sanity check | 2h | $1.40 |
| **Day 2 (Afternoon)** | Start full training | - | - |
| **Day 3-4** | Training running | 48h | $33.12 |
| **Day 4** | Download results | 1h | $0.69 |
| **Total** | | **~60h** | **~$36** |

---

## 🎓 Training Scripts Reference

### 1. `generate_dataset_25k.py`

Generates 25,000 synthetic training images.

**Usage:**
```bash
# Default: 25k images, 90/10 split
python scripts/generate_dataset_25k.py

# Custom: 50k images, 85/15 split
python scripts/generate_dataset_25k.py --images 50000 --train-split 0.85
```

**Parameters:**
- `--images`: Total images to generate (default: 25000)
- `--output`: Output directory (default: ../data/synthetic)
- `--train-split`: Training split fraction (default: 0.9)
- `--seed`: Random seed (default: 42)

### 2. `train_sanity_check.py`

Quick validation of synthetic data quality.

**Usage:**
```bash
# Default: 100 classes, 50 epochs
python scripts/train_sanity_check.py

# Custom: 200 classes, 100 epochs
python scripts/train_sanity_check.py --classes 200 --epochs 100
```

**Parameters:**
- `--data`: Path to data.yaml
- `--classes`: Number of classes to test (default: 100)
- `--epochs`: Training epochs (default: 50)
- `--batch`: Batch size (default: 32)

**Success Criteria:**
- mAP@0.5 > 85%
- Loss curves decreasing smoothly
- No overfitting (train/val gap < 10%)

### 3. `train_yolo11x.py`

Full production training on all 2641 classes.

**Usage:**
```bash
# Default: 400 epochs, batch 16
python scripts/train_yolo11x.py

# Custom: 500 epochs, batch 8 (if OOM)
python scripts/train_yolo11x.py --epochs 500 --batch 8

# Resume from checkpoint
python scripts/train_yolo11x.py --resume --checkpoint runs/detect/fab_2641_yolo11x/weights/last.pt
```

**Parameters:**
- `--data`: Path to data.yaml
- `--epochs`: Training epochs (default: 400)
- `--batch`: Batch size (default: 16)
- `--device`: GPU device ID (default: 0)
- `--resume`: Resume from checkpoint
- `--checkpoint`: Checkpoint path

**Hyperparameters (Accuracy-Optimized):**
- Optimizer: AdamW
- Learning rate: 0.001 → 0.00001 (cosine decay)
- Weight decay: 0.0005
- Warmup: 5 epochs
- Close mosaic: Last 10 epochs
- Patience: 50 epochs

### 4. `monitor_training.py`

Real-time training monitoring with alerts.

**Usage:**
```bash
# Continuous monitoring (updates every 60s)
python scripts/monitor_training.py

# Quick status check
python scripts/monitor_training.py --quick-check

# Custom interval (check every 5 minutes)
python scripts/monitor_training.py --interval 300
```

**Parameters:**
- `--results-dir`: Training results directory
- `--interval`: Check interval in seconds (default: 60)
- `--alert-threshold`: Alert if stalled (minutes, default: 30)
- `--quick-check`: Just check current status

**Displays:**
- Current epoch and progress
- Time elapsed and remaining
- Current metrics (mAP, precision, recall, losses)
- ETA for completion
- Warnings if training stalls or metrics are poor

---

## 🔧 Troubleshooting

### Dataset Generation Issues

**Problem:** "Out of memory during generation"
```bash
# Generate in smaller batches
python scripts/generate_dataset_25k.py --images 10000
python scripts/generate_dataset_25k.py --images 10000 --output ../data/synthetic_batch2
# Merge later
```

**Problem:** "Generation is too slow"
- Expected: ~1.07s per image
- Check: Task Manager → GPU usage should be >50%
- Fix: Close other GPU-intensive applications

### Training Issues

**Problem:** "CUDA out of memory"
```bash
# Reduce batch size
python scripts/train_yolo11x.py --batch 8

# Or use smaller model first
python scripts/train_yolo11x.py --batch 16  # YOLO11l instead
```

**Problem:** "Training is stuck/stalled"
```bash
# Check if training is still running
python scripts/monitor_training.py --quick-check

# If stuck, resume from checkpoint
python scripts/train_yolo11x.py --resume --checkpoint runs/detect/fab_2641_yolo11x/weights/last.pt
```

**Problem:** "mAP is not improving"
- Check TensorBoard: Loss curves should be decreasing
- Verify: Data paths in data.yaml are correct
- Try: More epochs or generate more training data

### Upload Issues

**Problem:** "Upload is too slow"
```bash
# Compress more aggressively
tar -czf synthetic_dataset.tar.gz data/synthetic/ --best

# Or use faster compression
tar -cf - data/synthetic/ | pigz > synthetic_dataset.tar.gz
```

**Problem:** "Upload interrupted"
```bash
# Resume with rsync instead of scp
rsync -avz --progress -e "ssh -p <port>" synthetic_dataset.tar.gz root@<pod-id>.runpod.io:/workspace/FaBCode/
```

---

## 📈 Success Metrics

### Sanity Check (100 classes)

✅ **Excellent:** mAP@0.5 > 90%
✅ **Good:** mAP@0.5 > 85%
⚠️ **Acceptable:** mAP@0.5 > 80%
❌ **Poor:** mAP@0.5 < 80%

### Full Training (2641 classes)

✅ **Excellent:** mAP@0.5 > 95%
✅ **Good:** mAP@0.5 > 90%
⚠️ **Acceptable:** mAP@0.5 > 85%
❌ **Needs Work:** mAP@0.5 < 85%

### Real-World Performance

✅ **Production Ready:** >95% accuracy on real cards
✅ **Usable:** >90% accuracy on real cards
⚠️ **Needs Tuning:** 85-90% accuracy
❌ **Not Ready:** <85% accuracy

---

## 💰 Budget Tracking

| Item | Cost | Status |
|------|------|--------|
| Dataset generation | $0 | Local |
| RunPod setup | $1 | ⏳ Pending |
| Sanity check | $1.40 | ⏳ Pending |
| Full training | $35 | ⏳ Pending |
| Buffer/testing | $3 | ⏳ Pending |
| **Subtotal** | **$40.40** | |
| **Total Budget** | **$200** | |
| **Remaining** | **$159.60** | For retraining/YOLO12x/ensemble |

---

## 🎯 Next Steps After Training

### 1. Test on Real Cards

```bash
# Export model
yolo export model=runs/detect/fab_2641_yolo11x/weights/best.pt format=onnx

# Test inference speed
yolo predict model=runs/detect/fab_2641_yolo11x/weights/best.pt source=path/to/real/cards/
```

### 2. Deploy for Streaming

```python
from ultralytics import YOLO

# Load model
model = YOLO("runs/detect/fab_2641_yolo11x/weights/best.pt")

# Run inference on stream
results = model.predict(source=0, stream=True)  # Webcam
for r in results:
    # Process detections
    boxes = r.boxes
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"Detected: Class {cls}, Confidence: {conf:.2f}")
```

### 3. Fine-tune if Needed

If real-world accuracy < 90%:
- Generate more training data (50k images)
- Add real card images to training set
- Train for more epochs (500-600)
- Try YOLO12x

---

## 📚 Additional Resources

- **Ultralytics Docs:** https://docs.ultralytics.com
- **RunPod Guide:** `RUNPOD_SETUP.md`
- **Implementation Notes:** `HARD_CASE_IMPLEMENTATION.md`
- **Model Architecture:** https://github.com/ultralytics/ultralytics

---

**Ready to start? Begin with Step 1: Generate Dataset** 🚀

```powershell
cd "c:\VS Code\FaB Code"
python scripts\generate_dataset_25k.py
```
