# RunPod Setup Guide for YOLO11x Training

Complete guide for setting up RunPod instance and training YOLO11x on 2641 FaB card classes.

---

## 📋 Prerequisites

- RunPod account with payment method
- ~50GB of data (25k synthetic images)
- Budget: $35-50 for full training run

---

## 🚀 Step 1: Create RunPod Instance

### 1.1 Select GPU

**Recommended: RTX 5090**
- **Cost**: $0.69/hour
- **VRAM**: 24GB
- **Speed**: Fastest training (~48 hours)
- **Total Cost**: ~$35

**Alternative: RTX 4090**
- **Cost**: $0.34/hour
- **VRAM**: 24GB
- **Speed**: Slower (~72 hours)
- **Total Cost**: ~$25

### 1.2 Select Template

Choose: **PyTorch 2.0+** or **Ultralytics YOLO** template

### 1.3 Configure Instance

- **Disk Space**: 100GB minimum
- **Container Image**: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`
- **Expose Ports**: 8888 (Jupyter), 6006 (TensorBoard)

### 1.4 Start Instance

Click **Deploy** and wait for instance to spin up (~2-3 minutes)

---

## 🔧 Step 2: Connect and Setup Environment

### 2.1 Connect via SSH

```bash
# Get SSH command from RunPod dashboard
ssh root@<pod-id>.runpod.io -p <port>
```

### 2.2 Install Dependencies

```bash
# Update system
apt-get update && apt-get install -y git vim htop tmux

# Install Python packages
pip install --upgrade pip
pip install ultralytics pillow numpy opencv-python pandas matplotlib seaborn tqdm
pip install tensorboard jupyter

# Verify CUDA
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 2.3 Clone Repository

```bash
# Clone your FaB Code repository
git clone https://github.com/mmurray16295/FaBCode.git
cd FaBCode
```

---

## 📦 Step 3: Upload Dataset

### Option A: Upload via SCP (Recommended for Speed)

**On your local machine:**
```powershell
# Compress dataset first (saves upload time)
cd "c:\VS Code\FaB Code"
tar -czf synthetic_dataset.tar.gz data/synthetic/

# Upload to RunPod (get SSH details from dashboard)
scp -P <port> synthetic_dataset.tar.gz root@<pod-id>.runpod.io:/workspace/FaBCode/
```

**On RunPod:**
```bash
cd /workspace/FaBCode
tar -xzf synthetic_dataset.tar.gz
rm synthetic_dataset.tar.gz
```

### Option B: Generate Dataset Directly on RunPod

```bash
cd /workspace/FaBCode

# Upload card images and json first
# Then generate synthetic data on RunPod (takes ~7 hours)
python scripts/generate_dataset_25k.py
```

---

## 🏋️ Step 4: Run Training

### 4.1 Start tmux Session (Important!)

```bash
# Start tmux so training continues if you disconnect
tmux new -s yolo_training
```

**tmux Commands:**
- Detach: `Ctrl+B, then D`
- Reattach: `tmux attach -t yolo_training`
- List sessions: `tmux ls`

### 4.2 Optional: Run Sanity Check First

```bash
cd /workspace/FaBCode
python scripts/train_sanity_check.py
```

**Expected Results:**
- Time: ~2 hours
- Cost: ~$1.40
- mAP@0.5: Should be >85%

### 4.3 Run Full Training

```bash
cd /workspace/FaBCode
python scripts/train_yolo11x.py
```

**Expected Timeline:**
- Time: ~48 hours (RTX 5090) or ~72 hours (RTX 4090)
- Cost: ~$35 (5090) or ~$25 (4090)

### 4.4 Monitor Training

**In another terminal/tmux pane:**

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Watch training output
tail -f runs/detect/fab_2641_yolo11x/train.log

# Start TensorBoard
tensorboard --logdir runs/detect --host 0.0.0.0 --port 6006
```

**Access TensorBoard:**
- Open: `https://<pod-id>-6006.proxy.runpod.net`

---

## 📊 Step 5: Monitor Progress

### Key Metrics to Watch

**During Training:**
- **mAP@0.5**: Should increase to 90-97%
- **mAP@0.5:0.95**: Should reach 70-85%
- **Precision**: Should reach 90%+
- **Recall**: Should reach 85%+

**Training Curves:**
- Box loss: Should decrease smoothly
- Class loss: Should decrease to near zero
- DFL loss: Should stabilize

### Warning Signs

⚠️ **If you see:**
- Loss increasing: Learning rate too high
- Loss stuck: Learning rate too low
- mAP plateauing early: Need more data
- GPU utilization <80%: Batch size too small

---

## 💾 Step 6: Save Results

### 6.1 Download Best Model

**On your local machine:**
```powershell
# Download best weights
scp -P <port> root@<pod-id>.runpod.io:/workspace/FaBCode/runs/detect/fab_2641_yolo11x/weights/best.pt .

# Download all results (plots, metrics)
scp -r -P <port> root@<pod-id>.runpod.io:/workspace/FaBCode/runs/detect/fab_2641_yolo11x ./results/
```

### 6.2 Backup to Cloud (Recommended)

```bash
# Install rclone
curl https://rclone.org/install.sh | bash

# Configure Google Drive/Dropbox/etc
rclone config

# Sync results
rclone copy runs/detect/fab_2641_yolo11x gdrive:FaB_Training/
```

---

## 🧹 Step 7: Cleanup

### Before Terminating Instance

```bash
# Ensure training is complete
ls runs/detect/fab_2641_yolo11x/weights/best.pt

# Download results (see Step 6)

# Verify download succeeded
md5sum runs/detect/fab_2641_yolo11x/weights/best.pt
```

### Terminate Instance

1. Go to RunPod dashboard
2. Click **Terminate** on your pod
3. Confirm termination

**⚠️ WARNING:** All data on the instance will be lost!

---

## 🎯 Training Optimization Tips

### If Training is Too Slow

```bash
# Reduce batch size (trades speed for memory)
python scripts/train_yolo11x.py --batch 8

# Reduce image size (trades accuracy for speed)
python scripts/train_yolo11x.py --imgsz 480

# Reduce epochs (not recommended for final model)
python scripts/train_yolo11x.py --epochs 200
```

### If Running Out of Memory

```bash
# Reduce batch size
python scripts/train_yolo11x.py --batch 8

# Disable AMP (slower but less memory)
# Edit train_yolo11x.py: amp=False
```

### If mAP is Low

1. **Check data quality**: View augmented images
2. **Train longer**: Increase epochs to 500
3. **More data**: Generate 50k images instead of 25k
4. **Try YOLO12x**: Might be better

---

## 📈 Expected Results

### Training Progress

| Epoch | mAP@0.5 | mAP@0.5:0.95 | Time Remaining |
|-------|---------|--------------|----------------|
| 50    | ~70%    | ~50%         | 42 hours       |
| 100   | ~80%    | ~60%         | 36 hours       |
| 200   | ~90%    | ~70%         | 24 hours       |
| 300   | ~94%    | ~75%         | 12 hours       |
| 400   | ~96%    | ~78%         | Complete       |

### Final Model Performance

**Target Metrics:**
- mAP@0.5: **95-97%**
- mAP@0.5:0.95: **75-80%**
- Precision: **93-95%**
- Recall: **90-93%**
- Inference Speed: **50-60 FPS** on RTX 5090

---

## 🐛 Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
python scripts/train_yolo11x.py --batch 8

# Or use gradient accumulation (slower but same effective batch)
# Edit train_yolo11x.py: nbs=64 (accumulates over 64/batch steps)
```

### Training Crashes

```bash
# Resume from last checkpoint
python scripts/train_yolo11x.py --resume --checkpoint runs/detect/fab_2641_yolo11x/weights/last.pt
```

### Slow Data Loading

```bash
# Reduce workers
python scripts/train_yolo11x.py --workers 4

# Or cache dataset in memory (needs 64GB+ RAM)
# Edit train_yolo11x.py: cache=True
```

### Connection Lost

```bash
# Reconnect via SSH
ssh root@<pod-id>.runpod.io -p <port>

# Reattach to tmux session
tmux attach -t yolo_training

# Check training is still running
ps aux | grep python
```

---

## 📞 Support

**RunPod Issues:**
- Discord: https://discord.gg/runpod
- Support: support@runpod.io

**YOLO Issues:**
- Ultralytics Docs: https://docs.ultralytics.com
- GitHub: https://github.com/ultralytics/ultralytics

**FaB Code Issues:**
- Check: `HARD_CASE_IMPLEMENTATION.md`
- Review: Training scripts in `scripts/`

---

## 💰 Cost Summary

| Task | Time | Cost (RTX 5090) | Cost (RTX 4090) |
|------|------|-----------------|-----------------|
| Sanity Check | 2h | $1.40 | $0.68 |
| Full Training | 48h | $33.12 | $24.48 |
| Buffer | 2h | $1.38 | $0.68 |
| **Total** | **52h** | **~$36** | **~$26** |

**Budget Allocation:**
- $36 primary (RTX 5090)
- $36 backup run if needed
- $36 YOLO12x comparison
- $36 ensemble training
- **Total Budget: $144 of $200** (leaves $56 buffer)

---

## ✅ Post-Training Checklist

- [ ] Download best.pt weights
- [ ] Download results.csv and plots
- [ ] Backup to cloud storage
- [ ] Test model on real cards
- [ ] Export to ONNX/TensorRT
- [ ] Document final metrics
- [ ] Terminate RunPod instance

---

**Ready to train? Start with Step 1!** 🚀
