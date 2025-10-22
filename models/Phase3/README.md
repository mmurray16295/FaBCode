# Phase 3 Pre-trained Weights

## 📦 Model Information

- **Model**: YOLOv11x
- **Classes**: 2,641 unique Flesh and Blood cards
- **Training Images**: 125,000 synthetic playmats
- **Best Epoch**: 27
- **Validation mAP50-95**: 90.5%
- **Model Size**: 360 MB (345 MB compressed)
- **Parameters**: 56.9M
- **Training Date**: October 18-20, 2025

---

## 🔧 Quick Start

### Windows
```batch
REASSEMBLE_WEIGHTS.bat
```

### Linux/Mac
```bash
chmod +x reassemble_weights.sh
./reassemble_weights.sh
```

This will create `best.pt` from the split parts.

---

## 📋 Split Archive Structure

Due to GitHub's 100MB file size limit, the model weights are split into 4 parts:

- `Phase3_best_part_a` (90 MB)
- `Phase3_best_part_b` (90 MB)
- `Phase3_best_part_c` (90 MB)
- `Phase3_best_part_d` (74 MB)

**Total**: 344 MB → `best.pt` (360,824,151 bytes)

---

## 🎯 Usage Scenarios

### 1. Transfer Learning (Recommended for RunPod)

Continue training from these weights with new synthetic data:

```bash
cd scripts/yolo_training
python train_yolo.py \
  --data ../../data/<your_training_run>/data.yaml \
  --model ../../models/Phase3/best.pt \
  --epochs 50 \
  --batch 16 \
  --device 0 \
  --project ../../models/<new_training_run> \
  --name run_001
```

**Benefits**:
- Faster convergence (starts from 90.5% mAP)
- Requires fewer epochs (20-50 instead of 100+)
- Better final accuracy
- Saves GPU time and cost

### 2. Training from Scratch

Start fresh with base YOLO11 weights:

```bash
python train_yolo.py \
  --data ../../data/<your_training_run>/data.yaml \
  --model yolo11n.pt \
  --epochs 100 \
  --batch 16 \
  --device 0
```

**When to use**:
- Testing major architecture changes
- Experimenting with different augmentation strategies
- Comparative benchmarking

### 3. Inference Testing

Test the model on real card images:

```bash
cd scripts/yolo_training
python test_inference.py \
  --model ../../models/Phase3/best.pt \
  --source <path_to_test_images> \
  --conf 0.25 \
  --save-txt
```

### 4. Application Deployment

Use with the detector application:

```bash
cd scripts/computer_vision_application
python fab_detector_app.py --model ../../models/Phase3/best.pt
```

---

## 📊 Training Details

### Dataset Composition
- **Format Split**: 70% Classic Constructed / 30% Blitz
- **Heroes**: 120 total (72 young, 71 adult)
- **Cards per Playmat**: 20-30
- **Combat Chains**: 70% probability
- **Tokens**: 75% probability (1-3 per playmat)
- **Augmentation**: Rotation, lighting, blur, occlusion

### Performance Metrics (Epoch 27)
- **mAP50**: 96.2%
- **mAP50-95**: 90.5%
- **Precision**: 93.8%
- **Recall**: 91.4%
- **Box Loss**: 0.623
- **Class Loss**: 0.441

### Hardware Used
- **GPU**: NVIDIA A100 (40GB VRAM)
- **Training Time**: ~48 hours
- **Batch Size**: 32
- **Image Size**: 640x640

---

## ⚠️ Important Notes

1. **Git LFS**: These split files are tracked with Git LFS. Make sure you have Git LFS installed:
   ```bash
   git lfs install
   git lfs pull
   ```

2. **File Integrity**: After reassembly, verify the file size:
   ```bash
   # Should be exactly 360,824,151 bytes
   ls -l best.pt
   ```

3. **Storage**: The reassembled `best.pt` file is included in `.gitignore` to avoid duplication.

4. **Updates**: When training produces better weights, update this folder by:
   - Splitting new `best.pt` into 4 parts
   - Updating this README with new metrics
   - Committing the new split parts

---

## 🔄 Manual Reassembly

If the scripts don't work, manually reassemble:

### Windows (CMD)
```batch
copy /b Phase3_best_part_a+Phase3_best_part_b+Phase3_best_part_c+Phase3_best_part_d best.pt
```

### Linux/Mac (Bash)
```bash
cat Phase3_best_part_a Phase3_best_part_b Phase3_best_part_c Phase3_best_part_d > best.pt
```

### Python
```python
with open('best.pt', 'wb') as outfile:
    for part in ['Phase3_best_part_a', 'Phase3_best_part_b', 'Phase3_best_part_c', 'Phase3_best_part_d']:
        with open(part, 'rb') as infile:
            outfile.write(infile.read())
print("Reassembly complete!")
```

---

## 📚 Additional Resources

- **RunPod Setup Guide**: `../../RUNPOD_SETUP_CLAUDE_DIRECTIONS.md`
- **Application Guide**: `../../scripts/computer_vision_application/APPLICATION_GUIDE.md`
- **Training Scripts**: `../../scripts/yolo_training/`
- **Synthetic Generation**: `../../scripts/synthetic_generation/`

---

**Last Updated**: October 22, 2025
**Model Version**: Phase 3 (2,641 classes)
**Checkpoint**: Epoch 27 (best validation mAP)
