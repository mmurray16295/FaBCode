# RunPod Setup - Claude Directions

**Purpose**: Complete workflow guide for setting up and training the FaB Card Detector on RunPod from scratch.

**Audience**: Claude AI assistant helping with RunPod training sessions.

---

## Overview

This guide covers the complete RunPod training workflow for any training phase. Each training phase (Phase 1, 2, 3, etc.) follows this same process.

**Current Project Status**: Phase 3 complete (90.5% mAP, 2,641 classes). Next training run will be Phase 4.

**Workflow Steps**:
1. Setup RunPod environment
2. Download card assets
3. Generate synthetic training data
4. Train YOLO model
5. Download trained weights

---

## Step 1: Initial Setup

### 1.0 Check Pod Specifications

**CRITICAL**: RunPod instances are partitioned. **Always refer to RunPod UI "Pod Details"** for accurate resource limits.

**Why internal checks can be misleading**:
- `df -h /workspace` shows the **entire shared storage pool**, not your volume quota
- `nproc` shows **host CPU count**, not your vCPU allocation
- `free -h` shows **host memory**, not your pod limit
- Only GPU specs and container disk are reliably detectable internally

**Detectable internally**:
```bash
# Memory limit (accurate from cgroup)
cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null | numfmt --to=iec-i

# Container disk (accurate)
df -h / | grep overlay

# GPU (accurate)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

**Must check in RunPod UI** (not reliably detectable):
- **vCPU allocation** (UI shows actual cores, cgroup may differ)
- **Volume Storage quota** (UI shows your limit, df shows shared pool total)

**Example pod specs** (always verify in RunPod UI):
- **GPU**: RTX 5090 1x (32GB VRAM)
- **vCPU**: 16 cores
- **Memory**: 141 GB
- **Container Disk**: 30 GB (temporary, for OS/deps only)
- **Volume Storage**: 600 GB at /workspace (persistent, for data/models)

**Important**: 
- Use `/workspace` for ALL data, models, and outputs (persistent volume with quota)
- Container disk is temporary and limited - only for dependencies
- Set worker processes based on vCPU count from RunPod UI, not `nproc`

### 1.1 Clone Repository
```bash
git clone https://github.com/mmurray16295/FaBCode.git
cd FaBCode
git checkout runpod-setup
```

### 1.2 Install Dependencies
```bash
pip install -r requirements.txt
```

**Key dependencies**:
- `ultralytics` - YOLO11 framework
- `opencv-python` - Image processing
- `pillow` - Image manipulation
- `numpy` - Numerical operations
- `pyyaml` - Configuration files
- `requests` - API calls
- `tqdm` - Progress bars

### 1.3 Prepare Phase 3 Pre-trained Weights

Reassemble the Phase 3 model weights (90.5% mAP, 2,641 classes):

**Windows**:
```bash
cd models/Phase3
REASSEMBLE_WEIGHTS.bat
```

**Linux/Mac**:
```bash
cd models/Phase3
chmod +x reassemble_weights.sh
./reassemble_weights.sh
```

This creates `best.pt` (345MB) from 4 split parts. See `models/Phase3/README.md` for details.

### 1.4 Verify Structure
Expected directory structure:
```
FaBCode/
├── data/
│   ├── card.json                           # Card database (4,218 cards)
│   ├── card_popularity_weights.json        # Card selection weights
│   ├── synthetic_smooth/                   # Smooth selector output (DEFAULT)
│   │   ├── train/images + labels/
│   │   ├── val/images + labels/
│   │   └── data.yaml                       # YOLO config (2,641 classes)
│   └── synthetic/                          # Weighted selector output
│       ├── train/images + labels/
│       ├── val/images + labels/
│       └── data.yaml                       # YOLO config (2,641 classes)
├── models/
│   ├── Phase3/
│   │   ├── best.pt                         # Phase 3 weights (reassembled)
│   │   ├── Phase3_best_part_*              # Split weight files (4 parts)
│   │   └── README.md                       # Detailed usage instructions
│   ├── training_smooth/                    # Smooth selector training runs
│   │   └── run_001/weights/
│   │       ├── best.pt
│   │       └── last.pt
│   └── training_weighted/                  # Weighted selector training runs
│       └── run_001/weights/
│           ├── best.pt
│           └── last.pt
├── scripts/
│   ├── synthetic_generation/               # Image generation scripts
│   ├── yolo_training/                      # Training scripts
│   └── asset_management/                   # Download utilities
└── yolo11x.pt                              # Base YOLO11x weights (optional)
```

---

## Step 2: Download Assets

### 2.1 Download Card Images
**Location**: `scripts/asset_management/download_all_printings_parallel.py`

```bash
cd scripts/asset_management
python download_all_printings_parallel.py --max-workers 15 --rate-limit 15
```

**What this does**:
- Downloads ALL card printings from FaB database (not just one per card)
- URL-based deduplication (skips duplicate images)
- Parallel downloads with rate limiting (15 req/sec for Google CDN)
- Downscales to 250px max dimension for efficiency
- Organizes by set: `data/images/<SET_ID>/<card_files>`
- ~4,218+ card printings total
- Estimated time: 10-20 minutes depending on connection

**Output structure**:
```
data/images/
├── WTR/
│   ├── Aether_Dart_R_WTR001.png
│   ├── Aether_Flare_Y_WTR002.png
│   └── ...
├── ARC/
│   ├── ...
└── [other sets]/
```

**Options**:
- `--max-workers`: Number of concurrent downloads (default: 15)
- `--rate-limit`: Requests per second (default: 15)
- `--max-size`: Max image dimension in pixels (default: 250)

### 2.2 Verify Background Template

The background template is already included in the repository at `data/Background Perfecting/`:

```
data/Background Perfecting/
├── images/
│   └── AI-Edit-High-Res-3_png.rf...jpg  # Base playmat image
├── labels/
│   └── AI-Edit-High-Res-3_png.rf...txt  # Zone placement labels
└── data.yaml                              # 25 zone class definitions
```

**Zone classes** (25 total):
- Hero positions (Hero, Hero 2)
- Equipment zones (Head, Chest, Arms, Legs, Weapon, etc.)
- Game zones (Pitch, Combat Chain, Graveyard, Banish)
- Reference markers

**Note**: The generator uses this single labeled background to understand WHERE to place cards. Visual variety comes from:
- Augmentations (lighting, blur, noise, color shifts)
- Rotation and overlap variations
- Optional windowed mode (simulates browser/desktop background)

---

## Step 3: Generate Synthetic Training Data

### 3.1 Understanding the Generator

**Main script**: `scripts/synthetic_generation/Core_Playmat_Generator.py`

**Key features**:
- Places 20-30 cards per synthetic playmat
- Two heroes per playmat (multi-player simulation)
- Realistic card positioning with rotation, overlap, occlusion
- Equipment, weapons, pitch zones, combat chains, graveyards
- Augmentation: lighting, blur, noise, color shifts

**Card Selection**:
- **Weighted selector** (`card_selector.py`): Uses popularity weights
- **Smooth selector** (`card_selector_smooth.py`): Even distribution across all cards

**Format Distribution**:
- 70% Classic Constructed (adult heroes)
- 30% Blitz (young heroes)

**Token Frequency**:
- 75% of playmats include 1-3 tokens
- Tokens appear in combat chain zones

### 3.2 Test Card Distribution (Optional)

Before generating thousands of images, test the card selector:

```bash
cd scripts/synthetic_generation

# Quick test (1,000 iterations, ~8 seconds)
python card_selector_test_quick.py quick

# Standard test (25,000 iterations, ~3 minutes)
python card_selector_test_quick.py standard

# Compare weighted vs smooth (5,000 each, ~1.5 minutes)
python card_selector_test_quick.py compare-quick
```

**Output files**:
- `card_distribution_<selector>_<iterations>_<timestamp>.json` - Full statistics
- `card_distribution_<selector>_<iterations>_<timestamp>_cards.txt` - Sorted card list

**What to check**:
- All cards represented (2,641 unique cards expected)
- Hero distribution matches format split (70% adult, 30% young)
- Token frequencies reasonable (~200 per token in 5k test)

### 3.3 Generate Training Dataset

**Single-process generation** (testing):
```bash
cd scripts/synthetic_generation

# Smooth selector (even distribution) - DEFAULT
python Core_Playmat_Generator.py --selector smooth

# Weighted selector (popularity-based)
python Core_Playmat_Generator.py --selector weighted
```

**Note**: Single-process mode generates one image per run. Output directory:
- `--selector smooth` → `data/synthetic_smooth/` (default, even distribution)
- `--selector weighted` → `data/synthetic/` (popularity-based)

**Parallel generation** (production):
```bash
# Smooth selector (even distribution) - RECOMMENDED
python parallel_generate_dataset.py --total-images 50000 --workers 8 --selector smooth

# Weighted selector (popularity-based)
python parallel_generate_dataset.py --total-images 50000 --workers 8 --selector weighted
```

**Note**: Output is automatically organized with `train/` and `val/` splits:
- `--selector smooth` → `data/synthetic_smooth/` (default)
- `--selector weighted` → `data/synthetic/`

**Recommended settings**:
- **Total images**: 50,000 minimum (auto-split 80/20 train/val)
- **Workers**: 4-8 depending on CPU cores
- **Time estimate**: ~6 hours for 50k images on 8-core CPU

**Generation parameters**:
```python
# In Core_Playmat_Generator.py (configurable)
MIN_CARDS = 20                    # Minimum cards per image
MAX_CARDS = 30                    # Maximum cards per image
COMBAT_CHAIN_PROB = 0.7           # 70% chance of combat chain
TOKEN_PROB = 0.75                 # 75% chance of tokens
TOKEN_COUNT = (1, 3)              # 1-3 tokens when present
```

**Output structure**:
```
data/synthetic_smooth/     # Smooth selector (even distribution) - DEFAULT
├── train/
│   ├── images/
│   │   ├── playmat_000001.jpg
│   │   ├── playmat_000002.jpg
│   │   └── ...
│   └── labels/
│       ├── playmat_000001.txt
│       ├── playmat_000002.txt
│       └── ...
├── val/
│   ├── images/
│   └── labels/
└── data.yaml              # YOLO config with 2,641 classes

data/synthetic/            # Weighted selector (popularity-based)
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── data.yaml
```

---

## Step 4: YOLO Training

**Location**: `scripts/yolo_training/train_yolo.py`

**Note**: The `data.yaml` file is auto-generated during image generation with all 2,641 card classes.

### 4.1 Start Training

**Transfer Learning from Phase 3 (Recommended)**:
```bash
cd scripts/yolo_training

# Training on smooth selector data (even distribution) - DEFAULT
python train_yolo.py \
  --data ../../data/synthetic_smooth/data.yaml \
  --model ../../models/Phase3/best.pt \
  --epochs 50 \
  --batch 16 \
  --imgsz 640 \
  --device 0 \
  --project ../../models/training_smooth \
  --name run_001

# Training on weighted selector data (popularity-based)
python train_yolo.py \
  --data ../../data/synthetic/data.yaml \
  --model ../../models/Phase3/best.pt \
  --epochs 50 \
  --batch 16 \
  --imgsz 640 \
  --device 0 \
  --project ../../models/training_weighted \
  --name run_001
```

**Training from Scratch** (not recommended - use transfer learning):
```bash
# Smooth selector
python train_yolo.py \
  --data ../../data/synthetic_smooth/data.yaml \
  --model yolo11x.pt \
  --epochs 100 \
  --batch 16 \
  --device 0 \
  --project ../../models/training_smooth \
  --name run_001
```

**Training parameters**:
- `--model`: Use `../../models/Phase3/best.pt` for transfer learning (faster, better results) or `yolo11x.pt` for from-scratch training
- `--epochs`: 50 for transfer learning, 100+ for from-scratch
- `--batch 16`: Batch size (adjust for GPU memory)
- `--imgsz 640`: Input image size (640x640)
- `--device 0`: GPU device (use `cpu` for CPU training)
- `--patience 20`: Early stopping after 20 epochs without improvement

**Hardware recommendations**:
- **GPU**: RTX 3090/4090 (24GB VRAM) - batch size 32-64
- **GPU**: RTX 3060 (12GB VRAM) - batch size 16
- **CPU**: Not recommended (100x slower)

**Transfer Learning Benefits**:
- Starts from 90.5% mAP baseline
- Converges in 20-50 epochs vs 100+
- Better final accuracy
- Saves GPU time and RunPod costs

### 4.2 Monitor Training

Training outputs:
```
models/training_smooth/run_001/
├── weights/
│   ├── best.pt          # Best model by validation mAP
│   └── last.pt          # Last epoch checkpoint
├── results.csv          # Training metrics per epoch
├── results.png          # Loss/mAP curves
└── confusion_matrix.png # Validation confusion matrix
```

**Key metrics to watch**:
- `mAP50`: Mean Average Precision at 50% IoU (target: >0.90)
- `mAP50-95`: Mean Average Precision at 50-95% IoU (target: >0.70)
- `box_loss`: Bounding box regression loss (should decrease steadily)
- `cls_loss`: Classification loss (should decrease steadily)

### 4.3 Resume or Continue Training

**Resume from last checkpoint** (if training was interrupted cleanly):
```bash
# Resume smooth selector training - uses last.pt with optimizer state
python train_yolo.py \
  --data ../../data/synthetic_smooth/data.yaml \
  --model ../../models/training_smooth/run_001/weights/last.pt \
  --resume
```

**Continue from best checkpoint** (if last.pt corrupted or want to train longer):
```bash
# Continue training from best.pt - starts fresh optimizer, keeps weights
python train_yolo.py \
  --data ../../data/synthetic_smooth/data.yaml \
  --model ../../models/training_smooth/run_001/weights/best.pt \
  --epochs 100 \
  --batch 16 \
  --device 0 \
  --project ../../models/training_smooth \
  --name run_002
```

**Note**: `--resume` only works with `last.pt` (contains optimizer state). To continue from `best.pt`, start a new training run without `--resume`.

---

## Step 5: Download Trained Model

After training completes, download both weight files:

```bash
# Copy best model (highest accuracy - for deployment)
cp models/training_smooth/run_001/weights/best.pt models/phase4_best_$(date +%Y%m%d).pt

# Copy last checkpoint (for resuming training if needed)
cp models/training_smooth/run_001/weights/last.pt models/phase4_last_$(date +%Y%m%d).pt

# Download from RunPod using their file browser or CLI
```

**Which file to download:**
- `best.pt` - **Always download** - Best validation accuracy, use for deployment and continuing training
- `last.pt` - Optional - Only needed if you want to resume exact training state (includes optimizer)
- `results.csv` and `results.png` - Recommended - Training metrics and visualizations

**What's next**: Test and deploy using applications in `scripts/computer_vision_application/`.

---

## Common Issues and Solutions

### Issue: Out of GPU Memory

**Solution**: Reduce batch size
```bash
python train_yolo.py --batch 8  # Instead of 16
```

### Issue: Training too slow

**Solution**: 
- Reduce image size: `--imgsz 416` instead of 640
- Use fewer training images
- Enable mixed precision: `--amp`

### Issue: Poor detection accuracy

**Solutions**:
1. Generate more training data (100k+ images)
2. Increase augmentation strength
3. Train for more epochs
4. Use larger model: `yolo11s.pt` or `yolo11m.pt`
5. Check token frequency (should be ~200 per token in 5k test)

### Issue: Cards not downloading

**Solution**: Check internet connection and retry with fewer workers
```bash
python download_all_printings_parallel.py --max-workers 5 --rate-limit 5
```

### Issue: "Module not found" errors

**Solution**: Reinstall dependencies
```bash
pip install -r requirements.txt --force-reinstall
```

---

## Performance Benchmarks

**Image Generation**:
- Weighted selector: ~119 playmats/second
- Smooth selector: ~138 playmats/second
- 50,000 images: ~2-3 hours (8-core CPU)

**Training (from scratch)**:
- RTX 4090: ~8 hours for 100 epochs (50k images, batch 64)
- RTX 3090: ~12 hours for 100 epochs (50k images, batch 32)
- RTX 3060: ~24 hours for 100 epochs (50k images, batch 16)

**Training (transfer learning from Phase 3)**:
- RTX 4090: ~4 hours for 50 epochs (50k images, batch 64)
- RTX 3090: ~6 hours for 50 epochs (50k images, batch 32)
- RTX 3060: ~12 hours for 50 epochs (50k images, batch 16)

**Expected Results**:
- From scratch - mAP50: 0.92-0.96, mAP50-95: 0.72-0.82
- Transfer learning - mAP50: 0.96-0.98, mAP50-95: 0.88-0.93
- Inference: 30-60 FPS on RTX 3090

---

## Quick Reference Commands

```bash
# Step 1: Setup
git clone https://github.com/mmurray16295/FaBCode.git
cd FaBCode && git checkout runpod-setup
pip install -r requirements.txt
cd models/Phase3 && ./reassemble_weights.sh && cd ../..

# Step 2: Download assets
cd scripts/asset_management
python download_all_printings_parallel.py --max-workers 15

# Step 3: Generate training data
cd ../synthetic_generation
python parallel_generate_dataset.py --total-images 50000 --workers 8 --selector smooth

# Step 4: Train model
cd ../yolo_training
python train_yolo.py --data ../../data/synthetic_smooth/data.yaml --model ../../models/Phase3/best.pt --epochs 50 --batch 16 --device 0 --project ../../models/training_smooth

# Step 5: Download trained model
cp models/training_smooth/run_001/weights/best.pt models/phase4_$(date +%Y%m%d).pt
```

---

## Additional Resources

- **Phase 3 Weights Guide**: `models/Phase3/README.md` - Detailed model information and usage
- **Card Distribution Testing**: `scripts/synthetic_generation/CARD_DISTRIBUTION_TESTING.md`
- **Data Organization**: `scripts/synthetic_generation/DATA_ORGANIZATION.md`
- **Application Deployment**: `scripts/computer_vision_application/README.md`
- **Repository README**: `README.md`
- **Troubleshooting**: `TROUBLESHOOTING.md`

---

## Configuration Summary

**Current Settings**:
- Format split: 70% CC / 30% Blitz
- Token probability: 75%
- Tokens per playmat: 1-3
- Cards per playmat: 20-30
- Combat chain: 70% probability
- Background variations: Enabled
- Augmentation: Medium strength

**To modify these settings**, edit `Core_Playmat_Generator.py` lines 1814-1819 (tokens) and the format selection in `card_selector.py` and `card_selector_smooth.py`.

---

**Last Updated**: October 22, 2025
**Branch**: runpod-setup
**Current Trained Model**: Phase 3 (90.5% mAP, 2,641 classes)
**Next Training Run**: Phase 4 (smooth selector, 50k+ images)
**Scope**: RunPod training workflow - see `scripts/computer_vision_application/` for deployment
