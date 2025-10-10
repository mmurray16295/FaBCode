# Pod Migration Guide - Phase 3 Setup

## ✅ What's Saved (On GitHub)

All critical files are now pushed to `runpod-setup` branch:

1. **Phase 2 Model Weights** (23 MB)
   - `models/phase2/phase2_500classes_best.pt` - 98.87% mAP50
   - `models/phase2/phase2_500classes_last.pt` - Last checkpoint

2. **All Scripts**
   - Data generation with improved dice occluders
   - Training scripts
   - Package builder

3. **Card Data**
   - `data/card.json` (19 MB)
   - `data/card_popularity_weights.json` (1.5 MB)
   - `data/phase2_classes.yaml`

4. **Windows Package**
   - Latest package in `packages/` (committed)

## 🚀 New Pod Setup (Quick Start)

### Recommended Pod Specs for Phase 3:
- **GPU**: A100 80GB or RTX 4090
- **RAM**: 128GB+ system RAM
- **Storage**: 200GB+ disk space
- **Note**: Phase 3 = 1000 classes, 66k images, needs more resources than Phase 2

### Setup Commands:

```bash
# 1. Clone repo
git clone https://github.com/mmurray16295/FaBCode.git
cd FaBCode
git checkout runpod-setup

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify model weights exist
ls -lh models/phase2/phase2_500classes_best.pt
# Should show: 12M file

# 4. Download card images (one-time)
python3 scripts/download_images.py

# 5. Generate Phase 3 data (60k synthetic images)
nohup python3 scripts/generate_synthetic_playmat_screenshots.py \
  --output_dir data/phase3_1000classes \
  --num_images 60000 \
  --ranks 1-1000 \
  --use_popularity \
  > logs/phase3_generation.log 2>&1 &

# 6. Generate random backgrounds (6.6k images)
python3 scripts/generate_random_playmat.py \
  --output_dir data/phase3_random \
  --num_images 6600

# 7. Train Phase 3 (no RAM cache to avoid memory issues)
nohup python3 scripts/train_yolo.py \
  --data data/phase3_1000classes/data.yaml \
  --model models/phase2/phase2_500classes_best.pt \
  --project runs/train \
  --name phase3_1000classes \
  --epochs 300 \
  --batch 16 \
  --imgsz 1280 \
  --patience 30 \
  --device 0 \
  --workers 8 \
  > logs/phase3_training.log 2>&1 &
```

## 📊 Phase 3 Improvements

**New Features:**
- **Realistic Dice Occluders**: 25% of cards have dice
  - Centered placement (±10% jitter)
  - 15-30% card coverage
  - Accurate colors: white, red, black, blue, green
  - Two shapes: circular (d20) and rounded square (d6)
  - 3D highlights and shadows

**Training Strategy:**
- Start from Phase 2 weights (transfer learning)
- 1000 classes (ranks 1-1000)
- 66,600 total images
- No RAM caching (disk I/O instead)
- Expect ~20 hours training time

## 🔍 Monitoring Training

```bash
# Watch progress
tail -f logs/phase3_training.log

# Check results
tail -20 runs/train/phase3_1000classes/results.csv
```

## 💾 Important Notes

**DO NOT regenerate on old pod:**
- Phase 3 needs 66k images = ~25-30GB
- Training needs >100GB RAM for caching
- Old pod will run out of memory

**On new pod:**
- All images are regenerated (not stored in git)
- Takes ~4-6 hours to generate 66k images
- Training takes ~20 hours without RAM cache
- Everything else restores from git

## 🎯 Success Criteria

Phase 3 should achieve:
- **mAP50**: >98.5%
- **mAP50-95**: >91%
- **Better dice detection** than Phase 2
- **1000 card classes** (double Phase 2)

## 🆘 If Something Goes Wrong

All critical data is in GitHub:
- Branch: `runpod-setup`
- Latest commit: Includes Phase 2 weights + dice improvements
- Can always re-clone and restart

**Contact info**: Check repo README for support links
