# Download & Test Instructions

## Package Ready! 🎉

Your FaB Card Detector is packaged and ready to test!

### Package Location
```
/root/FaBCode/packages/fab_card_detector_20251009_232204.tar.gz
```

**Size:** 12MB (compressed)

### Model Performance (Phase 1 - Top 100 Cards)
- **Training stopped at:** Epoch 71/100
- **Final mAP50:** 99.2%
- **Final mAP50-95:** 93.8%
- **Precision:** 99.2%
- **Recall:** 97.9%

This is **excellent performance** - the model has essentially mastered the top 100 cards!

## Download Options

### Option 1: RunPod File Browser (Easiest)
1. In RunPod web interface, click "Connect" → "File Browser"
2. Navigate to: `/root/FaBCode/packages/`
3. Right-click `fab_card_detector_20251009_232204.tar.gz`
4. Click "Download"

### Option 2: SCP (Command Line)
From your local machine:
```bash
# Replace <pod-id> and <ssh-key> with your RunPod details
scp -i ~/.ssh/runpod root@<pod-id>.ssh.runpod.io:/root/FaBCode/packages/fab_card_detector_20251009_232204.tar.gz .
```

### Option 3: Rsync (More Efficient)
```bash
rsync -avz -e "ssh -i ~/.ssh/runpod" root@<pod-id>.ssh.runpod.io:/root/FaBCode/packages/fab_card_detector_20251009_232204.tar.gz .
```

## Setup on Your Local Machine

### 1. Extract the Package
```bash
tar -xzf fab_card_detector_20251009_232204.tar.gz
cd fab_card_detector_20251009_232204
```

### 2. Install Dependencies
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Test the Application

**GUI Mode (Recommended):**
```bash
python fab_detector_app.py
```

**Command-Line Mode:**
```bash
# Webcam test
python live_detector.py --source webcam --model models/best.pt

# Screen capture test
python live_detector.py --source screen --model models/best.pt
```

## What's Included

```
fab_card_detector_20251009_232204/
├── README.md                    # Quick start guide
├── requirements.txt             # Python dependencies
├── fab_detector_app.py          # GUI application ⭐
├── live_detector.py             # Command-line tool
├── screen_detect.py             # Advanced features (Windows)
├── models/
│   ├── best.pt                  # Phase 1 model (11MB)
│   └── training_args.yaml       # Training configuration
├── data/
│   ├── classes.yaml             # Class names (100 cards)
│   ├── card.json                # Full card metadata (19MB)
│   └── card_popularity_weights.json  # Card rankings
└── examples/                    # (empty, for your test images)
```

## Quick Test Workflow

1. **Launch GUI:**
   ```bash
   python fab_detector_app.py
   ```

2. **In GUI:**
   - Model should auto-detect: `models/best.pt`
   - Choose "Windowed Mode" for first test
   - Set Capture Monitor to 1 (primary)
   - Leave other settings at defaults
   - Click "Start Detection"

3. **Test with Cards:**
   - Place some FaB cards in front of webcam, OR
   - Open card images on screen for screen capture mode
   - Hover over detected boxes to see full card preview

4. **Try Transparent Overlay:**
   - Stop detection
   - Switch to "Transparent Overlay Mode"
   - Start detection again
   - Only card preview shows when hovering

## Troubleshooting

### "No module named 'ultralytics'"
```bash
pip install ultralytics torch opencv-python mss numpy
```

### "Model not found"
In GUI, click "Browse" and navigate to `models/best.pt`

### No detections showing
- Lower confidence threshold to 0.25
- Ensure good lighting
- Cards must be from top 100 (see `data/classes.yaml`)

### Poor performance / Low FPS
**For GPU acceleration:**
```bash
# Uninstall CPU-only PyTorch
pip uninstall torch

# Install GPU version (CUDA 11.8)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Phase 1 Model Details

**Training Data:**
- 10,363 synthetic images (YouTube playmat backgrounds)
- Top 100 most popular FaB cards
- Weighted sampling with popularity dampening
- 70% train / 20% validation / 10% test split

**Model Architecture:**
- YOLOv11 nano (lightweight, fast)
- Input size: 1280px
- Trained for 71 epochs

**Classes Detected (Top 100):**
Check `data/classes.yaml` for full list of cards the model can detect.

## Performance Expectations

| Hardware | Webcam FPS | Screen Capture FPS |
|----------|------------|-------------------|
| CPU Only (i5/i7) | 5-10 | 8-15 |
| GTX 1060 | 25-35 | 30-45 |
| RTX 3060 | 40-60 | 50-80 |
| RTX 4090 | 80-120 | 100-150 |

## Next Steps After Testing

Once you've tested and are happy with Phase 1:

1. **Provide Feedback:**
   - Which cards work well?
   - Which cards are missed or misidentified?
   - Any false positives?
   - Performance issues?

2. **Phase 2 Planning:**
   - Generate 20,000 images for ranks 1-500 (all mission-critical cards)
   - Train Phase 2 model (500 classes)
   - Use Phase 1 weights as starting point (transfer learning)
   - Expected to take 4-6 hours total

3. **Optional Improvements:**
   - Add more training data for problematic cards
   - Adjust confidence thresholds
   - Try different image sizes
   - Fine-tune on real webcam captures

## Documentation

For comprehensive usage guide, see:
- `APPLICATION_GUIDE.md` (in GitHub repo)
- `SYSTEM_OPTIMIZATION.md` (in GitHub repo)
- `README.md` (in package)

## Support

If you encounter issues:
1. Check `data/classes.yaml` - card must be in top 100
2. Try lowering confidence threshold to 0.25
3. Test with good lighting conditions
4. Verify GPU is being used (check startup messages)

## Have Fun Testing! 🎮

The model should work great on the top 100 cards. Enjoy testing it with your collection!
