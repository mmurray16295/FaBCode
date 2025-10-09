#!/bin/bash
# Package FaB Card Detector for Local Testing
# Creates a portable package with model, detector, and card data

set -e

echo "=================================="
echo "FaB Card Detector - Model Packager"
echo "=================================="
echo ""

# Configuration
MODEL_PATH="runs/train/phase1_100classes/weights/best.pt"
PACKAGE_NAME="fab_card_detector_$(date +%Y%m%d_%H%M%S)"
PACKAGE_DIR="packages/$PACKAGE_NAME"

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "[error] Model not found at: $MODEL_PATH"
    echo "[info] Please wait for training to complete or specify a different model"
    exit 1
fi

echo "[info] Creating package: $PACKAGE_NAME"
echo ""

# Create package directory structure
mkdir -p "$PACKAGE_DIR"
mkdir -p "$PACKAGE_DIR/data"
mkdir -p "$PACKAGE_DIR/models"
mkdir -p "$PACKAGE_DIR/examples"

echo "[1/8] Copying detector scripts..."
cp live_detector.py "$PACKAGE_DIR/"
cp fab_detector_app.py "$PACKAGE_DIR/"
if [ -f "scripts/screen_detect.py" ]; then
    cp scripts/screen_detect.py "$PACKAGE_DIR/"
fi

echo "[2/8] Copying model weights..."
cp "$MODEL_PATH" "$PACKAGE_DIR/models/best.pt"

# Copy training metadata if available
if [ -f "runs/train/phase1_100classes/args.yaml" ]; then
    echo "[3/8] Copying training metadata..."
    cp "runs/train/phase1_100classes/args.yaml" "$PACKAGE_DIR/models/training_args.yaml"
else
    echo "[3/8] Training metadata not found, skipping..."
fi

# Copy class names
if [ -f "data/synthetic/classes.yaml" ]; then
    echo "[4/8] Copying class names..."
    cp "data/synthetic/classes.yaml" "$PACKAGE_DIR/data/"
else
    echo "[4/8] Class names not found, skipping..."
fi

# Copy card data (optional, for metadata)
if [ -f "data/card.json" ]; then
    echo "[5/8] Copying card metadata..."
    cp "data/card.json" "$PACKAGE_DIR/data/"
else
    echo "[5/8] Card metadata not found, skipping..."
fi

# Copy popularity weights (optional, for reference)
if [ -f "data/card_popularity_weights.json" ]; then
    echo "[6/8] Copying popularity weights..."
    cp "data/card_popularity_weights.json" "$PACKAGE_DIR/data/"
else
    echo "[6/8] Popularity weights not found, skipping..."
fi

echo "[7/8] Creating requirements.txt..."
cat > "$PACKAGE_DIR/requirements.txt" << 'EOF'
# FaB Card Detector Requirements
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
mss>=9.0.0
torch>=2.0.0
EOF

echo "[8/8] Creating README..."
cat > "$PACKAGE_DIR/README.md" << 'EOF'
# FaB Card Live Detector

Real-time Flesh and Blood card detection using YOLO.

## Quick Start

### Installation

1. Install Python 3.8 or later
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

**GUI Mode (Recommended):**
```bash
python fab_detector_app.py
```

This launches a user-friendly GUI where you can:
- Select detection mode (Windowed or Transparent Overlay)
- Configure monitors for multi-monitor setups
- Adjust confidence and IOU thresholds
- Enable/disable transparency and click-through
- Save/load your settings

**Command-Line Mode:**

Webcam Detection:
```bash
python live_detector.py --source webcam
```

Screen Capture Detection:
```bash
python live_detector.py --source screen
```

Test on Image:
```bash
python live_detector.py --source examples/test_image.jpg
```

Custom Model:
```bash
python live_detector.py --source webcam --model models/best.pt
```

## Controls

**GUI Mode:**
- Start/Stop buttons to control detection
- Settings persist between sessions
- Real-time FPS and card count display

**Command-Line Mode:**
- **'q'**: Quit
- **'c'**: Toggle confidence display

## Detection Modes

### Windowed Mode
Shows the captured screen with colored bounding boxes around detected cards.
- Green boxes around all detected cards
- Card names displayed above each box
- Card preview appears below the box when you hover over it
- Useful for debugging and seeing all detections at once

### Transparent Overlay Mode
Creates an invisible window that only shows card previews when you hover over them.
- Perfect for playing online (e.g., webcam games)
- Minimal visual interference
- Optional click-through mode (mouse clicks pass through the window)
- Keep-on-top option ensures it's always visible

### Multi-Monitor Support
- **Capture Monitor**: Which screen to watch for cards
- **Display Monitor**: Where to show the detection window
- Ideal for dual-monitor setups (play on one, detect on the other)

## Command-Line Options

- `--source`: Detection source (webcam/screen/image path)
- `--model`: Path to model weights (default: models/best.pt)
- `--conf`: Confidence threshold 0-1 (default: 0.25)
- `--iou`: IOU threshold for NMS (default: 0.45)
- `--camera`: Camera device ID for webcam (default: 0)
- `--monitor`: Monitor number for screen capture (default: 1)
- `--scale`: Display scale for screen capture (default: 0.5)
- `--save`: Save annotated image path (image mode only)

## Examples

**High confidence detection:**
```bash
python live_detector.py --source webcam --conf 0.5
```

**Screen capture on second monitor:**
```bash
python live_detector.py --source screen --monitor 2
```

**Batch process images:**
```bash
for img in examples/*.jpg; do
    python live_detector.py --source "$img" --save "output_$(basename $img)"
done
```

## Model Information

This model was trained on synthetic FaB card images with:
- YOLOv11 nano architecture
- Image size: 1280px
- Top 100 most popular cards
- Synthetic playmat backgrounds

See `models/training_args.yaml` for full training configuration.

## Troubleshooting

**No webcam found:**
- Try different camera IDs: `--camera 1`, `--camera 2`, etc.
- Check camera permissions

**Screen capture not working:**
- On macOS: Grant screen recording permission
- On Linux: May need X11 or Wayland configuration

**Slow detection:**
- Reduce image size internally (modify code)
- Use GPU-enabled PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

**Model not found:**
- Ensure `models/best.pt` exists
- Or specify custom path: `--model path/to/model.pt`

## System Requirements

- **CPU**: Any modern CPU (Intel/AMD/ARM)
- **RAM**: 4GB minimum, 8GB recommended
- **GPU**: Optional (NVIDIA with CUDA for faster inference)
- **Storage**: ~100MB for model and dependencies

## Performance

- **CPU-only**: 5-15 FPS (Intel i5/i7)
- **With GPU**: 30-60 FPS (NVIDIA GTX 1060+)

## License

Model and code for personal/educational use.
EOF

echo ""
echo "=================================="
echo "Package created successfully!"
echo "=================================="
echo ""
echo "Location: $PACKAGE_DIR"
echo "Size: $(du -sh "$PACKAGE_DIR" | cut -f1)"
echo ""

# Get model info
if [ -f "$MODEL_PATH" ]; then
    MODEL_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
    echo "Model: best.pt ($MODEL_SIZE)"
fi

echo ""
echo "To test locally:"
echo "  1. Copy $PACKAGE_DIR to your local machine"
echo "  2. cd $PACKAGE_NAME"
echo "  3. pip install -r requirements.txt"
echo "  4. python live_detector.py --source webcam"
echo ""

# Create compressed archive
echo "Creating compressed archive..."
cd packages
tar -czf "$PACKAGE_NAME.tar.gz" "$PACKAGE_NAME"
ARCHIVE_SIZE=$(du -h "$PACKAGE_NAME.tar.gz" | cut -f1)
echo ""
echo "Archive created: packages/$PACKAGE_NAME.tar.gz ($ARCHIVE_SIZE)"
echo ""
echo "Download this file to test on your local machine!"
echo ""
