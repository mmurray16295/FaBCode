#!/bin/bash

# ============================================================
# FaB Card Detector - Windows Package Builder
# ============================================================

set -e

echo "============================================================"
echo "   FaB Card Detector - Windows Package Builder"
echo "============================================================"
echo ""

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"
echo ""

# Create timestamp for package name
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
PACKAGE_NAME="FaBCardDetector_Windows_${TIMESTAMP}"
PACKAGE_DIR="packages/${PACKAGE_NAME}"

echo "[1/6] Creating package directory..."
mkdir -p "$PACKAGE_DIR"
echo "  ✓ Created: $PACKAGE_DIR"

# Copy installer scripts
echo ""
echo "[2/6] Copying installer scripts..."
cp "INSTALL_WINDOWS.bat" "$PACKAGE_DIR/"
cp "RUN_DETECTOR.bat" "$PACKAGE_DIR/"
cp "README_WINDOWS.txt" "$PACKAGE_DIR/"
echo "  ✓ Copied installer files"

# Copy application files
echo ""
echo "[3/6] Copying application files..."
cp "fab_detector_app.py" "$PACKAGE_DIR/"
cp "live_detector.py" "$PACKAGE_DIR/"
echo "  ✓ Copied Python scripts"

# Copy model weights
echo ""
echo "[4/6] Copying model weights..."
MODEL_PATH="runs/train/phase1_100classes/weights/best.pt"
if [ -f "$MODEL_PATH" ]; then
    mkdir -p "$PACKAGE_DIR/models"
    cp "$MODEL_PATH" "$PACKAGE_DIR/models/best.pt"
    MODEL_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
    echo "  ✓ Copied model: best.pt ($MODEL_SIZE)"
else
    echo "  ✗ WARNING: Model not found at $MODEL_PATH"
fi

# Copy data files
echo ""
echo "[5/6] Copying data files..."
mkdir -p "$PACKAGE_DIR/data"

DATA_FILES=(
    "data/card.json"
    "data/card_popularity_weights.json"
    "data/classes.yaml"
)

for file in "${DATA_FILES[@]}"; do
    if [ -f "$file" ]; then
        cp "$file" "$PACKAGE_DIR/data/"
        FILE_SIZE=$(du -h "$file" | cut -f1)
        FILE_NAME=$(basename "$file")
        echo "  ✓ Copied: $FILE_NAME ($FILE_SIZE)"
    else
        echo "  ✗ WARNING: $file not found"
    fi
done

# Create requirements.txt for reference
echo ""
echo "[6/6] Creating requirements.txt..."
cat > "$PACKAGE_DIR/requirements.txt" <<EOF
# FaB Card Detector - Python Dependencies
# This file is for reference only - INSTALL_WINDOWS.bat handles installation

# Core AI/ML
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0

# Computer Vision
opencv-python-headless>=4.8.0
pillow>=10.0.0

# Screen Capture
mss>=9.0.0

# Utilities
numpy>=1.24.0
pyyaml>=6.0
requests>=2.31.0

# GUI (built-in with Python)
# tkinter (included with Python)

# Installation Notes:
# - INSTALL_WINDOWS.bat uses CPU-only PyTorch for lightweight installation
# - For GPU support, manually install: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# - Total size: ~200MB installed
EOF
echo "  ✓ Created requirements.txt"

# Create QUICKSTART file
echo ""
echo "Creating QUICKSTART guide..."
cat > "$PACKAGE_DIR/QUICKSTART.txt" <<EOF
=============================================================
   FaB Card Detector - QUICK START
=============================================================

🚀 FIRST TIME SETUP (2-3 minutes):

   1. Double-click: INSTALL_WINDOWS.bat
   2. Wait for setup to complete
   3. App launches automatically!

=============================================================

✨ AFTER FIRST INSTALL:

   - Double-click: RUN_DETECTOR.bat
   - Or use desktop shortcut: "FaB Card Detector"

=============================================================

📖 NEED HELP?

   Read: README_WINDOWS.txt (detailed guide)

=============================================================

⚡ SUPER QUICK USAGE:

   1. Launch the app
   2. Select model: models/best.pt
   3. Choose "Windowed Mode"
   4. Click "Start Detection"
   5. Point at FaB cards!

=============================================================

💡 TIPS:

   - Use Windowed Mode for testing
   - Use Overlay Mode while playing
   - Lower confidence to 0.25 if cards not detected
   - Works on top 100 most popular cards

=============================================================

🎯 SYSTEM REQUIREMENTS:

   - Windows 10/11 (64-bit)
   - 4GB RAM (8GB recommended)
   - 250MB disk space
   - Internet (first install only)

=============================================================

Enjoy detecting! 🃏
EOF
echo "  ✓ Created QUICKSTART.txt"

# Calculate package size
echo ""
echo "Calculating package size..."
UNCOMPRESSED_SIZE=$(du -sh "$PACKAGE_DIR" | cut -f1)
echo "  Uncompressed size: $UNCOMPRESSED_SIZE"

# Create ZIP archive (Windows-friendly)
echo ""
echo "Creating ZIP archive..."
cd "packages"
ZIP_NAME="${PACKAGE_NAME}.zip"

# Use zip if available, otherwise use tar
if command -v zip &> /dev/null; then
    zip -r -q "$ZIP_NAME" "$(basename "$PACKAGE_DIR")"
else
    # Fallback to creating a tar.gz
    tar -czf "${PACKAGE_NAME}.tar.gz" "$(basename "$PACKAGE_DIR")"
    ZIP_NAME="${PACKAGE_NAME}.tar.gz"
fi

cd "$PROJECT_ROOT"

ARCHIVE_PATH="packages/$ZIP_NAME"
if [ -f "$ARCHIVE_PATH" ]; then
    ARCHIVE_SIZE=$(du -h "$ARCHIVE_PATH" | cut -f1)
    echo "  ✓ Created: $ZIP_NAME ($ARCHIVE_SIZE)"
else
    echo "  ✗ ERROR: Failed to create archive"
    exit 1
fi

# Create checksum
echo ""
echo "Creating checksum..."
cd "packages"
if command -v sha256sum &> /dev/null; then
    sha256sum "$ZIP_NAME" > "${ZIP_NAME}.sha256"
    echo "  ✓ Created SHA256 checksum"
elif command -v shasum &> /dev/null; then
    shasum -a 256 "$ZIP_NAME" > "${ZIP_NAME}.sha256"
    echo "  ✓ Created SHA256 checksum"
fi
cd "$PROJECT_ROOT"

# Summary
echo ""
echo "============================================================"
echo "   ✅ Windows Package Created Successfully!"
echo "============================================================"
echo ""
echo "Package Details:"
echo "  Name: $PACKAGE_NAME"
echo "  Location: $PACKAGE_DIR"
echo "  Archive: $ARCHIVE_PATH"
echo "  Uncompressed: $UNCOMPRESSED_SIZE"
echo "  Compressed: $ARCHIVE_SIZE"
echo ""
echo "Contents:"
echo "  ✓ INSTALL_WINDOWS.bat (one-click installer)"
echo "  ✓ RUN_DETECTOR.bat (quick launcher)"
echo "  ✓ QUICKSTART.txt (30-second guide)"
echo "  ✓ README_WINDOWS.txt (full documentation)"
echo "  ✓ fab_detector_app.py (main application)"
echo "  ✓ live_detector.py (CLI tool)"
echo "  ✓ models/best.pt (trained model)"
echo "  ✓ data/ (card metadata + classes)"
echo "  ✓ requirements.txt (dependency reference)"
echo ""
echo "User Experience:"
echo "  1. Download and extract ZIP"
echo "  2. Double-click INSTALL_WINDOWS.bat"
echo "  3. Done! App launches automatically"
echo ""
echo "After first install:"
echo "  - Use desktop shortcut: 'FaB Card Detector'"
echo "  - Or double-click: RUN_DETECTOR.bat"
echo ""
echo "Ready to upload to GitHub!"
echo "============================================================"
