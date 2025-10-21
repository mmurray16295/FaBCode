# Computer Vision Application Scripts

Scripts for packaging and deploying the FaB Card Detector application with trained YOLO models.

## Directory Structure

```
computer_vision_application/
├── README.md                      # This file
├── screen_detect.py              # Main detector application with overlay UI
├── build_executable.py           # PyInstaller executable builder
├── create_windows_package.py     # Portable Windows package creator
├── package_model.sh              # Package model + detector for local testing
└── package_windows.sh            # Windows package builder (bash)
```

## Scripts Overview

### Main Application

**screen_detect.py** (443 lines)
- Real-time YOLO-based card detection with overlay UI
- Captures screen via MSS or processes video files
- Shows minimal overlay window with detected card preview
- Prevents "feedback" detection loops by masking overlay region
- Usage: `python screen_detect.py [options]`

### Packaging Scripts

**build_executable.py** (315 lines)
- Creates standalone executable using PyInstaller
- Platform-specific builds (Windows/Linux)
- Bundles all dependencies and model weights
- Usage: `python build_executable.py`
- Output: Single executable file

**create_windows_package.py** (176 lines)
- Creates portable Windows directory package
- No compilation needed - includes Python interpreter
- Easier to debug and modify than executable
- Usage: `python create_windows_package.py`
- Output: `packages/FaBCardDetector_Windows_YYYYMMDD_HHMMSS/`

**package_model.sh** (Bash)
- Packages trained model with detector for local testing
- Includes model weights, detector script, and card data
- Usage: `bash package_model.sh`
- Output: `packages/fab_card_detector_YYYYMMDD_HHMMSS/`

**package_windows.sh** (Bash)
- Complete Windows package builder
- Bundles everything needed for Windows deployment
- Usage: `bash package_windows.sh`
- Output: `packages/FaBCardDetector_Windows_YYYYMMDD_HHMMSS/`

## Typical Workflow

### Development Testing
```bash
# Test detector directly
python screen_detect.py --weights runs/train/best.pt

# Quick package for local testing
bash package_model.sh
```

### Windows Deployment
```bash
# Option 1: Portable package (recommended for debugging)
python create_windows_package.py

# Option 2: Standalone executable (recommended for distribution)
python build_executable.py
```

### Linux/RunPod Deployment
```bash
# Package with model
bash package_model.sh

# Transfer to target system
scp -r packages/fab_card_detector_* user@target:/path/
```

## Requirements

**For packaging:**
- PyInstaller (for build_executable.py)
- All dependencies from requirements.txt

**For runtime:**
- ultralytics (YOLO)
- opencv-python
- mss (screen capture)
- PIL/Pillow
- requests

## Output Locations

All packaging scripts output to:
```
FaBCode/
└── packages/
    ├── FaBCardDetector_Windows_*/  # Windows packages
    └── fab_card_detector_*/        # Model packages
```

## Notes

- **Model Weights**: Packaging scripts automatically include model weights from `runs/train/*/weights/best.pt`
- **Card Data**: Includes `card.json` and `card_name_to_class_id.json` for card lookup
- **Screen Capture**: Requires `mss` library (Linux/Windows compatible)
- **Overlay UI**: Uses OpenCV for window management and rendering
