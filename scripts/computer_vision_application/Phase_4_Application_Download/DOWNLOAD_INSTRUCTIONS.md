# FaB Card Detector - Phase 4 Application Download

## Overview
Phase 4 of the FaB Card Detector brings two-hero support with dynamic detection, visual reset controls, and flexible preview positioning. This package contains the complete application with the full 344MB YOLO11x model for maximum detection accuracy.

## What's New in Phase 4

### Two-Hero Detection System
- **Dual Hero Support**: Detects and tracks two heroes simultaneously
- **Dynamic Threshold Adjustment**: Starts at 0.65 confidence, automatically lowers by 0.10 every second until reaching 0.07 or detecting both heroes
- **Smart Detection**: Requires 5 seconds of consistent detection with 10+ detections for hero confirmation
- **Hero Status Display**: Shows "Hero1: [Name]" and "Hero2: [Name]" on overlay instead of FPS counter

### Visual Reset Button
- **On-Screen Button**: Click the reset button on the overlay to clear all detected cards and heroes
- **Keyboard Shortcut**: Press F9 to reset (same as before)
- **Visual Feedback**: Button highlights on hover for better usability

### Manual Preview Positioning
- **Left/Right Toggle**: Choose whether manual card preview appears on left (default) or right side
- **Consistent Positioning**: 50px from edge, 59px from bottom
- **Control Panel Option**: Toggle in Display Settings section

### Enhanced Configuration
- **Fixed Slider Sync**: Confidence and IOU threshold sliders now correctly display their values on startup
- **Improved UI**: Cleaner layout with better organization of controls

## Download Files

This package is split into multiple parts for Git compatibility. You need to download **all parts**:

- **Phase4_part_aa** (90 MB)
- **Phase4_part_ab** (90 MB)
- **Phase4_part_ac** (14 MB)
- **REASSEMBLE_PHASE4.bat** (reassembly script)

**Total Size**: ~194 MB

## Installation Instructions

### Step 1: Download All Files
Download all the files listed above to the **same folder** on your computer.

### Step 2: Reassemble the Package
1. Navigate to the folder containing all downloaded files
2. Double-click **REASSEMBLE_PHASE4.bat**
3. This will combine the parts into `FaBCardDetector_Phase4_20251024.zip`
4. You can delete the `Phase4_part_*` files after successful reassembly

### Step 3: Extract the Package
1. Right-click `FaBCardDetector_Phase4_20251024.zip`
2. Select "Extract All..."
3. Choose a location (e.g., your Desktop or Documents folder)
4. Extract the files

### Step 4: Install Dependencies
1. Navigate to the extracted `FaBCardDetector_Phase4_20251024` folder
2. Double-click **INSTALL.bat**
3. Wait for Python packages to install (this may take a few minutes)
4. A success message will appear when complete

### Step 5: Run the Application
1. In the same folder, double-click **RUN.bat**
2. The application will start and open the camera overlay
3. Position your heroes and cards in view

## System Requirements

### Required
- **Operating System**: Windows 10/11
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum (16GB recommended for best performance)
- **Webcam**: Any USB or built-in camera
- **GPU**: NVIDIA GPU with CUDA support (optional, but significantly improves performance)

### Python Packages (automatically installed)
- ultralytics >= 8.0.0
- opencv-python >= 4.8.0
- Pillow >= 10.0.0
- PyYAML
- numpy
- torch (with CUDA support if available)
- keyboard

## Package Contents

```
FaBCardDetector_Phase4_20251024/
├── fab_detector_app.py          (Main application - 180KB)
├── models/
│   └── best.pt                  (YOLO11x model - 344MB)
├── data/
│   ├── card.json               (Card database - 60KB)
│   └── card_name_to_class_id.json  (Class mappings - 8KB)
├── INSTALL.bat                  (Dependency installer)
├── RUN.bat                      (Application launcher)
├── requirements.txt             (Python dependencies)
└── README.txt                   (Quick reference)
```

## Quick Start Guide

### First Time Setup
1. Download all 3 parts + reassembly script
2. Run REASSEMBLE_PHASE4.bat
3. Extract the ZIP file
4. Run INSTALL.bat
5. Run RUN.bat

### Daily Use
- Just run RUN.bat to start the application
- Position heroes in camera view first (detection adapts automatically)
- Move cards into view for detection
- Use reset button or F9 to clear and start fresh

## Application Controls

### Keyboard Shortcuts
- **F9**: Reset all detections (heroes and cards)
- **ESC**: Exit application
- **Q**: Exit application (alternate)

### Mouse Controls
- **Click Reset Button**: Clear all detections (on-screen button in overlay)
- **Drag Sliders**: Adjust confidence and IOU thresholds in control panel

### Control Panel Options

**Detection Mode**:
- Auto: Automatically detects cards as they appear
- Manual: Requires spacebar confirmation for each card

**Display Settings**:
- Show Overlay: Toggle detection overlay on/off
- Show Boxes: Toggle bounding boxes around detected cards
- Manual Preview Position: Left (default) or Right

**Thresholds**:
- Confidence: Detection confidence threshold (default: 0.69)
- IOU: Intersection over Union threshold for duplicate removal (default: 0.50)

## Troubleshooting

### "Python is not recognized" Error
**Solution**: Install Python 3.8+ from python.org and ensure "Add Python to PATH" is checked during installation.

### Camera Not Opening
**Solution**: 
- Check if another application is using the camera
- Try closing other camera apps (Zoom, Skype, etc.)
- Restart the application

### Model Loading Errors
**Solution**: 
- Ensure `models/best.pt` exists (344 MB file)
- Check available disk space (need at least 1GB free)
- Verify file wasn't corrupted during download/extraction

### Poor Detection Performance
**Solution**: 
- Improve lighting conditions
- Ensure cards are fully visible and not overlapping
- Lower confidence threshold if cards aren't being detected
- Wait for dynamic hero detection to stabilize (starts at 0.65, lowers to 0.07)

### Application Crashes on Startup
**Solution**: 
- Run INSTALL.bat again to reinstall dependencies
- Check if CUDA/GPU drivers are up to date (for NVIDIA GPUs)
- Try disabling GPU acceleration by editing `fab_detector_app.py` (change `device='0'` to `device='cpu'`)

### Reset Button Not Working
**Solution**: 
- Ensure the overlay window has focus (click on it first)
- Try F9 keyboard shortcut as alternative
- Check if the button area is visible (not covered by other windows)

## Support

### Reporting Issues
If you encounter problems:
1. Note the exact error message
2. Check the troubleshooting section above
3. Open an issue on the GitHub repository with details

### Feature Requests
Have ideas for improvements? Open a feature request on the GitHub repository!

### Community
Join discussions and share your experience with other users on the project's GitHub Discussions page.

## Version History

### Phase 4 (October 2025)
- Added two-hero detection with dynamic threshold adjustment
- Implemented visual reset button with hover effects
- Added manual preview positioning (left/right toggle)
- Fixed slider label sync on startup
- Improved UI with hero status display
- Upgraded to 344MB YOLO11x model for better accuracy

### Phase 3 (October 2025)
- Initial packaged release
- Single hero support
- Basic card detection
- F9 reset hotkey
- Configuration panel

## License

This application is provided for personal use with Flesh and Blood TCG. All card images and data are property of Legend Story Studios.

---

**Ready to get started? Follow the installation instructions above!**
