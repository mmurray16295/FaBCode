# FaB Card Detector - Windows Installation Guide

## 🚀 One-Click Installation (Recommended)

### Super Easy Setup:

1. **Download** the package from GitHub
2. **Extract** the ZIP file
3. **Double-click** `INSTALL_WINDOWS.bat`
4. **Done!** The app launches automatically

That's it! The installer handles everything automatically.

---

## 📦 What the Installer Does

The installer is smart and lightweight:

### First Time Setup (2-3 minutes):
1. ✅ Checks if Python is installed
   - If missing: Downloads embedded Python (~20MB)
   - Uses your existing Python if available
2. ✅ Installs required packages (~150MB total)
   - PyTorch (CPU version - lightweight)
   - OpenCV, Ultralytics, MSS
3. ✅ Creates desktop shortcut "FaB Card Detector"
4. ✅ Launches the app automatically

### After First Install:
- **Just double-click** the desktop shortcut
- Or double-click `RUN_DETECTOR.bat`
- App launches in 1-2 seconds

---

## 💾 Download Size

**Total Download:** ~12MB (compressed package)

**After Installation:** ~200MB total
- Package: 12MB (model + scripts + data)
- Python (if needed): 20MB
- Dependencies: ~150MB (PyTorch, OpenCV, etc.)

---

## 🎯 No Bloat, No Hassle

This installer is designed to be:
- ✅ **Lightweight**: Only installs what's needed
- ✅ **Fast**: 2-3 minutes first run, instant after
- ✅ **Smart**: Uses existing Python if available
- ✅ **Clean**: Everything stays in one folder
- ✅ **Portable**: Can move folder anywhere
- ✅ **User-friendly**: One double-click setup

---

## 📋 System Requirements

- **OS**: Windows 10/11 (64-bit)
- **RAM**: 4GB minimum, 8GB recommended
- **Disk Space**: 250MB free
- **Display**: 1920x1080 or higher
- **Internet**: Required for first-time setup only

---

## 🔧 What Gets Installed

### If Python Not Found:
- **Python 3.11 Embedded** (20MB, portable)
- Stays in the package folder
- Doesn't affect system Python

### Required Packages:
- **PyTorch** (CPU version) - AI/ML framework
- **OpenCV** - Computer vision
- **Ultralytics** - YOLO detection
- **MSS** - Screen capture
- **Pillow, NumPy, PyYAML** - Utilities

---

## ⚡ Quick Start After Install

1. **Launch from desktop shortcut** or `RUN_DETECTOR.bat`
2. **Select model**: `models/best.pt` (auto-detected)
3. **Choose mode**: Windowed (for testing) or Overlay (for gaming)
4. **Click "Start Detection"**
5. **Point at cards** and watch it detect!

---

## 🎮 Detection Modes

### Windowed Mode (Best for Testing)
- Shows full screen with detection boxes
- Green boxes around detected cards
- Hover to see full card image
- FPS and card count displayed

### Transparent Overlay Mode (Best for Gaming)
- Invisible window overlay
- Only shows card preview on hover
- Minimal interference
- Perfect for online matches

---

## 🖥️ Multi-Monitor Setup

Perfect for dual screens:
1. Set **Capture Monitor** to your game screen (#1)
2. Set **Display Monitor** to your other screen (#2)
3. See detections without blocking your game

---

## 📊 Performance

**Detection Speed:**
- **CPU Only**: 10-20 FPS
- **With GPU**: 40-80 FPS (requires NVIDIA GPU + CUDA)

**Accuracy:**
- **mAP50**: 99.2% (excellent!)
- **Classes**: Top 100 most popular FaB cards

---

## ❓ Troubleshooting

### Installer won't run
- **Right-click** `INSTALL_WINDOWS.bat` → "Run as Administrator"
- Check antivirus isn't blocking it

### "Python not found" error
- The installer should handle this automatically
- If it fails, install Python from python.org first

### App won't start after install
- Make sure `models/best.pt` exists
- Make sure `data/card.json` exists
- Check `data/classes.yaml` is present

### No detections showing
- Lower confidence threshold to 0.25
- Ensure good lighting
- Cards must be from top 100 (see `data/classes.yaml`)

### Slow performance
- Close other applications
- The CPU version is slower than GPU
- Normal range: 10-20 FPS

---

## 🔄 Updating

To update to a new version:
1. Download new package
2. Extract to **same folder** (overwrite files)
3. Run `INSTALL_WINDOWS.bat` again
4. Done!

---

## 🗑️ Uninstalling

To completely remove:
1. Delete the package folder
2. Delete desktop shortcut
3. That's it! Nothing installed system-wide

---

## 📦 Package Contents

```
FaBCardDetector/
├── INSTALL_WINDOWS.bat    <- Double-click this first time
├── RUN_DETECTOR.bat        <- Use this to launch later
├── fab_detector_app.py     <- Main application
├── live_detector.py        <- CLI tool (optional)
├── models/
│   └── best.pt             <- Trained model (11MB)
├── data/
│   ├── card.json           <- Card metadata (19MB)
│   ├── classes.yaml        <- Top 100 cards list
│   └── card_popularity_weights.json
└── README_WINDOWS.txt      <- This file
```

---

## 🌟 Features

✅ Detects top 100 most popular FaB cards  
✅ Real-time detection (10-60 FPS)  
✅ Two detection modes (windowed/overlay)  
✅ Multi-monitor support  
✅ Card hover preview with full image  
✅ Adjustable confidence threshold  
✅ FPS counter and detection stats  
✅ Settings persistence (remembers preferences)  
✅ Portable (no system-wide installation)  

---

## 🆘 Need Help?

**GitHub Repository:**  
https://github.com/mmurray16295/FaBCode

**Common Issues:**  
Check the troubleshooting section above

**Model Performance:**  
99.2% accuracy on top 100 cards

---

## 📝 Version Info

- **Model**: Phase 1 (100 classes)
- **Build Date**: October 9, 2025
- **Accuracy**: 99.2% mAP50
- **Python**: 3.11+ required
- **Platform**: Windows 10/11 (64-bit)

---

## 🎉 Enjoy!

You're all set! The FaB Card Detector is ready to help you identify cards in real-time.

Perfect for:
- 📱 Identifying cards from photos
- 🎮 Playing online with card detection
- 📚 Sorting your collection
- 🎓 Learning card names and sets

Have fun detecting! 🃏
