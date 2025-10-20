# FaB Card Detector Phase 3 - Download Instructions

## ⚠️ SPLIT ARCHIVE - Assembly Required

Due to GitHub's 100MB file size limit, the Phase 3 application (183MB) has been split into 3 parts:

- `Phase3_part_aa` (90MB)
- `Phase3_part_ab` (90MB) 
- `Phase3_part_ac` (2.2MB)

### 📥 Windows Download & Assembly

1. Download all 3 part files to the same folder
2. Double-click `REASSEMBLE_PHASE3.bat`
3. Extract the resulting `FaBCardDetector_Phase3_20251020.zip`
4. Run `INSTALL_WINDOWS.bat` in the extracted folder
5. Run `RUN_DETECTOR.bat` to launch the application

### 📥 Linux/Mac Download & Assembly

```bash
# Download all 3 parts to the same directory
# Then reassemble:
cat Phase3_part_aa Phase3_part_ab Phase3_part_ac > FaBCardDetector_Phase3_20251020.zip

# Extract:
unzip FaBCardDetector_Phase3_20251020.zip

# Run:
cd FaBCardDetector_Application_Phase3
python fab_detector_app.py
```

## 📦 What's Included

- **Model**: YOLOv11x trained on 2,641 FaB card classes
- **Size**: 345MB model + supporting files = 370MB total (183MB compressed)
- **Accuracy**: 90.5% mAP (Epoch 27 checkpoint)
- **Training**: 125,000 synthetic playmat images

## 🎯 System Requirements

- Python 3.8+
- 8GB RAM minimum (16GB recommended)
- GPU recommended but not required
- Windows 10/11, Linux, or macOS

## 📖 Documentation

Full documentation included in the package:
- `README.txt` - Quick start guide
- `MODEL_INFO.txt` - Training details
- `README_WINDOWS.txt` - Windows-specific instructions

---

**Package Date**: October 20, 2025  
**Model Checkpoint**: Epoch 27/30 (best performance)  
**Repository**: https://github.com/mmurray16295/FaBCode
