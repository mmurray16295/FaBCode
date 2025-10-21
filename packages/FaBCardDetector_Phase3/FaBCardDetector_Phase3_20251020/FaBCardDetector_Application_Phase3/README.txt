
========================================
FaB Card Detector - Phase 3 Application
========================================

🎯 COMPLETE COLLECTION DETECTOR
   2,641 Flesh and Blood Card Classes
   YOLOv11x Model (345MB)
   90.5% mAP Accuracy

📦 PACKAGE CONTENTS
   ├── models/best.pt (345MB) - Trained YOLOv11x checkpoint
   ├── data/classes.yaml - 2,641 card class names
   ├── data/card.json - Card metadata
   ├── fab_detector_app.py - GUI application
   ├── live_detector.py - Detection engine
   └── detector_config.json - Default settings

⚙️ QUICK START
   
   Windows:
   1. Run INSTALL_WINDOWS.bat (first time only)
   2. Run RUN_DETECTOR.bat
   
   Linux/Mac:
   1. pip install -r requirements.txt
   2. python fab_detector_app.py

📋 SYSTEM REQUIREMENTS
   - Python 3.8 or higher
   - 8GB RAM minimum (16GB recommended)
   - GPU highly recommended (CUDA-compatible)
   - Windows 10/11, Linux, or macOS

🔧 DEFAULT SETTINGS
   - Confidence Threshold: 0.25 (adjustable in GUI)
   - IOU Threshold: 0.45
   - Detection Modes: Windowed or Transparent Overlay
   - Multi-monitor support

📊 MODEL PERFORMANCE (Epoch 27)
   - Training Images: 125,000 synthetic playmats
   - Validation mAP50-95: 90.5%
   - Model Type: YOLOv11x
   - Parameters: 56.9M
   - Training Duration: October 18-20, 2025

🎮 FEATURES
   ✓ Real-time card detection
   ✓ Multi-monitor capture and display
   ✓ Card preview on hover
   ✓ Adjustable confidence thresholds
   ✓ Windowed and overlay modes
   ✓ Performance metrics display

⚡ USAGE TIPS
   - Start with confidence 0.25 and adjust based on results
   - Use GPU for better performance (10-30 FPS)
   - CPU mode works but slower (2-5 FPS)
   - Good lighting improves accuracy
   - Minimize card overlap for best detection

📝 CHANGELOG FROM PHASE 2
   - Upgraded from 500 to 2,641 classes (complete collection)
   - New YOLOv11x model (from YOLOv8)
   - Improved training on 125K synthetic images
   - Reduced default confidence from 0.69 to 0.25
   - Added MODEL_INFO.txt for detailed specs

🐛 TROUBLESHOOTING
   
   Low FPS / Slow Performance:
   - Enable GPU acceleration if available
   - Reduce image quality in settings
   - Close other GPU-intensive applications
   
   No Detections:
   - Lower confidence threshold (try 0.15-0.20)
   - Ensure good lighting on cards
   - Check that model file loaded correctly
   
   False Positives:
   - Raise confidence threshold (try 0.35-0.50)
   - Adjust IOU threshold (try 0.50-0.60)
   
   Application Won't Start:
   - Verify Python 3.8+ installed: python --version
   - Re-run INSTALL_WINDOWS.bat
   - Check antivirus isn't blocking Python

📖 DOCUMENTATION
   - README_WINDOWS.txt - Full Windows installation guide
   - MODEL_INFO.txt - Detailed model specifications
   - QUICKSTART.txt - Minimal setup instructions

💻 GITHUB
   https://github.com/mmurray16295/FaBCode

📅 VERSION INFO
   Package: Phase 3 (Complete Collection)
   Model Date: October 20, 2025
   Training Checkpoint: Epoch 27/30
   Classes: 2,641 FaB cards

