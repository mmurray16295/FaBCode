
========================================
FaB Card Detector - Windows Package
========================================

Model: Phase 2 (500 classes)
Package Date: 2025-10-10 11:39:35

INSTALLATION
============

1. Make sure you have Python 3.8 or higher installed
   Download from: https://www.python.org/downloads/
   
2. Double-click INSTALL_WINDOWS.bat
   This will install all required packages

3. Wait for installation to complete

RUNNING THE APPLICATION
=======================

1. Double-click RUN_DETECTOR.bat
2. The GUI will open automatically
3. Configure your settings:
   - Model file: models/best.pt (pre-selected)
   - Choose Windowed or Overlay mode
   - Adjust confidence threshold (default: 0.69)
   - Enable/disable card preview on hover
4. Click "Start Detection"

FEATURES
========

• Windowed Mode: Shows captured screen with detection boxes
• Overlay Mode: Transparent overlay for streaming (with detection boxes)
• Card Preview: Hover over detected cards to see full image
• Multi-monitor Support: Capture from one monitor, display on another
• Adjustable Confidence: Fine-tune detection sensitivity

TROUBLESHOOTING
===============

Q: Python not found error
A: Install Python from python.org and make sure to check "Add Python to PATH" during installation

Q: Module not found errors
A: Run INSTALL_WINDOWS.bat again

Q: Card preview not working
A: Make sure data/card.json exists and card names match your model

Q: Low detection accuracy
A: Try adjusting the confidence threshold (lower = more detections, higher = fewer false positives)

SYSTEM REQUIREMENTS
===================

- Windows 10 or 11
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- Webcam or screen capture support
- Internet connection (for initial setup only)

WHAT'S INCLUDED
===============

Files:
- fab_detector_app.py: Main application (GUI)
- live_detector.py: Detection engine
- models/best.pt: Trained YOLO model
- data/: Card metadata and class information
- requirements.txt: Python dependencies

For support or issues, visit: https://github.com/mmurray16295/FaBCode

