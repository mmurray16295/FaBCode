========================================
FaB Card Detector - Windows Package
========================================

Model: Phase 2 (500 classes)
Package Date: 2025-10-10 12:35:29

QUICK START
===========

1. Right-click INSTALL.ps1 and select "Run with PowerShell"
   - If prompted, click "Yes" to allow admin privileges
   - Wait 5-10 minutes for automatic installation
   - Python will be installed automatically if needed

2. Double-click RUN.bat to start the application

That's it!

WHAT DOES THE INSTALLER DO?
============================

The INSTALL.ps1 script automatically:
- Checks if Python 3.8+ is installed
- Downloads and installs Python 3.11.9 if needed
- Installs all required packages (PyTorch, OpenCV, etc.)
- No manual steps required!

SYSTEM REQUIREMENTS
===================

- Windows 10 or 11 (64-bit)
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space
- Internet connection (for initial installation only)
- Administrator privileges (for installation only)

APPLICATION FEATURES
====================

- 500 card detection classes
- Windowed Mode: Shows captured screen with detection boxes
- Overlay Mode: Transparent overlay for streaming
- Card Preview: Hover over detected cards to see full image
- Multi-monitor Support: Capture from one monitor, display on another
- Adjustable Confidence: Fine-tune detection sensitivity

TROUBLESHOOTING
===============

Q: "Running scripts is disabled on this system"
A: Right-click INSTALL.ps1 -> Properties -> Check "Unblock" -> Apply -> OK
   Then try running again

Q: Installation fails or hangs
A: Check your internet connection and try again
   Make sure you have administrator privileges

Q: Python already installed but packages fail
A: Open PowerShell as Administrator and run:
   python -m pip install -r requirements.txt

Q: Application won't start
A: Make sure INSTALL.ps1 completed successfully
   Check that Python is installed: python --version

WHAT'S INCLUDED
===============

- models/best.pt: Pre-trained YOLO model (500 card classes)
- data/: Card metadata and class information
- fab_detector_app.py: Main application
- live_detector.py: Detection engine
- requirements.txt: Python dependencies
- INSTALL.ps1: Automatic installer
- RUN.bat: Application launcher

For support or issues, visit: https://github.com/mmurray16295/FaBCode
