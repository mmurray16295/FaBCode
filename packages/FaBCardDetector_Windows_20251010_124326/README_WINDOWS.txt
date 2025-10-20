========================================
FaB Card Detector - Windows Package
========================================

Model: Phase 2 (500 classes)
Package Date: 2025-10-10 12:43:26

QUICK START
===========

STEP 1: Install Python (if you don't have it)
   - Go to: https://www.python.org/downloads/
   - Download Python 3.8 or higher
   - During installation, CHECK THE BOX "Add Python to PATH"
   - This is a one-time setup

STEP 2: Run INSTALL.bat
   - Double-click INSTALL.bat
   - Wait 5-10 minutes for packages to install
   - You only need to do this once

STEP 3: Run the application
   - Double-click RUN.bat
   - Configure your settings and click "Start Detection"

That's it!

SYSTEM REQUIREMENTS
===================

- Windows 10 or 11 (64-bit)
- Python 3.8 or higher (download from python.org)
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space
- Internet connection (for initial installation only)

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

Q: "Python not found" error
A: Install Python from https://www.python.org/downloads/
   Make sure to check "Add Python to PATH" during installation!

Q: Installation fails
A: 1. Check your internet connection
   2. Try running INSTALL.bat as Administrator (right-click -> Run as Administrator)
   3. Make sure you have Python 3.8 or higher

Q: Application won't start
A: Make sure INSTALL.bat completed successfully
   Check that Python is installed: Open Command Prompt and type: python --version

Q: "Module not found" errors
A: Run INSTALL.bat again

WHAT'S INCLUDED
===============

- models/best.pt: Pre-trained YOLO model (500 card classes)
- data/: Card metadata and class information
- fab_detector_app.py: Main application
- live_detector.py: Detection engine
- requirements.txt: Python dependencies
- INSTALL.bat: Package installer
- RUN.bat: Application launcher

For support or issues, visit: https://github.com/mmurray16295/FaBCode
