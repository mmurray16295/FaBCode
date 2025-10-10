========================================
FaB Card Detector - Windows Package
========================================

QUICK START (EASIEST)
=====================

1. Double-click INSTALL.bat
   - A professional GUI installer will open
   - Click "Install" button
   - Wait 5-10 minutes for completion
   - Will automatically install Python if needed (NO IDE OR CODE EDITOR REQUIRED!)

2. Double-click RUN.bat
   - The card detector application will launch
   - No console window clutter!

FEATURES
========

* Professional GUI installer with progress bar
* Automatic Python installation (no manual steps!)
* Hidden console windows (clean interface)
* Optional detailed output for troubleshooting
* Works on ANY Windows computer (10/11)
* NO PROGRAMMING KNOWLEDGE NEEDED!

INSTALLATION OPTIONS
====================

OPTION 1: GUI Installer (RECOMMENDED)
--------------------------------------
Double-click: INSTALL.bat
- Professional interface with progress bar
- Automatic Python detection and installation
- No console windows
- Perfect for end users
- NO IDE OR CODE EDITOR REQUIRED!
- The installer downloads and installs Python automatically

OPTION 2: Advanced Auto-Installer
----------------------------------
Right-click AUTO_INSTALL.bat -> "Run as Administrator"
- Command-line interface
- Automatic Python installation
- More detailed output
- For advanced users

OPTION 3: Manual Installation
------------------------------
1. Install Python 3.8+ from: https://www.python.org/downloads/
2. Double-click: INSTALL_WINDOWS_MANUAL.bat
3. For users who prefer manual control

TROUBLESHOOTING
===============

If INSTALL.bat does not work:
- Run CHECK_SYSTEM.bat to diagnose issues
- See TROUBLESHOOTING.txt for detailed help
- Try AUTO_INSTALL.bat as Administrator

RUNNING THE APPLICATION
=======================

1. Double-click RUN.bat (recommended)
   - Launches without console window
   - Clean, professional appearance

2. Or double-click RUN_DETECTOR.bat (shows console)
   - Useful for debugging
   - Shows detailed output

3. Configure settings in the GUI:
   - Model file: models/best.pt (pre-selected)
   - Choose Windowed or Overlay mode
   - Adjust confidence threshold (default: 0.69)
   - Enable/disable card preview on hover

4. Click "Start Detection"

APPLICATION FEATURES
====================

- Windowed Mode: Shows captured screen with detection boxes
- Overlay Mode: Transparent overlay for streaming (with detection boxes)
- Card Preview: Hover over detected cards to see full image
- Multi-monitor Support: Capture from one monitor, display on another
- Adjustable Confidence: Fine-tune detection sensitivity

COMMON QUESTIONS
================

Q: Do I need to know how to code?
A: NO! Just double-click INSTALL.bat and then RUN.bat. Everything is automatic.

Q: Do I need Visual Studio Code or any IDE?
A: NO! The installer will download and install Python automatically.
   You don't need any coding tools or editors.

Q: Python not found error?
A: Just run INSTALL.bat - it will automatically install Python for you!
   No IDE or code editor needed. The GUI installer handles everything.

Q: Module not found errors?
A: Run INSTALL.bat again or try AUTO_INSTALL.bat as Administrator

Q: Card preview not working?
A: Make sure data/card.json exists and card names match your model

Q: Low detection accuracy?
A: Try adjusting the confidence threshold (lower = more detections, higher = fewer false positives)

SYSTEM REQUIREMENTS
===================

- Windows 10 or 11 (64-bit)
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space
- Python 3.8+ (will be installed automatically if missing)
- Webcam or screen capture capability
- Internet connection (for initial Python download only)

WHAT'S INCLUDED
===============

- models/: Pre-trained YOLO detection model (500 card classes)
- data/: Card metadata and class information
- requirements.txt: Python dependencies
- GUI_INSTALLER.py: Automatic installer (installs Python if needed)
- INSTALL.bat: Simple double-click installation
- RUN.bat: Launch application without console window
- AUTO_INSTALL.ps1/bat: Advanced PowerShell installer
- CHECK_SYSTEM.py/bat: System diagnostic tool
- TROUBLESHOOTING.txt: Detailed help guide

For support or issues, visit: https://github.com/mmurray16295/FaBCode
