========================================
FaB CARD DETECTOR - PHASE 4
========================================

Package Date: 2025-10-24
Version: Phase 4 (Two-Hero Support with Dynamic Detection)

========================================
QUICK START
========================================

1. Install Python 3.10 or newer from https://www.python.org/downloads/
   WARNING: IMPORTANT - Check "Add Python to PATH" during installation!

2. Double-click INSTALL.bat
   - Wait 5-10 minutes for installation
   - This installs all required packages

3. Double-click RUN.bat
   - The GUI will appear

4. Configure settings in the GUI:
   - Select monitors for capture and display
   - Choose detection mode (Window or Overlay)
   - Set hero names (or use Auto Detect)

5. Click "Start Detection"

========================================
NEW FEATURES IN PHASE 4
========================================

* TWO-HERO SUPPORT
   - Automatically detects up to 2 heroes
   - Filters cards to show only legal plays for both heroes
   - Dynamic threshold that adapts to find heroes

* IMPROVED HERO DETECTION
   - Starts at 65% confidence
   - Lowers by 10% every second until heroes found
   - Minimum threshold of 7%
   - Requires 5 seconds of consistent detection

* RESET BUTTON
   - Visual button on overlay to reset hero detection
   - Also supports F9 hotkey
   - Positioned under hero names display

* MANUAL PREVIEW POSITIONING
   - Choose left or right side for manual card preview
   - Mirrored positioning for right-side display
   - Maintains Y position (59px from bottom)

* BETTER UI
   - Hero1 and Hero2 status display on overlay
   - Threshold values sync correctly on startup
   - Improved settings organization

========================================
CONTROLS
========================================

DETECTION MODES:
  - Window Mode: Separate detection window
  - Overlay Mode: Transparent overlay on capture monitor

KEYBOARD SHORTCUTS:
  - M: Toggle Manual Mode (click to select cards)
  - F9: Reset hero auto-detection
  - Q: Quit application

MOUSE:
  - Hover over detected cards to see full preview
  - Click "Reset Heroes" button to restart detection
  - In Manual Mode: Click cards to display for 15 seconds

========================================
SYSTEM REQUIREMENTS
========================================

- Windows 10/11
- Python 3.10 or newer
- 4GB RAM minimum (8GB recommended)
- Dual monitor setup recommended
- Internet connection for initial setup

========================================
FILES INCLUDED
========================================

fab_detector_app.py      - Main application
phase2_best.pt           - YOLO model weights (~182MB)
data/card.json           - Card database
data/card_name_to_class_id.json - Class mappings
requirements.txt         - Python dependencies
INSTALL.bat              - Automated installer
RUN.bat                  - Application launcher
README.txt               - This file

========================================
TROUBLESHOOTING
========================================

"Python not found":
  -> Install Python and check "Add to PATH"
  -> Restart computer after Python installation

"Module not found":
  -> Run INSTALL.bat again
  -> Try: pip install --upgrade -r requirements.txt

"No cards detected":
  -> Adjust Confidence Threshold slider (try 0.3-0.7)
  -> Ensure cards are clearly visible on capture monitor
  -> Check monitor selection is correct

"Heroes not detected":
  -> Wait for threshold to lower automatically
  -> Ensure hero cards are clearly visible
  -> Try clicking "Reset Heroes" button

"Overlay not working":
  -> Try Window Mode first
  -> Check display monitor selection
  -> Disable click-through mode to move window

========================================
SUPPORT
========================================

For issues, feature requests, or questions:
GitHub: https://github.com/mmurray16295/FaBCode

========================================
