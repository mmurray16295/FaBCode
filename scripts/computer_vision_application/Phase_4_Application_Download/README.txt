================================================================================
           FaB Card Detector - Phase 4 Application
================================================================================

PACKAGE CONTENTS:
  - Phase4_part_aa (90 MB)
  - Phase4_part_ab (90 MB)
  - Phase4_part_ac (14 MB)
  - REASSEMBLE_PHASE4.bat
  - This README

TOTAL SIZE: ~194 MB (includes 344MB YOLO11x model)

================================================================================
                         QUICK START
================================================================================

1. DOWNLOAD ALL FILES
   Make sure you have all 3 parts (aa, ab, ac) and the reassembly script
   in the SAME FOLDER.

2. REASSEMBLE
   Double-click: REASSEMBLE_PHASE4.bat
   This creates: FaBCardDetector_Phase4_20251024.zip

3. EXTRACT
   Right-click the ZIP -> Extract All
   Choose a location (Desktop, Documents, etc.)

4. INSTALL
   In extracted folder, double-click: INSTALL.bat
   Wait for packages to install (takes a few minutes)

5. RUN
   Double-click: RUN.bat
   Application starts with camera overlay

================================================================================
                      WHAT'S NEW IN PHASE 4
================================================================================

* TWO-HERO DETECTION
  Automatically detects both heroes with dynamic threshold adjustment
  Starts at 0.65 confidence, lowers by 0.10/second to 0.07 if needed

* VISUAL RESET BUTTON
  Click the reset button on overlay or press F9 to clear detections

* MANUAL PREVIEW POSITIONING
  Choose left or right side for card preview in manual mode

* IMPROVED UI
  Fixed slider sync, better hero status display, cleaner layout

================================================================================
                        SYSTEM REQUIREMENTS
================================================================================

Required:
  - Windows 10/11
  - Python 3.8 or higher
  - 8GB RAM (16GB recommended)
  - Webcam (USB or built-in)
  - 1GB free disk space

Optional (for better performance):
  - NVIDIA GPU with CUDA support

================================================================================
                         TROUBLESHOOTING
================================================================================

PROBLEM: Python not recognized
SOLUTION: Install Python from python.org, check "Add to PATH" option

PROBLEM: Camera won't open
SOLUTION: Close other apps using camera (Zoom, Skype, etc.)

PROBLEM: Model loading error
SOLUTION: Verify models/best.pt exists and is 344 MB

PROBLEM: Poor detection
SOLUTION: Improve lighting, wait for dynamic threshold to adjust

PROBLEM: Application crashes
SOLUTION: Run INSTALL.bat again, update GPU drivers if applicable

================================================================================
                            SUPPORT
================================================================================

For detailed instructions, see: DOWNLOAD_INSTRUCTIONS.md
For issues or questions, visit the GitHub repository

================================================================================
