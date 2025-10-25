#!/usr/bin/env python3
"""
Create Phase 4 Windows package for FaB Card Detector
Packages the current working application from FaBCardDetector_Windows_20251010_113935
"""

import os
import shutil
import zipfile
from pathlib import Path
from datetime import datetime

def create_phase4_package():
    """Create Phase 4 portable package"""
    
    print("=" * 60)
    print("FaB Card Detector - Phase 4 Package Creator")
    print("=" * 60)
    
    # Paths
    script_dir = Path(__file__).parent
    source_dir = Path("c:/VS Code/FaB Code/FaBCardDetector_Phase2_Windows/FaBCardDetector_Windows_20251010_113935")
    project_root = Path("c:/VS Code/FaB Code")
    
    # Package name with date
    package_date = datetime.now().strftime("%Y%m%d")
    package_name = f"FaBCardDetector_Phase4_{package_date}"
    temp_package_dir = script_dir / "temp_phase4_package" / package_name
    
    # Clean and create temp package directory
    if temp_package_dir.parent.exists():
        shutil.rmtree(temp_package_dir.parent)
    temp_package_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCreating package: {package_name}")
    print(f"Source: {source_dir}")
    
    # Copy main application file
    print("\n📦 Copying application files...")
    shutil.copy2(source_dir / "fab_detector_app.py", temp_package_dir / "fab_detector_app.py")
    print(f"  ✅ fab_detector_app.py")
    
    # Copy model weights (use the actual 344MB model being tested)
    print("\n📦 Copying model weights...")
    model_src = source_dir / "models" / "best.pt"
    (temp_package_dir / "models").mkdir(exist_ok=True)
    if model_src.exists():
        shutil.copy2(model_src, temp_package_dir / "models" / "best.pt")
        model_size = model_src.stat().st_size / (1024 * 1024)
        print(f"  ✅ best.pt ({model_size:.1f} MB)")
    else:
        print(f"  ⚠️  WARNING: Model not found at {model_src}")
    
    # Copy data files
    print("\n📦 Copying data files...")
    data_files = [
        "data/card.json",
        "data/card_name_to_class_id.json"
    ]
    
    (temp_package_dir / "data").mkdir(exist_ok=True)
    for file in data_files:
        src = project_root / file
        if src.exists():
            dest = temp_package_dir / file
            shutil.copy2(src, dest)
            print(f"  ✅ {file}")
        else:
            print(f"  ⚠️  WARNING: {file} not found")
    
    # Create requirements.txt
    print("\n📦 Creating requirements.txt...")
    requirements_content = """# FaB Card Detector Phase 4 Requirements
ultralytics>=8.0.0
opencv-python>=4.8.0
Pillow>=10.0.0
PyYAML>=6.0
mss>=9.0.0
pywin32>=306
keyboard>=0.13.5
"""
    (temp_package_dir / "requirements.txt").write_text(requirements_content)
    print(f"  ✅ requirements.txt")
    
    # Create INSTALL.bat
    print("\n📦 Creating installer script...")
    install_bat_content = """@echo off
echo ========================================
echo FaB Card Detector Phase 4 - Installation
echo ========================================
echo.
echo This will install all required Python packages.
echo Please ensure Python 3.10+ is installed and in PATH.
echo.
pause

echo.
echo Installing packages...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ========================================
echo Installation complete!
echo ========================================
echo.
echo Run RUN.bat to start the application.
echo.
pause
"""
    (temp_package_dir / "INSTALL.bat").write_text(install_bat_content)
    print(f"  ✅ INSTALL.bat")
    
    # Create RUN.bat
    print("\n📦 Creating launcher script...")
    run_bat_content = """@echo off
echo ========================================
echo FaB Card Detector Phase 4
echo ========================================
echo.
echo Starting application...
echo.

python fab_detector_app.py

echo.
echo Application closed.
pause
"""
    (temp_package_dir / "RUN.bat").write_text(run_bat_content)
    print(f"  ✅ RUN.bat")
    
    # Create README
    print("\n📦 Creating README...")
    readme_content = f"""========================================
FaB CARD DETECTOR - PHASE 4
========================================

Package Date: {datetime.now().strftime('%Y-%m-%d')}
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
"""
    (temp_package_dir / "README.txt").write_text(readme_content, encoding='utf-8')
    print(f"  ✅ README.txt")
    
    # Create ZIP file
    print(f"\n📦 Creating ZIP archive...")
    zip_path = script_dir / f"{package_name}.zip"
    if zip_path.exists():
        zip_path.unlink()
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in temp_package_dir.rglob('*'):
            if file.is_file():
                arcname = file.relative_to(temp_package_dir.parent)
                zipf.write(file, arcname)
                print(f"  Added: {arcname}")
    
    zip_size = zip_path.stat().st_size / (1024 * 1024)
    print(f"\n  ✅ Created {package_name}.zip ({zip_size:.1f} MB)")
    
    # Split into 90MB chunks for GitHub
    print(f"\n📦 Splitting into 90MB chunks...")
    chunk_size = 90 * 1024 * 1024  # 90MB
    
    with open(zip_path, 'rb') as f:
        chunk_num = 0
        while True:
            chunk_data = f.read(chunk_size)
            if not chunk_data:
                break
            
            # Use letter suffixes: aa, ab, ac, etc.
            suffix = chr(97 + chunk_num // 26) + chr(97 + chunk_num % 26)
            chunk_path = script_dir / f"Phase4_part_{suffix}"
            
            with open(chunk_path, 'wb') as chunk_file:
                chunk_file.write(chunk_data)
            
            chunk_size_mb = len(chunk_data) / (1024 * 1024)
            print(f"  ✅ Phase4_part_{suffix} ({chunk_size_mb:.2f} MB)")
            chunk_num += 1
    
    # Create reassemble batch file
    print(f"\n📦 Creating reassemble script...")
    
    # Build the copy command with all chunks
    chunk_parts = []
    for i in range(chunk_num):
        suffix = chr(97 + i // 26) + chr(97 + i % 26)
        chunk_parts.append(f"Phase4_part_{suffix}")
    
    reassemble_content = f"""@echo off
echo ========================================
echo Reassembling FaB Detector Phase 4
echo ========================================
echo.
echo This will combine the split files...
echo.

copy /b {" + ".join(chunk_parts)} {package_name}.zip

echo.
echo ========================================
echo Done! Extract {package_name}.zip
echo ========================================
pause
"""
    
    reassemble_path = script_dir / "REASSEMBLE_PHASE4.bat"
    reassemble_path.write_text(reassemble_content)
    print(f"  ✅ REASSEMBLE_PHASE4.bat")
    
    # Clean up temp directory
    print(f"\n📦 Cleaning up...")
    shutil.rmtree(temp_package_dir.parent)
    print(f"  ✅ Removed temporary files")
    
    # Calculate total package size
    total_size = sum(
        (script_dir / f"Phase4_part_{chr(97 + i // 26) + chr(97 + i % 26)}").stat().st_size 
        for i in range(chunk_num)
    )
    total_size_mb = total_size / (1024 * 1024)
    
    print(f"\n" + "=" * 60)
    print(f"✅ Phase 4 Package created successfully!")
    print(f"=" * 60)
    print(f"\nPackage: {package_name}")
    print(f"Total Size: {total_size_mb:.1f} MB")
    print(f"Split into {chunk_num} parts (90MB each)")
    print(f"\nFiles created in: {script_dir}")
    print(f"  - Phase4_part_aa, Phase4_part_ab, etc.")
    print(f"  - REASSEMBLE_PHASE4.bat")
    print(f"\nTo distribute:")
    print(f"  1. Commit all Phase4_part_* files to Git")
    print(f"  2. Commit REASSEMBLE_PHASE4.bat")
    print(f"  3. Users download all files")
    print(f"  4. Users run REASSEMBLE_PHASE4.bat")
    print(f"  5. Users extract {package_name}.zip")
    print(f"  6. Users run INSTALL.bat then RUN.bat")
    print(f"\n" + "=" * 60)
    
    return True

if __name__ == "__main__":
    try:
        create_phase4_package()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
