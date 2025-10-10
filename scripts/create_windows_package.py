#!/usr/bin/env python3
"""
Create portable Windows package for FaB Card Detector
This creates a directory with all necessary files that can be zipped and distributed
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

def create_portable_package():
    """Create portable package without PyInstaller"""
    
    print("=" * 60)
    print("FaB Card Detector - Portable Windows Package Creator")
    print("=" * 60)
    
    project_root = Path(__file__).parent.parent.absolute()
    os.chdir(project_root)
    
    # Create timestamp for unique package name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    package_name = f"FaBCardDetector_Windows_{timestamp}"
    package_dir = project_root / "packages" / package_name
    
    # Clean and create package directory
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCreating package: {package_dir}")
    
    # Copy main application files
    print("\n📦 Copying application files...")
    app_files = [
        "fab_detector_app.py",
        "live_detector.py",
        "requirements.txt"
    ]
    
    for file in app_files:
        src = project_root / file
        if src.exists():
            shutil.copy2(src, package_dir / file)
            print(f"  ✅ {file}")
    
    # Copy model weights
    print("\n📦 Copying model weights...")
    models_dir = package_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    model_phase2 = project_root / "models" / "phase2_best.pt"
    model_phase1 = project_root / "runs" / "train" / "phase1_100classes" / "weights" / "best.pt"
    
    if model_phase2.exists():
        shutil.copy2(model_phase2, models_dir / "best.pt")
        model_info = "Phase 2 (500 classes)"
        print(f"  ✅ Phase 2 model weights (500 classes)")
    elif model_phase1.exists():
        shutil.copy2(model_phase1, models_dir / "best.pt")
        model_info = "Phase 1 (100 classes)"
        print(f"  ✅ Phase 1 model weights (100 classes)")
    else:
        print(f"  ⚠️  Warning: No model weights found!")
        model_info = "Not found"
    
    # Copy data files
    print("\n📦 Copying data files...")
    data_dir = package_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Copy Phase 2 classes if available
    classes_phase2 = project_root / "data" / "phase2_classes.yaml"
    classes_default = project_root / "data" / "classes.yaml"
    
    if classes_phase2.exists():
        shutil.copy2(classes_phase2, data_dir / "classes.yaml")
        print(f"  ✅ Phase 2 classes.yaml (500 classes)")
    elif classes_default.exists():
        shutil.copy2(classes_default, data_dir / "classes.yaml")
        print(f"  ✅ classes.yaml")
    
    # Copy other data files
    data_files = [
        "card.json",
        "card_popularity_weights.json"
    ]
    
    for file in data_files:
        src = project_root / "data" / file
        if src.exists():
            shutil.copy2(src, data_dir / file)
            print(f"  ✅ {file}")
    
    # Create batch files for Windows
    print("\n📦 Creating Windows batch files...")
    
    # Install script
    install_bat = package_dir / "INSTALL_WINDOWS.bat"
    install_bat.write_text("""@echo off
echo ========================================
echo FaB Card Detector - Installation
echo ========================================
echo.
echo This will install required Python packages.
echo Make sure you have Python 3.8+ installed!
echo.
pause

python --version
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3.8 or higher.
    pause
    exit /b 1
)

echo.
echo Installing packages...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Run RUN_DETECTOR.bat to start the application.
pause
""")
    print(f"  ✅ INSTALL_WINDOWS.bat")
    
    # Run script
    run_bat = package_dir / "RUN_DETECTOR.bat"
    run_bat.write_text("""@echo off
echo ========================================
echo FaB Card Detector - Starting...
echo ========================================
echo.

python fab_detector_app.py

if errorlevel 1 (
    echo.
    echo ERROR: Application failed to start!
    echo Make sure you ran INSTALL_WINDOWS.bat first.
    pause
)
""")
    print(f"  ✅ RUN_DETECTOR.bat")
    
    # Create README
    print("\n📦 Creating documentation...")
    
    readme = package_dir / "README_WINDOWS.txt"
    readme.write_text(f"""
========================================
FaB Card Detector - Windows Package
========================================

Model: {model_info}
Package Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

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

""")
    print(f"  ✅ README_WINDOWS.txt")
    
    # Create quickstart
    quickstart = package_dir / "QUICKSTART.txt"
    quickstart.write_text("""
QUICK START GUIDE
==================

Step 1: Double-click INSTALL_WINDOWS.bat
Step 2: Wait for installation (5-10 minutes)
Step 3: Double-click RUN_DETECTOR.bat
Step 4: Click "Start Detection" in the GUI

That's it!

""")
    print(f"  ✅ QUICKSTART.txt")
    
    # Calculate package size
    total_size = sum(f.stat().st_size for f in package_dir.rglob('*') if f.is_file())
    size_mb = total_size / (1024 * 1024)
    
    print(f"\n" + "=" * 60)
    print(f"✅ Package created successfully!")
    print(f"=" * 60)
    print(f"\nLocation: {package_dir}")
    print(f"Size: {size_mb:.1f} MB")
    print(f"Model: {model_info}")
    print(f"\nTo distribute:")
    print(f"  1. Zip the entire folder: {package_name}")
    print(f"  2. Send to Windows users")
    print(f"  3. They run INSTALL_WINDOWS.bat then RUN_DETECTOR.bat")
    print(f"\n" + "=" * 60)
    
    return True

if __name__ == "__main__":
    create_portable_package()
