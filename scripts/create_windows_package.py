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
        "requirements.txt",
        "GUI_INSTALLER.py"
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
    print("\n📦 Creating Windows installer scripts...")
    
    # GUI installer (main installer - professional)
    install_bat = project_root / "INSTALL.bat"
    if install_bat.exists():
        shutil.copy2(install_bat, package_dir / "INSTALL.bat")
        print(f"  ✅ INSTALL.bat (GUI installer launcher)")
    
    # GUI runner
    run_bat = project_root / "RUN.bat"
    if run_bat.exists():
        shutil.copy2(run_bat, package_dir / "RUN.bat")
        print(f"  ✅ RUN.bat (Application launcher)")
    
    # Auto-installer PowerShell script (advanced users)
    auto_install_ps1 = project_root / "AUTO_INSTALL.ps1"
    if auto_install_ps1.exists():
        shutil.copy2(auto_install_ps1, package_dir / "AUTO_INSTALL.ps1")
        print(f"  ✅ AUTO_INSTALL.ps1 (PowerShell installer - advanced)")
    
    # Auto-installer batch launcher (advanced users)
    auto_install_bat = project_root / "AUTO_INSTALL.bat"
    if auto_install_bat.exists():
        shutil.copy2(auto_install_bat, package_dir / "AUTO_INSTALL.bat")
        print(f"  ✅ AUTO_INSTALL.bat (Auto installer - advanced)")
    
    # System checker
    check_system_py = project_root / "CHECK_SYSTEM.py"
    if check_system_py.exists():
        shutil.copy2(check_system_py, package_dir / "CHECK_SYSTEM.py")
        print(f"  ✅ CHECK_SYSTEM.py")
    
    check_system_bat = project_root / "CHECK_SYSTEM.bat"
    if check_system_bat.exists():
        shutil.copy2(check_system_bat, package_dir / "CHECK_SYSTEM.bat")
        print(f"  ✅ CHECK_SYSTEM.bat (System checker)")
    
    # Troubleshooting guide
    troubleshooting_txt = project_root / "TROUBLESHOOTING.txt"
    if troubleshooting_txt.exists():
        shutil.copy2(troubleshooting_txt, package_dir / "TROUBLESHOOTING.txt")
        print(f"  ✅ TROUBLESHOOTING.txt (Help guide)")
    
    # Legacy manual installer (for reference)
    legacy_install_bat = package_dir / "INSTALL_WINDOWS_MANUAL.bat"
    install_bat.write_text("""@echo off
echo ========================================
echo FaB Card Detector - Installation
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo.
    echo Please install Python 3.8 or higher from:
    echo https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation!
    echo.
    pause
    exit /b 1
)

REM Show Python version
echo Checking Python version...
python --version

REM Check Python version is 3.8+
python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: Python 3.8 or higher is required!
    echo Your Python version is too old.
    echo.
    echo Please install Python 3.8 or higher from:
    echo https://www.python.org/downloads/
    echo.
    echo See TROUBLESHOOTING.txt for help.
    echo.
    pause
    exit /b 1
)

echo Python version OK!
echo.

REM Upgrade pip first
echo Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo WARNING: Failed to upgrade pip, continuing anyway...
)

echo.
echo Installing required packages...
echo This may take 5-10 minutes...
echo.

python -m pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Installation failed!
    echo.
    echo Please check:
    echo 1. You have Python 3.8 or higher
    echo 2. You have an internet connection
    echo 3. Run this as Administrator
    echo.
    echo See TROUBLESHOOTING.txt for detailed help.
    echo Or run CHECK_SYSTEM.bat to diagnose issues.
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Run RUN_DETECTOR.bat to start the application.
echo Or run CHECK_SYSTEM.bat to verify installation.
echo.
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
    
    # Copy README
    print("\n📦 Copying documentation...")
    
    readme_source = Path(__file__).parent.parent / "README_WINDOWS.txt"
    readme_dest = package_dir / "README_WINDOWS.txt"
    
    # Add model info and date to the top of README
    with open(readme_source, 'r') as f:
        readme_content = f.read()
    
    # Insert model info after the title
    lines = readme_content.split('\n')
    header_end = next(i for i, line in enumerate(lines) if line.startswith('QUICK START'))
    lines.insert(header_end, f"Model: {model_info}")
    lines.insert(header_end + 1, f"Package Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.insert(header_end + 2, "")
    
    readme_dest.write_text('\n'.join(lines))
    print(f"  ✅ README_WINDOWS.txt")
    
    # Create quickstart
    quickstart = package_dir / "QUICKSTART.txt"
    quickstart_content = """QUICK START GUIDE
==================

Step 1: Double-click INSTALL.bat (GUI installer)
Step 2: Wait for installation (5-10 minutes)
Step 3: Double-click RUN.bat
Step 4: Click "Start Detection" in the GUI

Done!
"""
    quickstart.write_text(quickstart_content)
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
