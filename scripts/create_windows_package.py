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
    
    # Create installer files for Windows
    print("\n📦 Copying installer scripts...")
    
    # PowerShell installer
    install_ps1 = project_root / "INSTALL.ps1"
    if install_ps1.exists():
        shutil.copy2(install_ps1, package_dir / "INSTALL.ps1")
        print(f"  ✅ INSTALL.ps1 (Automatic installer)")
    else:
        print(f"  ⚠️  WARNING: INSTALL.ps1 not found!")
    
    # Application runner
    run_bat = project_root / "RUN.bat"
    if run_bat.exists():
        shutil.copy2(run_bat, package_dir / "RUN.bat")
        print(f"  ✅ RUN.bat (Application launcher)")
    else:
        print(f"  ⚠️  WARNING: RUN.bat not found!")
    
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

Step 1: Right-click INSTALL.ps1 -> "Run with PowerShell"
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
