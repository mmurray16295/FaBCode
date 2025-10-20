"""
Build standalone executable for FaB Card Detector
Uses PyInstaller to create platform-specific executables
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def build_executable():
    """Build standalone executable using PyInstaller"""
    
    print("=" * 60)
    print("FaB Card Detector - Executable Builder")
    print("=" * 60)
    
    # Get project root
    project_root = Path(__file__).parent.parent.absolute()
    os.chdir(project_root)
    
    print(f"\nProject root: {project_root}")
    print(f"Platform: {platform.system()}")
    
    # Check if PyInstaller is installed
    try:
        import PyInstaller
        print(f"PyInstaller version: {PyInstaller.__version__}")
    except ImportError:
        print("\n❌ PyInstaller not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)
        print("✅ PyInstaller installed")
    
    # Prepare paths
    main_script = project_root / "fab_detector_app.py"
    build_dir = project_root / "build"
    dist_dir = project_root / "dist"
    
    # Check if main script exists
    if not main_script.exists():
        print(f"\n❌ Error: {main_script} not found!")
        return False
    
    print(f"\nBuilding executable from: {main_script}")
    
    # PyInstaller command
    # --onefile: Bundle everything into single executable
    # --windowed: No console window (GUI only)
    # --name: Name of the executable
    # --add-data: Include additional files
    # --hidden-import: Include modules not automatically detected
    
    pyinstaller_args = [
        sys.executable, "-m", "PyInstaller",
        "--clean",
        "--noconfirm",
        "--onefile",  # Single executable file
        "--windowed",  # No console window
        "--name", "FaBCardDetector",
        "--icon", "NONE",  # TODO: Add icon if available
        
        # Add data files
        f"--add-data=data/card.json{os.pathsep}data",
        f"--add-data=data/card_popularity_weights.json{os.pathsep}data",
        f"--add-data=data/classes.yaml{os.pathsep}data",
        
        # Hidden imports (modules PyInstaller might miss)
        "--hidden-import=ultralytics",
        "--hidden-import=cv2",
        "--hidden-import=torch",
        "--hidden-import=torchvision",
        "--hidden-import=PIL",
        "--hidden-import=yaml",
        "--hidden-import=mss",
        "--hidden-import=tkinter",
        "--hidden-import=numpy",
        
        # Collect submodules
        "--collect-submodules=ultralytics",
        "--collect-data=ultralytics",
        
        # Main script
        str(main_script)
    ]
    
    print("\nRunning PyInstaller...")
    print(f"Command: {' '.join(pyinstaller_args)}")
    
    try:
        result = subprocess.run(pyinstaller_args, check=True, capture_output=True, text=True)
        print("\n✅ Build successful!")
        
        # Find the executable
        exe_name = "FaBCardDetector.exe" if platform.system() == "Windows" else "FaBCardDetector"
        exe_path = dist_dir / exe_name
        
        if exe_path.exists():
            exe_size = exe_path.stat().st_size / (1024 * 1024)  # MB
            print(f"\n📦 Executable created: {exe_path}")
            print(f"   Size: {exe_size:.1f} MB")
            
            # Create package with executable + model
            print("\n📦 Creating deployable package...")
            create_deployable_package(exe_path)
            
            return True
        else:
            print(f"\n❌ Error: Executable not found at {exe_path}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Build failed!")
        print(f"Error: {e}")
        print(f"Output: {e.output}")
        return False

def create_deployable_package(exe_path):
    """Create a deployable package with exe + model + instructions"""
    
    project_root = Path(__file__).parent.parent.absolute()
    
    # Create package directory
    package_name = f"FaBCardDetector_Standalone_{platform.system()}"
    package_dir = project_root / "packages" / package_name
    package_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating package: {package_dir}")
    
    # Copy executable
    import shutil
    shutil.copy2(exe_path, package_dir / exe_path.name)
    print(f"  ✅ Copied executable")
    
    # Copy model weights - try Phase 2 first, fallback to Phase 1
    model_weights_phase2 = project_root / "models" / "phase2_best.pt"
    model_weights_phase1 = project_root / "runs" / "train" / "phase1_100classes" / "weights" / "best.pt"
    
    models_dir = package_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    if model_weights_phase2.exists():
        shutil.copy2(model_weights_phase2, models_dir / "best.pt")
        print(f"  ✅ Copied Phase 2 model weights (500 classes)")
    elif model_weights_phase1.exists():
        shutil.copy2(model_weights_phase1, models_dir / "best.pt")
        print(f"  ✅ Copied Phase 1 model weights (100 classes)")
    else:
        print(f"  ⚠️  Warning: No model weights found!")
    
    # Copy data files - use Phase 2 classes if available
    classes_phase2 = project_root / "data" / "phase2_classes.yaml"
    classes_default = project_root / "data" / "classes.yaml"
    
    data_files = [
        ("data/card.json", "card.json"),
        ("data/card_popularity_weights.json", "card_popularity_weights.json"),
    ]
    
    data_dir = package_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Copy classes.yaml (Phase 2 preferred)
    if classes_phase2.exists():
        shutil.copy2(classes_phase2, data_dir / "classes.yaml")
        print(f"  ✅ Copied Phase 2 classes.yaml (500 classes)")
    elif classes_default.exists():
        shutil.copy2(classes_default, data_dir / "classes.yaml")
        print(f"  ✅ Copied classes.yaml")
    
    # Copy other data files
    for src_path, dst_name in data_files:
        src = project_root / src_path
        if src.exists():
            shutil.copy2(src, data_dir / dst_name)
            print(f"  ✅ Copied {src_path}")
    
    # Create README
    model_phase = "Phase 2 (500 classes)" if model_weights_phase2.exists() else "Phase 1 (100 classes)"
    readme_content = f"""
# FaB Card Detector - Standalone Application

## Quick Start

1. **Double-click** `{exe_path.name}` to launch the application
2. The GUI will open automatically
3. Model file is pre-configured: `models/best.pt`
4. Choose detection mode and settings
5. Click "Start Detection"

## What's Included

- **{exe_path.name}**: The main application (no Python needed!)
- **models/best.pt**: {model_phase} trained model
- **data/**: Card metadata and class information

## Features

- **Windowed Mode**: Shows captured screen with detection boxes
- **Overlay Mode**: Transparent overlay for streaming (now with detection boxes!)
- **Card Preview**: Hover over detected cards to see full card image (toggle in settings)
- **Multi-monitor Support**: Capture from one monitor, display on another
- **Adjustable Confidence**: Fine-tune detection sensitivity

## No Installation Required!

This is a standalone executable - just run it! No need to install Python or any dependencies.

## System Requirements

- **OS**: {platform.system()} ({platform.machine()})
- **RAM**: 4GB minimum, 8GB recommended
- **GPU**: Optional but recommended (NVIDIA GPU with CUDA for best performance)
- **Display**: 1920x1080 or higher

## Model Performance

- **Accuracy**: 99.2% mAP50
- **Classes**: Top 100 most popular FaB cards
- **Speed**: 30-60 FPS (with GPU), 10-20 FPS (CPU only)

## Detection Modes

### Windowed Mode
- Shows full screen capture with detection boxes
- Good for testing and debugging
- See all detections in real-time

### Transparent Overlay Mode (Windows only)
- Minimal UI, only shows card preview on hover
- Perfect for playing online
- Click-through mode available

## Multi-Monitor Support

Ideal for dual-screen setups:
- Set "Capture Monitor" to your game screen
- Set "Display Monitor" to your other screen
- Detections appear where you want them

## Troubleshooting

### Application won't start
- Make sure `models/best.pt` exists in the models folder
- Check that `data/card.json` exists in the data folder

### No detections showing
- Lower confidence threshold to 0.25
- Ensure good lighting on cards
- Cards must be from top 100 list (see data/classes.yaml)

### Low FPS
- GPU not detected - application may be using CPU only
- Close other applications to free up resources
- Lower resolution if capturing full screen

## Known Limitations

- Only detects top 100 most popular cards
- Requires decent lighting conditions
- Transparent overlay mode only works on Windows

## Support

For issues or questions, check the GitHub repository:
https://github.com/mmurray16295/FaBCode

## Version

- **Build Date**: {platform.sys.platform}
- **Python Version**: {platform.python_version()}
- **Model**: Phase 1 (100 classes)
"""
    
    readme_path = package_dir / "README.txt"
    readme_path.write_text(readme_content)
    print(f"  ✅ Created README.txt")
    
    # Create archive
    print("\n📦 Creating compressed archive...")
    archive_name = f"{package_name}_{platform.machine()}"
    
    if platform.system() == "Windows":
        # Create ZIP for Windows
        shutil.make_archive(
            str(project_root / "packages" / archive_name),
            'zip',
            package_dir
        )
        archive_path = project_root / "packages" / f"{archive_name}.zip"
        print(f"  ✅ Created ZIP: {archive_path}")
    else:
        # Create tar.gz for Linux/Mac
        shutil.make_archive(
            str(project_root / "packages" / archive_name),
            'gztar',
            package_dir
        )
        archive_path = project_root / "packages" / f"{archive_name}.tar.gz"
        print(f"  ✅ Created archive: {archive_path}")
    
    archive_size = archive_path.stat().st_size / (1024 * 1024)  # MB
    print(f"  📊 Archive size: {archive_size:.1f} MB")
    
    print("\n" + "=" * 60)
    print("✅ Deployable package created!")
    print("=" * 60)
    print(f"\nPackage location: {package_dir}")
    print(f"Archive: {archive_path}")
    print(f"\nUsers can extract and double-click {exe_path.name} to run!")

if __name__ == "__main__":
    success = build_executable()
    sys.exit(0 if success else 1)
