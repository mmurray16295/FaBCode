import sys
import platform

print("=" * 60)
print("FaB Card Detector - System Check")
print("=" * 60)
print()

# Python version
print(f"Python Version: {sys.version}")
print(f"Python Path: {sys.executable}")
print(f"Platform: {platform.platform()}")
print(f"Architecture: {platform.machine()}")
print()

# Check version requirements
major, minor = sys.version_info[:2]
print(f"Python {major}.{minor} detected")

if (major, minor) >= (3, 8):
    print("✅ Python version is compatible (3.8+ required)")
else:
    print("❌ Python version is TOO OLD (3.8+ required)")
    print()
    print("Please run AUTO_INSTALL.bat as Administrator to install Python 3.11")
    print("Or manually install Python 3.8+ from: https://www.python.org/downloads/")
    sys.exit(1)

print()
print("Checking required modules...")
print()

# Check each module
modules = [
    ('ultralytics', 'YOLO detection'),
    ('torch', 'PyTorch deep learning'),
    ('cv2', 'OpenCV computer vision'),
    ('PIL', 'Pillow image processing'),
    ('numpy', 'NumPy arrays'),
    ('yaml', 'YAML config'),
    ('mss', 'Screen capture'),
    ('tkinter', 'GUI framework'),
    ('requests', 'HTTP requests'),
]

missing = []
for module, description in modules:
    try:
        __import__(module)
        print(f"✅ {module:20s} - {description}")
    except ImportError:
        print(f"❌ {module:20s} - {description} (NOT INSTALLED)")
        missing.append(module)

print()
print("=" * 60)

if missing:
    print("❌ Missing modules:", ", ".join(missing))
    print()
    print("To fix this:")
    print("1. Run AUTO_INSTALL.bat as Administrator (recommended)")
    print("2. Or run INSTALL_WINDOWS.bat")
    print("3. See TROUBLESHOOTING.txt for help")
else:
    print("✅ All required modules are installed!")
    print()
    print("You can run RUN_DETECTOR.bat to start the application")

print("=" * 60)
input("\nPress Enter to close...")
