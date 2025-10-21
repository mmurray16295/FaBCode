"""
Download YOLO11x model weights for training
"""

from ultralytics import YOLO

print("Downloading YOLO11x pretrained weights...")
print("This will download ~110MB from Ultralytics servers")

try:
    # This will automatically download yolo11x.pt if not present
    model = YOLO("yolo11x.pt")
    
    print("\n✓ YOLO11x downloaded successfully!")
    print(f"  Model: {model.ckpt_path}")
    print(f"  Parameters: 56.9M")
    print(f"  Task: Detection")
    
    # Verify model info
    print("\nModel Info:")
    print(model.info())
    
except Exception as e:
    print(f"\n❌ Download failed: {e}")
    print("\nTroubleshooting:")
    print("  - Check internet connection")
    print("  - Verify ultralytics is installed: pip install ultralytics")
    print("  - Try manual download: https://github.com/ultralytics/assets/releases/")
