"""
Batch generate synthetic playmat images for YOLO training.
Generates images with proper train/val/test splits and YOLO labels.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from test_generate_simple import main as generate_one

def generate_batch(num_images: int = 100, enable_augmentations: bool = True):
    """
    Generate a batch of synthetic playmat images.
    
    Args:
        num_images: Number of images to generate
        enable_augmentations: Whether to apply augmentations
    """
    print("=" * 70)
    print(f"BATCH GENERATION: {num_images} images")
    print(f"Augmentations: {'ENABLED' if enable_augmentations else 'DISABLED'}")
    print("=" * 70)
    
    success_count = 0
    error_count = 0
    
    for i in range(num_images):
        try:
            print(f"\n[{i+1}/{num_images}] Generating image...")
            generate_one(enable_augmentations=enable_augmentations, draw_bboxes=False)
            success_count += 1
            print(f"✓ Success ({success_count}/{i+1})")
        except Exception as e:
            error_count += 1
            print(f"✗ Error: {e}")
            print(f"Failed ({error_count}/{i+1})")
            # Continue with next image even if one fails
            continue
    
    print("\n" + "=" * 70)
    print(f"BATCH COMPLETE")
    print(f"Success: {success_count}/{num_images} ({100*success_count/num_images:.1f}%)")
    print(f"Errors: {error_count}/{num_images}")
    print("=" * 70)
    
    # Print dataset statistics
    base_dir = Path(r'c:\VS Code\FaB Code\data\synthetic')
    
    if base_dir.exists():
        train_images = list((base_dir / 'train' / 'images').glob('*.jpg')) if (base_dir / 'train' / 'images').exists() else []
        valid_images = list((base_dir / 'valid' / 'images').glob('*.jpg')) if (base_dir / 'valid' / 'images').exists() else []
        test_images = list((base_dir / 'test' / 'images').glob('*.jpg')) if (base_dir / 'test' / 'images').exists() else []
        
        print(f"\nDataset Statistics:")
        print(f"  Train: {len(train_images)} images")
        print(f"  Valid: {len(valid_images)} images")
        print(f"  Test: {len(test_images)} images")
        print(f"  Total: {len(train_images) + len(valid_images) + len(test_images)} images")
        
        # Check for data.yaml
        yaml_path = base_dir / 'data.yaml'
        if yaml_path.exists():
            print(f"\n✓ data.yaml exists at {yaml_path}")
        else:
            print(f"\n✗ data.yaml NOT FOUND at {yaml_path}")

if __name__ == '__main__':
    # Default: generate 10 images for testing
    # Use command line arg for custom count: python generate_batch.py 100
    import sys
    
    num_images = 10
    if len(sys.argv) > 1:
        try:
            num_images = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number: {sys.argv[1]}, using default: {num_images}")
    
    generate_batch(num_images=num_images, enable_augmentations=True)
