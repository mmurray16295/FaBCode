"""
Quick test script to generate a small dataset and verify all paths are correct
"""
import os
import sys
import subprocess
from pathlib import Path

def main():
    print("=" * 80)
    print("TEST DATASET GENERATION")
    print("=" * 80)
    print("\nThis will generate 20 test images (18 train, 2 val) to verify:")
    print("  - Card images are accessible")
    print("  - Background images are accessible")
    print("  - YOLO labels are generated correctly")
    print("  - data.yaml is created with all 2,641 classes")
    print("  - Output directory structure is correct")
    print("\n" + "=" * 80)
    
    # Navigate to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir.parent)
    
    print("\nGenerating 20 test images...")
    print("Running: python3 scripts/test_generate_simple.py (20 times)")
    print("-" * 80)
    
    # Call test_generate_simple.py 20 times
    for i in range(20):
        print(f"\nGenerating image {i+1}/20...")
        result = subprocess.run(
            [sys.executable, "scripts/test_generate_simple.py"],
            capture_output=False,
            text=True
        )
        if result.returncode != 0:
            print(f"ERROR: Failed to generate image {i+1}")
            return 1
    
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    
    # Check output directory
    synthetic_dir = Path("data/synthetic")
    
    # Count files
    splits = ['train', 'valid', 'test']
    total_images = 0
    total_labels = 0
    
    for split in splits:
        images_dir = synthetic_dir / split / 'images'
        labels_dir = synthetic_dir / split / 'labels'
        
        if images_dir.exists():
            images = list(images_dir.glob("*.jpg"))
            labels = list(labels_dir.glob("*.txt"))
            total_images += len(images)
            total_labels += len(labels)
            print(f"\n{split.upper()}:")
            print(f"  Images: {len(images)}")
            print(f"  Labels: {len(labels)}")
            
            # Check a sample label file
            if labels:
                sample_label = labels[0]
                with open(sample_label, 'r') as f:
                    lines = f.readlines()
                print(f"  Sample label ({sample_label.name}):")
                print(f"    Lines (cards): {len(lines)}")
                if lines:
                    first_line = lines[0].strip().split()
                    class_id = first_line[0]
                    print(f"    First card class ID: {class_id}")
    
    print(f"\nTOTAL: {total_images} images, {total_labels} labels")
    
    # Check data.yaml
    yaml_path = synthetic_dir / 'data.yaml'
    if yaml_path.exists():
        print(f"\n✓ data.yaml exists: {yaml_path}")
        with open(yaml_path, 'r') as f:
            yaml_content = f.read()
        
        # Count classes in yaml
        import re
        class_matches = re.findall(r'^\s+\d+:', yaml_content, re.MULTILINE)
        num_classes = len(class_matches)
        print(f"  Classes defined: {num_classes}")
        
        if num_classes == 2641:
            print("  ✓ Correct! All 2,641 classes present")
        else:
            print(f"  ⚠ WARNING: Expected 2,641 classes, found {num_classes}")
    else:
        print(f"\n✗ data.yaml NOT FOUND: {yaml_path}")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    
    if total_images == 20 and total_labels == 20:
        print("✓ SUCCESS: All 20 images generated with labels")
        return 0
    else:
        print(f"⚠ WARNING: Expected 20 images and labels, got {total_images} images and {total_labels} labels")
        return 1

if __name__ == "__main__":
    sys.exit(main())
