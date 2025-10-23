#!/usr/bin/env python3
"""
Merge smooth dataset into synthetic dataset.

Copies smooth images from:
- /workspace/FaBCode/data/synthetic_smooth/ (15k smooth selector images)

Into:
- /workspace/FaBCode/data/synthetic/ (already has 5k weighted selector images)

Creates data.yaml for YOLO training with 2641 classes.
"""

import shutil
import json
from pathlib import Path
from tqdm import tqdm

def merge_datasets():
    """Copy smooth dataset into synthetic dataset (which already has weighted images)."""
    
    # Paths
    base_path = Path(__file__).resolve().parent.parent.parent
    smooth_dir = base_path / 'data' / 'synthetic_smooth'
    synthetic_dir = base_path / 'data' / 'synthetic'
    
    print("=" * 80)
    print("DATASET MERGER")
    print("=" * 80)
    print(f"Smooth source: {smooth_dir}")
    print(f"Synthetic target (has weighted): {synthetic_dir}")
    print("=" * 80)
    
    # Check current counts
    print("\nCurrent image counts in synthetic:")
    for split in ['train', 'valid', 'test']:
        split_dir = synthetic_dir / split / 'images'
        if split_dir.exists():
            count = len(list(split_dir.glob('*.*')))
            print(f"  {split}: {count} images")
    
    # Copy smooth files into synthetic
    splits = ['train', 'valid', 'test']
    stats = {'train': 0, 'valid': 0, 'test': 0, 'conflicts': 0}
    
    for split in splits:
        print(f"\n{split.upper()}:")
        
        smooth_split = smooth_dir / split
        synthetic_split = synthetic_dir / split
        
        if not smooth_split.exists():
            print(f"  Warning: {smooth_split} does not exist, skipping...")
            continue
        
        # Ensure synthetic split directories exist
        (synthetic_split / 'images').mkdir(parents=True, exist_ok=True)
        (synthetic_split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Copy images
        smooth_images = [f for f in (smooth_split / 'images').glob('*.*') if f.is_file()]
        print(f"  Copying {len(smooth_images)} smooth images...")
        for img in tqdm(smooth_images, desc=f"  Smooth {split} images"):
            dest = synthetic_split / 'images' / img.name
            if dest.exists():
                stats['conflicts'] += 1
                continue
            shutil.copy2(img, dest)
            stats[split] += 1
        
        # Copy labels
        smooth_labels = [f for f in (smooth_split / 'labels').glob('*.txt') if f.is_file()]
        print(f"  Copying {len(smooth_labels)} smooth labels...")
        for lbl in tqdm(smooth_labels, desc=f"  Smooth {split} labels"):
            dest = synthetic_split / 'labels' / lbl.name
            if dest.exists():
                continue
            shutil.copy2(lbl, dest)
    
    print("\n" + "=" * 80)
    print("MERGE SUMMARY")
    print("=" * 80)
    print(f"Smooth images copied:")
    print(f"  Train: {stats['train']}")
    print(f"  Valid: {stats['valid']}")
    print(f"  Test: {stats['test']}")
    print(f"  Total: {stats['train'] + stats['valid'] + stats['test']}")
    if stats['conflicts'] > 0:
        print(f"  Conflicts skipped: {stats['conflicts']}")
    
    # Count final totals
    print(f"\nFinal image counts in synthetic:")
    total = 0
    for split in ['train', 'valid', 'test']:
        split_dir = synthetic_dir / split / 'images'
        if split_dir.exists():
            count = len(list(split_dir.glob('*.*')))
            print(f"  {split}: {count} images")
            total += count
    print(f"  TOTAL: {total} images")
    print("=" * 80)
    
    # Create data.yaml
    print("\nCreating data.yaml...")
    create_data_yaml(synthetic_dir, base_path)
    
    print(f"\n✓ Dataset merged successfully!")
    print(f"✓ Output: {synthetic_dir}")
    print(f"✓ data.yaml: {synthetic_dir / 'data.yaml'}")
    print("\nReady for training!")

def create_data_yaml(synthetic_dir, base_path):
    """Create data.yaml for YOLO training."""
    
    # Load class names from backup yaml
    backup_yaml = base_path / 'data' / 'databackup2641.yaml'
    
    # Parse the yaml manually to extract class names
    with open(backup_yaml, 'r') as f:
        content = f.read()
    
    # Extract the names list (it's on one line in the backup)
    names_start = content.find("names: [")
    names_end = content.find("]", names_start)
    names_str = content[names_start+8:names_end]  # Skip "names: ["
    
    # Parse the class names
    class_names = [name.strip().strip("'") for name in names_str.split("', '")]
    
    # Create data.yaml content
    yaml_content = f"""# YOLO Dataset Configuration - Combined Smooth + Weighted
# Synthetic playmat images for FaB card detection
# Smooth selector: Even distribution across all cards
# Weighted selector: Popular cards emphasized

train: train/images
val: valid/images
test: test/images

nc: {len(class_names)}
names: {class_names}
"""
    
    # Write data.yaml
    yaml_path = synthetic_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"  Created data.yaml with {len(class_names)} classes")

if __name__ == '__main__':
    merge_datasets()
