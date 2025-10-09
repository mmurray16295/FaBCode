"""
Organize new labeled background data into train/valid/test splits.

This script:
1. Clears old data from YouTube_Labeled/train, /valid, /test folders
2. Randomly distributes new backgrounds from Card Detection.v3 folder
3. Keeps image/label pairs together
4. Uses 70/20/10 split ratio (train/valid/test)
"""

import os
import shutil
import random
from pathlib import Path

# Paths
NEW_DATA_BASE = r'C:\VS Code\FaB Code\data\images\YouTube_Labeled\Card Detection.v3-ref_included_full_size.yolov11'
OLD_DATA_BASE = r'C:\VS Code\FaB Code\data\images\YouTube_Labeled'

# Split ratios (must sum to 1.0)
TRAIN_RATIO = 0.70
VALID_RATIO = 0.20
TEST_RATIO = 0.10

def clear_old_data(base_dir):
    """Clear old images and labels from train/valid/test folders."""
    for split in ['train', 'valid', 'test']:
        for subfolder in ['images', 'labels']:
            path = os.path.join(base_dir, split, subfolder)
            if os.path.exists(path):
                print(f"Clearing {path}...")
                for file in os.listdir(path):
                    file_path = os.path.join(path, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                print(f"  Cleared {len(os.listdir(path))} items")
            else:
                print(f"Creating {path}...")
                os.makedirs(path, exist_ok=True)


def get_image_label_pairs(source_dir):
    """
    Get all image/label pairs from the source directory.
    Returns list of (image_path, label_path, basename) tuples.
    """
    images_dir = os.path.join(source_dir, 'train', 'images')
    labels_dir = os.path.join(source_dir, 'train', 'labels')
    
    if not os.path.exists(images_dir):
        print(f"Error: {images_dir} not found!")
        return []
    
    pairs = []
    for img_file in os.listdir(images_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            basename = os.path.splitext(img_file)[0]
            img_path = os.path.join(images_dir, img_file)
            label_path = os.path.join(labels_dir, basename + '.txt')
            
            if os.path.exists(label_path):
                pairs.append((img_path, label_path, basename, os.path.splitext(img_file)[1]))
            else:
                print(f"Warning: No label found for {img_file}")
    
    return pairs


def distribute_to_splits(pairs, dest_base, train_ratio, valid_ratio, test_ratio):
    """
    Randomly distribute image/label pairs into train/valid/test splits.
    """
    # Shuffle pairs randomly
    random.shuffle(pairs)
    
    total = len(pairs)
    train_count = int(total * train_ratio)
    valid_count = int(total * valid_ratio)
    # test gets the remainder to ensure all files are assigned
    
    train_pairs = pairs[:train_count]
    valid_pairs = pairs[train_count:train_count + valid_count]
    test_pairs = pairs[train_count + valid_count:]
    
    splits = {
        'train': train_pairs,
        'valid': valid_pairs,
        'test': test_pairs
    }
    
    print(f"\nDistribution:")
    print(f"  Train: {len(train_pairs)} images ({len(train_pairs)/total*100:.1f}%)")
    print(f"  Valid: {len(valid_pairs)} images ({len(valid_pairs)/total*100:.1f}%)")
    print(f"  Test:  {len(test_pairs)} images ({len(test_pairs)/total*100:.1f}%)")
    print()
    
    # Copy files to their respective splits
    for split_name, split_pairs in splits.items():
        img_dest = os.path.join(dest_base, split_name, 'images')
        label_dest = os.path.join(dest_base, split_name, 'labels')
        
        os.makedirs(img_dest, exist_ok=True)
        os.makedirs(label_dest, exist_ok=True)
        
        print(f"Copying {len(split_pairs)} pairs to {split_name}...")
        for img_path, label_path, basename, ext in split_pairs:
            # Copy image
            shutil.copy2(img_path, os.path.join(img_dest, basename + ext))
            # Copy label
            shutil.copy2(label_path, os.path.join(label_dest, basename + '.txt'))
        
        print(f"  ✓ {split_name} complete")


def main():
    print("=" * 70)
    print("ORGANIZING BACKGROUND DATA INTO TRAIN/VALID/TEST SPLITS")
    print("=" * 70)
    print()
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Step 1: Clear old data
    print("Step 1: Clearing old data...")
    print("-" * 70)
    clear_old_data(OLD_DATA_BASE)
    print()
    
    # Step 2: Get image/label pairs from new data
    print("Step 2: Reading new labeled data...")
    print("-" * 70)
    pairs = get_image_label_pairs(NEW_DATA_BASE)
    print(f"Found {len(pairs)} image/label pairs")
    print()
    
    if not pairs:
        print("Error: No image/label pairs found. Aborting.")
        return
    
    # Step 3: Distribute to splits
    print("Step 3: Distributing to train/valid/test splits...")
    print("-" * 70)
    distribute_to_splits(pairs, OLD_DATA_BASE, TRAIN_RATIO, VALID_RATIO, TEST_RATIO)
    print()
    
    # Summary
    print("=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"\nNew background data is now organized in:")
    print(f"  {OLD_DATA_BASE}\\train\\")
    print(f"  {OLD_DATA_BASE}\\valid\\")
    print(f"  {OLD_DATA_BASE}\\test\\")
    print()
    print("You can now run the synthetic data generation script!")


if __name__ == '__main__':
    # Confirm before proceeding
    print("\n" + "=" * 70)
    print("WARNING: This will DELETE all existing data in:")
    print(f"  {OLD_DATA_BASE}\\train\\")
    print(f"  {OLD_DATA_BASE}\\valid\\")
    print(f"  {OLD_DATA_BASE}\\test\\")
    print("=" * 70)
    
    response = input("\nDo you want to continue? (yes/no): ").strip().lower()
    
    if response == 'yes':
        main()
    else:
        print("Aborted.")
