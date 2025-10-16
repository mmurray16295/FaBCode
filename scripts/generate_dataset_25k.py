"""
Dataset Generation Script for YOLO11x Training
Generates 25,000 synthetic training images for all 2641 FaB card classes
Optimized for maximum accuracy with proper train/val split
"""

import os
import sys
import time
import json
from pathlib import Path

# Add parent directory to path to import test_generate_simple
sys.path.insert(0, str(Path(__file__).parent))

def generate_dataset(
    total_images=25000,
    output_base_dir="../data/synthetic",
    train_split=0.9,
    seed=42
):
    """
    Generate synthetic dataset for YOLO11x training
    
    Args:
        total_images: Total number of images to generate
        output_base_dir: Base directory for output
        train_split: Fraction of images for training (0.9 = 90% train, 10% val)
        seed: Random seed for reproducibility
    """
    
    print("=" * 80)
    print("YOLO11x Dataset Generation")
    print("=" * 80)
    print(f"Total images: {total_images:,}")
    print(f"Train split: {train_split * 100:.0f}%")
    print(f"Validation split: {(1 - train_split) * 100:.0f}%")
    print(f"Output directory: {output_base_dir}")
    print("=" * 80)
    
    # Calculate splits
    n_train = int(total_images * train_split)
    n_val = total_images - n_train
    
    print(f"\nTraining images: {n_train:,}")
    print(f"Validation images: {n_val:,}")
    print(f"Images per class (avg): {total_images / 2641:.1f}")
    
    # Estimate time
    time_per_image = 1.07  # seconds (from performance analysis)
    total_seconds = total_images * time_per_image
    total_hours = total_seconds / 3600
    
    print(f"\nEstimated generation time: {total_hours:.1f} hours")
    print(f"Expected completion: {time.strftime('%I:%M %p', time.localtime(time.time() + total_seconds))}")
    
    # Confirm
    response = input("\nProceed with generation? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Generation cancelled.")
        return
    
    print("\n" + "=" * 80)
    print("Starting dataset generation...")
    print("=" * 80)
    
    start_time = time.time()
    
    # Import after confirmation to avoid slow import on cancel
    from test_generate_simple import main as generate_image
    
    # Generate training images
    print(f"\n[1/2] Generating {n_train:,} TRAINING images...")
    train_dir = os.path.join(output_base_dir, "train")
    os.makedirs(os.path.join(train_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(train_dir, "labels"), exist_ok=True)
    
    for i in range(n_train):
        if i % 100 == 0:
            elapsed = time.time() - start_time
            images_per_sec = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (n_train - i) / images_per_sec if images_per_sec > 0 else 0
            print(f"  Progress: {i:,}/{n_train:,} ({100*i/n_train:.1f}%) | "
                  f"Speed: {images_per_sec:.1f} img/s | "
                  f"Remaining: {remaining/3600:.1f}h")
        
        # Generate image with augmentations enabled
        try:
            generate_image(
                output_dir=train_dir,
                num_images=1,
                augmentations_enabled=True,
                base_seed=seed + i
            )
        except Exception as e:
            print(f"  ERROR on image {i}: {e}")
            continue
    
    # Generate validation images
    print(f"\n[2/2] Generating {n_val:,} VALIDATION images...")
    val_dir = os.path.join(output_base_dir, "valid")
    os.makedirs(os.path.join(val_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(val_dir, "labels"), exist_ok=True)
    
    for i in range(n_val):
        if i % 100 == 0:
            elapsed = time.time() - start_time
            total_generated = n_train + i
            images_per_sec = total_generated / elapsed if elapsed > 0 else 0
            remaining = (total_images - total_generated) / images_per_sec if images_per_sec > 0 else 0
            print(f"  Progress: {i:,}/{n_val:,} ({100*i/n_val:.1f}%) | "
                  f"Speed: {images_per_sec:.1f} img/s | "
                  f"Remaining: {remaining/3600:.1f}h")
        
        # Generate image with augmentations enabled
        try:
            generate_image(
                output_dir=val_dir,
                num_images=1,
                augmentations_enabled=True,
                base_seed=seed + n_train + i  # Different seed than training
            )
        except Exception as e:
            print(f"  ERROR on image {i}: {e}")
            continue
    
    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("Dataset Generation Complete!")
    print("=" * 80)
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Average speed: {total_images / total_time:.2f} images/second")
    print(f"\nOutput directories:")
    print(f"  Training: {os.path.abspath(train_dir)}")
    print(f"  Validation: {os.path.abspath(val_dir)}")
    
    # Count files
    train_images = len(list(Path(train_dir, "images").glob("*.jpg")))
    train_labels = len(list(Path(train_dir, "labels").glob("*.txt")))
    val_images = len(list(Path(val_dir, "images").glob("*.jpg")))
    val_labels = len(list(Path(val_dir, "labels").glob("*.txt")))
    
    print(f"\nGenerated files:")
    print(f"  Training: {train_images:,} images, {train_labels:,} labels")
    print(f"  Validation: {val_images:,} images, {val_labels:,} labels")
    
    # Verify data.yaml exists
    yaml_path = os.path.join(output_base_dir, "data.yaml")
    if os.path.exists(yaml_path):
        print(f"\n✓ data.yaml found: {yaml_path}")
    else:
        print(f"\n⚠ WARNING: data.yaml not found at {yaml_path}")
        print("  You may need to create it before training.")
    
    print("\n" + "=" * 80)
    print("Next Steps:")
    print("=" * 80)
    print("1. Run sanity check: python scripts/train_sanity_check.py")
    print("2. If sanity check passes, run full training: python scripts/train_yolo11x.py")
    print("3. Monitor training: tensorboard --logdir runs/detect")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic dataset for YOLO11x")
    parser.add_argument("--images", type=int, default=25000, help="Total images to generate")
    parser.add_argument("--output", type=str, default="../data/synthetic", help="Output directory")
    parser.add_argument("--train-split", type=float, default=0.9, help="Training split (0.9 = 90%%)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    generate_dataset(
        total_images=args.images,
        output_base_dir=args.output,
        train_split=args.train_split,
        seed=args.seed
    )
