"""
Consolidated test script for synthetic playmat generation.
Generates test images with configurable options and comprehensive verification.
"""

import sys
import time
import argparse
from pathlib import Path
from generation_utils import ensure_background_variations, print_background_usage_stats

def generate_image(enable_augmentations=True, draw_bboxes=False, preset_name=None, use_background_cycling=False, target_images=1, selector_type='weighted'):
    """
    Wrapper to call Core_Playmat_Generator.main() with specified parameters.
    Separates the generation logic from test orchestration.
    """
    from Core_Playmat_Generator import main as generate_playmat
    generate_playmat(
        enable_augmentations=enable_augmentations,
        draw_bboxes=draw_bboxes,
        preset_name=preset_name,
        use_background_cycling=use_background_cycling,
        target_images=target_images,
        selector_type=selector_type
    )

def verify_output(base_dir, expected_count=None):
    """
    Verify generated output files and provide statistics.
    
    Args:
        base_dir: Base directory containing synthetic data
        expected_count: Expected total number of images (optional)
    
    Returns:
        True if verification passes, False otherwise
    """
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    
    synthetic_dir = Path(base_dir)
    
    if not synthetic_dir.exists():
        print(f"✗ Synthetic directory not found: {synthetic_dir}")
        return False
    
    # Count files per split
    splits = ['train', 'valid', 'test']
    total_images = 0
    total_labels = 0
    split_stats = {}
    
    for split in splits:
        images_dir = synthetic_dir / split / 'images'
        labels_dir = synthetic_dir / split / 'labels'
        
        if images_dir.exists():
            images = list(images_dir.glob("*.jpg"))
            labels = list(labels_dir.glob("*.txt"))
            total_images += len(images)
            total_labels += len(labels)
            split_stats[split] = {'images': len(images), 'labels': len(labels)}
            
            print(f"\n{split.upper()}:")
            print(f"  Images: {len(images)}")
            print(f"  Labels: {len(labels)}")
            
            # Check a sample label file
            if labels:
                sample_label = labels[0]
                with open(sample_label, 'r') as f:
                    lines = f.readlines()
                print(f"  Sample label ({sample_label.name}):")
                print(f"    Cards labeled: {len(lines)}")
                if lines:
                    first_line = lines[0].strip().split()
                    if len(first_line) == 5:
                        class_id = first_line[0]
                        print(f"    First card class ID: {class_id}")
                        print(f"    ✓ Label format valid")
                    else:
                        print(f"    ✗ Invalid label format")
                        return False
    
    print(f"\nTOTAL: {total_images} images, {total_labels} labels")
    
    # Verify image/label count match
    if total_images != total_labels:
        print(f"\n✗ MISMATCH: {total_images} images but {total_labels} labels")
        return False
    
    # Check expected count if provided
    if expected_count and total_images != expected_count:
        print(f"\n⚠ WARNING: Expected {expected_count} images, found {total_images}")
        # Don't fail, just warn (some may have failed during generation)
    
    # Check data.yaml
    yaml_path = synthetic_dir / 'data.yaml'
    if yaml_path.exists():
        print(f"\n✓ data.yaml exists: {yaml_path}")
        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_content = f.read()
        
        # Count classes in yaml (format: "  - Card Name" or "  0: Card Name")
        import re
        # Try Python list format first (  - Card Name)
        class_matches = re.findall(r'^\s+-\s+.+$', yaml_content, re.MULTILINE)
        if not class_matches:
            # Try dictionary format (  0: Card Name)
            class_matches = re.findall(r'^\s+\d+:\s*.+$', yaml_content, re.MULTILINE)
        num_classes = len(class_matches)
        print(f"  Classes defined: {num_classes}")
        
        if num_classes == 2641:
            print("  ✓ Correct! All 2,641 card classes present")
        else:
            print(f"  ⚠ WARNING: Expected 2,641 classes, found {num_classes}")
    else:
        print(f"\n✗ data.yaml NOT FOUND: {yaml_path}")
        print("  Run: python scripts/synthetic_generation/generate_card_data_yaml.py")
        return False
    
    return total_images > 0 and total_images == total_labels

def main():
    parser = argparse.ArgumentParser(
        description='Generate test synthetic playmat images with verification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image with visualization
  python test_generation.py --count 1 --visualize
  
  # Quick 5-image test
  python test_generation.py --count 5
  
  # Full verification test (20 images)
  python test_generation.py --count 20 --verify
  
  # Debug without augmentations
  python test_generation.py --count 3 --no-augmentations --visualize
        """
    )
    
    parser.add_argument('--count', '-n', type=int, default=5,
                        help='Number of images to generate (default: 5)')
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Draw bounding boxes on images for visual inspection')
    parser.add_argument('--no-augmentations', action='store_true',
                        help='Disable augmentations (for debugging)')
    parser.add_argument('--verify', action='store_true',
                        help='Run full verification checks after generation')
    parser.add_argument('--preset', type=str, default=None,
                        help='Augmentation preset name (e.g., "phase1", "phase2")')
    parser.add_argument('--selector', type=str, choices=['weighted', 'smooth'], default='weighted',
                        help='Card selector type: "weighted" (popularity-based) or "smooth" (even distribution)')
    parser.add_argument('--reset-smooth', action='store_true',
                        help='Reset smooth selector state (clear all usage counts)')
    parser.add_argument('--ensure-backgrounds', action='store_true',
                        help='Check and generate background variations if needed (auto-enabled for count >= 10)')
    
    args = parser.parse_args()
    
    # Handle reset-smooth flag
    if args.reset_smooth:
        from card_selector_smooth import SmoothCardSelector
        selector = SmoothCardSelector()
        selector.reset_state()
        print("✓ Smooth selector state has been reset")
        return
    
    # Configuration
    count = args.count
    enable_augmentations = not args.no_augmentations
    draw_bboxes = args.visualize
    verify = args.verify
    preset_name = args.preset
    selector_type = args.selector
    ensure_backgrounds = args.ensure_backgrounds or count >= 10  # Auto-enable for larger batches
    
    # Print configuration
    print("=" * 80)
    print("SYNTHETIC PLAYMAT GENERATION TEST")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Images to generate: {count}")
    print(f"  Card selector: {selector_type.upper()}")
    print(f"  Augmentations: {'ENABLED' if enable_augmentations else 'DISABLED'}")
    print(f"  Visualization (bboxes): {'ENABLED' if draw_bboxes else 'DISABLED'}")
    print(f"  Verification: {'ENABLED' if verify else 'DISABLED'}")
    print(f"  Background management: {'ENABLED' if ensure_backgrounds else 'DISABLED'}")
    if preset_name:
        print(f"  Augmentation preset: {preset_name}")
    print("\n" + "=" * 80)
    
    # Ensure sufficient background variations exist (for larger batches)
    if ensure_backgrounds:
        num_backgrounds = ensure_background_variations(count)
        print_background_usage_stats(count, num_backgrounds)
    
    # Track statistics
    success_count = 0
    error_count = 0
    errors = []
    start_time = time.time()
    
    # Generate images
    for i in range(count):
        print(f"\n[{i+1}/{count}] Generating image...")
        print("-" * 80)
        
        try:
            generate_image(
                enable_augmentations=enable_augmentations,
                draw_bboxes=draw_bboxes,
                preset_name=preset_name,
                use_background_cycling=(count > 1),  # Use cycling for multi-image generation
                target_images=count,
                selector_type=selector_type
            )
            success_count += 1
            print(f"✓ Success ({success_count}/{i+1})")
            
        except Exception as e:
            error_count += 1
            error_msg = str(e)
            errors.append(f"Image {i+1}: {error_msg}")
            print(f"✗ Error: {error_msg}")
            print(f"Failed ({error_count}/{i+1})")
            # Continue with next image
            continue
    
    # Calculate timing
    total_time = time.time() - start_time
    avg_time = total_time / count if count > 0 else 0
    
    # Print summary
    print("\n" + "=" * 80)
    print("GENERATION SUMMARY")
    print("=" * 80)
    print(f"Success: {success_count}/{count} ({100*success_count/count:.1f}%)")
    print(f"Errors: {error_count}/{count}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average per image: {avg_time:.2f}s")
    
    if errors:
        print("\nErrors encountered:")
        for error in errors:
            print(f"  - {error}")
    
    # Run verification if requested or if count >= 10
    if verify or count >= 10:
        # Determine base directory based on selector type
        import platform
        subdir = 'synthetic_smooth' if selector_type == 'smooth' else 'synthetic'
        if platform.system() == 'Windows':
            base_dir = Path(rf'c:\VS Code\FaB Code\data\{subdir}')
        else:
            base_dir = Path(__file__).parent.parent.parent / 'data' / subdir
        
        verification_passed = verify_output(base_dir, expected_count=success_count)
        
        print("\n" + "=" * 80)
        print("TEST COMPLETE")
        print("=" * 80)
        
        if verification_passed and error_count == 0:
            print("✓ ALL CHECKS PASSED")
            return 0
        elif verification_passed:
            print(f"⚠ PASSED WITH WARNINGS ({error_count} generation errors)")
            return 0
        else:
            print("✗ VERIFICATION FAILED")
            return 1
    else:
        print("\n" + "=" * 80)
        print("TEST COMPLETE")
        print("=" * 80)
        
        if error_count == 0:
            print("✓ ALL IMAGES GENERATED SUCCESSFULLY")
            return 0
        else:
            print(f"⚠ COMPLETED WITH {error_count} ERRORS")
            return 1

if __name__ == '__main__':
    sys.exit(main())
