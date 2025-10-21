"""
Fix Card Corner Artifacts
==========================
This script removes white/black corner artifacts from card images that were created
during the resize process when backgrounds weren't properly handled.

Strategy:
1. Detect if card has rounded corners (most cards do)
2. Create a rounded corner mask based on card dimensions
3. Apply mask to remove sharp corners
4. Save back to original location

Can process individual files or entire directories recursively.
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional
import argparse


def create_rounded_corner_mask(width: int, height: int, corner_radius: int = 11) -> np.ndarray:
    """
    Create a mask with rounded corners for a card image.
    Creates an alpha mask where 255=opaque (keep) and 0=transparent (remove).
    Uses super-sampling for smooth anti-aliased edges.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        corner_radius: Radius of rounded corners in pixels (default 11 for clean corner removal)
        
    Returns:
        Binary mask (0-255) where 255=keep card, 0=remove corner artifacts
    """
    # Use 4x super-sampling for smooth edges
    scale = 4
    large_width = width * scale
    large_height = height * scale
    
    # Start with black mask (everything transparent)
    mask_large = np.zeros((large_height, large_width), dtype=np.uint8)
    
    # Scale up radius (use the provided corner_radius directly)
    radius = corner_radius * scale
    
    # Draw a white filled rectangle for the main card body
    cv2.rectangle(mask_large, (radius, 0), (large_width - radius, large_height), 255, -1)  # Vertical center strip
    cv2.rectangle(mask_large, (0, radius), (large_width, large_height - radius), 255, -1)  # Horizontal center strip
    
    # Draw filled circles at corners to round them with LINE_AA for anti-aliasing
    # Top-left
    cv2.circle(mask_large, (radius, radius), radius, 255, -1, cv2.LINE_AA)
    
    # Top-right
    cv2.circle(mask_large, (large_width - radius - 1, radius), radius, 255, -1, cv2.LINE_AA)
    
    # Bottom-left
    cv2.circle(mask_large, (radius, large_height - radius - 1), radius, 255, -1, cv2.LINE_AA)
    
    # Bottom-right
    cv2.circle(mask_large, (large_width - radius - 1, large_height - radius - 1), radius, 255, -1, cv2.LINE_AA)
    
    # Downsample with INTER_AREA for smooth blending
    mask = cv2.resize(mask_large, (width, height), interpolation=cv2.INTER_AREA)
    
    return mask


def detect_corner_artifacts(image: np.ndarray, corner_size: int = 3, threshold: int = 50) -> bool:
    """
    Detect if image has corner artifacts (white or black pixels in actual corner).
    Checks only the very corner pixels that should be transparent/rounded.
    
    Args:
        image: Input image (BGR)
        corner_size: Size of corner pixel sample to check (default 3x3)
        threshold: Threshold for detecting uniform color (0-255)
        
    Returns:
        True if artifacts detected, False otherwise
    """
    height, width = image.shape[:2]
    
    # Check only the actual corner pixels (3x3 samples at each corner)
    corners = [
        image[0:corner_size, 0:corner_size],  # Top-left
        image[0:corner_size, width-corner_size:width],  # Top-right
        image[height-corner_size:height, 0:corner_size],  # Bottom-left
        image[height-corner_size:height, width-corner_size:width]  # Bottom-right
    ]
    
    artifacts_found = 0
    for corner in corners:
        # Calculate mean color of corner pixels
        mean_color = np.mean(corner, axis=(0, 1))
        
        # Check if corner pixels are very white (>205) or very black (<50)
        # These should be transparent/rounded, not solid color
        if np.all(mean_color > 255 - threshold) or np.all(mean_color < threshold):
            artifacts_found += 1
    
    # If 3 or more corners have artifacts, likely a problem
    return artifacts_found >= 3


def fix_card_corners(image_path: Path, corner_radius: int = 15, dry_run: bool = False, 
                    test_output_dir: Optional[Path] = None, verbose: bool = False) -> Tuple[bool, str]:
    """
    Fix corner artifacts in a card image by applying rounded corner mask.
    
    Args:
        image_path: Path to image file
        corner_radius: Radius of rounded corners
        dry_run: If True, don't save changes (just report)
        test_output_dir: If provided, save to this directory instead of overwriting
        verbose: If True, print detailed processing info
        
    Returns:
        Tuple of (was_processed, status_message)
    """
    # Load image with alpha channel if present
    img_pil = Image.open(image_path)
    original_mode = img_pil.mode
    
    # Convert to RGBA if not already
    if img_pil.mode != 'RGBA':
        img_pil = img_pil.convert('RGBA')
    
    # Convert to numpy array
    img_array = np.array(img_pil)
    
    # Check if image needs fixing
    img_bgr = cv2.cvtColor(img_array[:, :, :3], cv2.COLOR_RGB2BGR)
    has_artifacts = detect_corner_artifacts(img_bgr)
    
    if not has_artifacts:
        return False, "No artifacts detected"  # Skip
    
    height, width = img_array.shape[:2]
    
    # Create rounded corner mask
    mask = create_rounded_corner_mask(width, height, corner_radius)
    
    # Check if original had alpha channel
    had_alpha = original_mode in ('RGBA', 'LA')
    
    # Apply mask to alpha channel
    if img_array.shape[2] == 4:
        # Combine existing alpha with corner mask
        original_alpha = img_array[:, :, 3].copy()
        img_array[:, :, 3] = np.minimum(img_array[:, :, 3], mask)
        
        # Check how much we're actually changing
        alpha_diff = np.sum(original_alpha != img_array[:, :, 3])
        status = f"Modified {alpha_diff} alpha pixels (had alpha: {had_alpha})"
    else:
        # Add alpha channel with corner mask
        img_array = np.dstack([img_array, mask])
        status = f"Added alpha channel with rounded corners (had alpha: {had_alpha})"
    
    if not dry_run:
        result_img = Image.fromarray(img_array, mode='RGBA')
        
        if test_output_dir:
            # Save to test directory with original filename
            test_output_dir.mkdir(parents=True, exist_ok=True)
            output_path = test_output_dir / image_path.name
            result_img.save(output_path, 'PNG')
        else:
            # Save back to original location (overwrite)
            result_img.save(image_path, 'PNG')
        return True, status
    
    return True, f"Would fix: {status}"


def process_directory(directory: Path, corner_radius: int = 15, dry_run: bool = False, 
                      recursive: bool = True, test_output_dir: Optional[Path] = None) -> Tuple[int, int]:
    """
    Process all PNG images in a directory.
    
    Args:
        directory: Directory to process
        corner_radius: Radius of rounded corners
        dry_run: If True, don't save changes (just report)
        recursive: If True, process subdirectories
        test_output_dir: If provided, save to this directory instead of overwriting
        
    Returns:
        Tuple of (total_images, images_fixed)
    """
    pattern = "**/*.png" if recursive else "*.png"
    image_files = list(directory.glob(pattern))
    
    total = len(image_files)
    fixed = 0
    
    print(f"\nProcessing {total} images in {directory}...")
    print(f"{'DRY RUN - ' if dry_run else ''}Recursive: {recursive}")
    print("=" * 70)
    
    for i, img_path in enumerate(image_files, 1):
        try:
            was_processed, status_msg = fix_card_corners(img_path, corner_radius, dry_run, test_output_dir)
            if was_processed:
                fixed += 1
                mode = "DRY RUN" if dry_run else ("TEST" if test_output_dir else "FIXED")
                print(f"[{i}/{total}] {mode}: {img_path.name} - {status_msg}")
            else:
                if i % 100 == 0:  # Progress update for skipped files
                    print(f"[{i}/{total}] Processed... ({fixed} fixed so far)")
        except Exception as e:
            print(f"[{i}/{total}] ERROR processing {img_path.name}: {e}")
    
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  Total images: {total}")
    print(f"  Images {'that would be ' if dry_run else ''}fixed: {fixed}")
    print(f"  Images skipped: {total - fixed}")
    
    return total, fixed


def main():
    parser = argparse.ArgumentParser(
        description="Fix corner artifacts in card images by applying rounded corner masks"
    )
    parser.add_argument(
        'path',
        type=str,
        help='Path to image file or directory to process'
    )
    parser.add_argument(
        '--corner-radius',
        type=int,
        default=15,
        help='Radius of rounded corners in pixels (default: 15)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Scan and report what would be fixed without making any changes'
    )
    parser.add_argument(
        '--test-mode',
        type=str,
        metavar='DIR',
        help='Save fixed copies to this directory instead of overwriting originals'
    )
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='Do not process subdirectories'
    )
    
    args = parser.parse_args()
    
    path = Path(args.path)
    
    if not path.exists():
        print(f"Error: Path does not exist: {path}")
        return 1
    
    test_output_dir = Path(args.test_mode) if args.test_mode else None
    
    if path.is_file():
        # Process single file
        print(f"Processing single file: {path}")
        was_processed, status_msg = fix_card_corners(path, args.corner_radius, args.dry_run, test_output_dir)
        if was_processed:
            mode = "Would fix" if args.dry_run else ("Fixed (test copy)" if test_output_dir else "Fixed")
            print(f"{mode}: {path}")
            print(f"  Status: {status_msg}")
        else:
            print(f"Skipped: {path}")
            print(f"  Reason: {status_msg}")
    else:
        # Process directory
        total, fixed = process_directory(
            path,
            args.corner_radius,
            args.dry_run,
            not args.no_recursive,
            test_output_dir
        )
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
