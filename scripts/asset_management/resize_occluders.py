"""
Check and resize occluder images to ensure they fall within 10-40px range.
Maintains aspect ratio and transparency.
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image

def resize_occluder_if_needed(image_path, min_size=10, max_size=40):
    """
    Resize occluder image if it's outside the acceptable size range.
    Maintains aspect ratio and transparency.
    
    Args:
        image_path: Path to occluder image
        min_size: Minimum dimension (width or height)
        max_size: Maximum dimension (width or height)
        
    Returns:
        (was_resized, original_size, new_size)
    """
    # Load image with alpha channel
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return False, None, None, "Failed to load"
    
    h, w = img.shape[:2]
    original_size = (w, h)
    
    # Check if resize is needed
    max_dim = max(w, h)
    min_dim = min(w, h)
    
    needs_resize = False
    scale = 1.0
    
    if max_dim > max_size:
        # Too large - scale down
        scale = max_size / max_dim
        needs_resize = True
        reason = f"Too large ({max_dim}px)"
    elif min_dim < min_size:
        # Too small - scale up
        scale = min_size / min_dim
        needs_resize = True
        reason = f"Too small ({min_dim}px)"
    else:
        return False, original_size, original_size, "OK"
    
    # Resize
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Use high-quality interpolation
    if scale < 1.0:
        interpolation = cv2.INTER_AREA  # Better for downscaling
    else:
        interpolation = cv2.INTER_LANCZOS4  # Better for upscaling
    
    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    
    # Save resized image
    cv2.imwrite(str(image_path), resized)
    
    return True, original_size, (new_w, new_h), reason

def check_and_resize_all_occluders(occluders_dir, min_size=10, max_size=40):
    """
    Check all occluder images and resize if needed.
    
    Args:
        occluders_dir: Directory containing occluder images
        min_size: Minimum dimension
        max_size: Maximum dimension
    """
    occluders_path = Path(occluders_dir)
    
    # Find all PNG images
    occluder_files = list(occluders_path.glob('*.png'))
    print(f"Found {len(occluder_files)} occluder images in {occluders_path}\n")
    
    if len(occluder_files) == 0:
        print("No occluder images found!")
        return
    
    # Process each occluder
    resized_count = 0
    too_small_count = 0
    too_large_count = 0
    ok_count = 0
    error_count = 0
    
    print(f"Checking size range: {min_size}px - {max_size}px")
    print(f"{'='*70}")
    
    for occluder_file in sorted(occluder_files):
        was_resized, orig_size, new_size, reason = resize_occluder_if_needed(
            occluder_file, min_size, max_size
        )
        
        if reason == "Failed to load":
            error_count += 1
            print(f"✗ {occluder_file.name}: {reason}")
        elif was_resized:
            resized_count += 1
            if "small" in reason:
                too_small_count += 1
            else:
                too_large_count += 1
            print(f"↻ {occluder_file.name}: {reason} -> {orig_size} → {new_size}")
        else:
            ok_count += 1
    
    print(f"{'='*70}")
    print(f"\n✓ Processing complete!")
    print(f"  OK (no resize needed): {ok_count}")
    print(f"  Resized (too small): {too_small_count}")
    print(f"  Resized (too large): {too_large_count}")
    print(f"  Total resized: {resized_count}")
    print(f"  Errors: {error_count}")
    print(f"\nAll occluders now fit within {min_size}px - {max_size}px range")

if __name__ == "__main__":
    # Path to occluders directory
    occluders_dir = Path(__file__).parent.parent.parent / "data" / "occluders"
    
    # Check and resize all occluders to 10-40px range (sized for occluders)
    check_and_resize_all_occluders(occluders_dir, min_size=10, max_size=40)
