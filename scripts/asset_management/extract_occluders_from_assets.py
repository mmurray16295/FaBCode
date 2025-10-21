"""
Extract individual occluders (dice/counters) from asset images.
Uses background removal and contour detection to segment individual objects.
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import json

def remove_background(image):
    """
    Remove background from image using various techniques.
    Returns image with alpha channel (transparent background).
    """
    # Convert to RGB if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    
    # Create mask using multiple methods and combine them
    
    # Method 1: White background detection (common for product photos)
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for white background
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Method 2: GrabCut for more complex backgrounds
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # Define a rectangle around the image (assume objects are not at the very edge)
    h, w = image.shape[:2]
    rect = (5, 5, w-10, h-10)
    
    try:
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        grabcut_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    except:
        grabcut_mask = np.ones(image.shape[:2], dtype=np.uint8)
    
    # Combine masks
    final_mask = cv2.bitwise_or(white_mask, cv2.bitwise_not(grabcut_mask * 255))
    final_mask = cv2.bitwise_not(final_mask)
    
    # Clean up the mask with morphological operations
    kernel = np.ones((3, 3), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Create RGBA image
    rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = final_mask
    
    return rgba

def find_objects(image_with_alpha):
    """
    Find individual objects in an image using contour detection.
    Returns list of bounding boxes (x, y, w, h).
    """
    # Extract alpha channel as mask
    if image_with_alpha.shape[2] == 4:
        mask = image_with_alpha[:, :, 3]
    else:
        # If no alpha, create mask from non-white areas
        gray = cv2.cvtColor(image_with_alpha, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get bounding boxes for each contour
    bboxes = []
    min_area = 500  # Minimum area to consider (filter out noise)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            bboxes.append((x, y, w, h))
    
    return bboxes

def extract_and_save_objects(image_path, output_dir, min_size=30, padding=5):
    """
    Extract individual objects from an asset image and save them separately.
    
    Args:
        image_path: Path to source image
        output_dir: Directory to save extracted objects
        min_size: Minimum width/height for an object to be saved
        padding: Pixels of padding around each object
    """
    print(f"\nProcessing: {image_path.name}")
    
    # Load image
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"  ✗ Failed to load image")
        return 0
    
    # Remove background if image doesn't have alpha channel
    if image.shape[2] == 3:
        print(f"  Removing background...")
        image_with_alpha = remove_background(image)
    else:
        image_with_alpha = image
    
    # Find objects
    print(f"  Finding objects...")
    bboxes = find_objects(image_with_alpha)
    print(f"  Found {len(bboxes)} objects")
    
    if len(bboxes) == 0:
        print(f"  ✗ No objects detected")
        return 0
    
    # Extract and save each object
    saved_count = 0
    base_name = image_path.stem
    
    for idx, (x, y, w, h) in enumerate(bboxes):
        # Skip if too small
        if w < min_size or h < min_size:
            continue
        
        # Add padding
        x_pad = max(0, x - padding)
        y_pad = max(0, y - padding)
        w_pad = min(image_with_alpha.shape[1] - x_pad, w + 2 * padding)
        h_pad = min(image_with_alpha.shape[0] - y_pad, h + 2 * padding)
        
        # Extract object
        obj = image_with_alpha[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
        
        # Save as PNG with alpha channel
        output_path = output_dir / f"{base_name}_object_{idx:03d}.png"
        cv2.imwrite(str(output_path), obj)
        saved_count += 1
    
    print(f"  ✓ Saved {saved_count} objects")
    return saved_count

def process_all_assets(assets_dir, output_dir, min_size=30, padding=5):
    """
    Process all asset images in a directory.
    
    Args:
        assets_dir: Directory containing asset images
        output_dir: Directory to save extracted occluders
        min_size: Minimum object size to extract
        padding: Padding around each object
    """
    assets_path = Path(assets_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_extensions = ['.png', '.jpg', '.jpeg', '.webp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(assets_path.glob(f'*{ext}'))
        image_files.extend(assets_path.glob(f'*{ext.upper()}'))
    
    print(f"Found {len(image_files)} asset images in {assets_path}")
    
    # Process each image
    total_extracted = 0
    for image_file in sorted(image_files):
        count = extract_and_save_objects(image_file, output_path, min_size, padding)
        total_extracted += count
    
    print(f"\n{'='*60}")
    print(f"✓ Extraction complete!")
    print(f"  Total objects extracted: {total_extracted}")
    print(f"  Output directory: {output_path}")
    print(f"{'='*60}")
    
    return total_extracted

if __name__ == "__main__":
    # Paths
    assets_dir = Path(__file__).parent.parent.parent / "data" / "images" / "ZZAssets"
    output_dir = Path(__file__).parent.parent.parent / "data" / "occluders"
    
    # Process all assets
    process_all_assets(assets_dir, output_dir, min_size=30, padding=5)
