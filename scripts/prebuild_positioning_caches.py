#!/usr/bin/env python3
"""
Pre-build positioning caches for all YouTube background templates.
This allows parallel generation processes to load pre-computed caches
instead of rebuilding them, saving significant time.
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path

# Add parent directory to path to import from generate_synthetic_playmat_screenshots
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

YOUTUBE_LABELED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'images', 'YouTube_Labeled')
CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'positioning_cache')


def ensure_dir(path):
    """Ensure directory exists."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def parse_yolo_label(label_path, img_width, img_height):
    """
    Parse YOLO format label file.
    Returns list of boxes: [(class_id, x_center, y_center, width, height), ...]
    All values are in absolute pixel coordinates.
    """
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height
                boxes.append((class_id, x_center, y_center, width, height))
    
    return boxes


def compute_iou(box1, box2):
    """
    Compute IoU between two boxes.
    Each box is (x_center, y_center, width, height).
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to corner coordinates
    x1_min, y1_min = x1 - w1/2, y1 - h1/2
    x1_max, y1_max = x1 + w1/2, y1 + h1/2
    x2_min, y2_min = x2 - w2/2, y2 - h2/2
    x2_max, y2_max = x2 + w2/2, y2 + h2/2
    
    # Compute intersection
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    inter_width = max(0, inter_xmax - inter_xmin)
    inter_height = max(0, inter_ymax - inter_ymin)
    inter_area = inter_width * inter_height
    
    # Compute union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def build_positioning_cache(image_path, label_path, ref_boxes):
    """
    Build positioning cache for a template.
    Tests grid of positions to find valid card placements that don't overlap with Ref boxes.
    
    Returns dict with valid positions and metadata.
    """
    # Load image to get dimensions
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not load image: {image_path}")
        return None
    
    img_height, img_width = img.shape[:2]
    
    # Calculate average Ref box size
    avg_width = np.mean([box[3] for box in ref_boxes])
    avg_height = np.mean([box[4] for box in ref_boxes])
    
    # Define grid search parameters
    # Test positions at 10% intervals across the image
    grid_step_x = img_width // 10
    grid_step_y = img_height // 10
    
    # Test various card sizes (80% to 120% of Ref box size)
    size_multipliers = [0.8, 0.9, 1.0, 1.1, 1.2]
    
    # IoU threshold for overlap detection
    iou_threshold = 0.1
    
    valid_positions = []
    
    print(f"[cache] Building positioning cache for {len(ref_boxes)} boxes...")
    
    # Test each position
    for y_center in range(grid_step_y, img_height - grid_step_y, grid_step_y):
        for x_center in range(grid_step_x, img_width - grid_step_x, grid_step_x):
            for size_mult in size_multipliers:
                test_width = avg_width * size_mult
                test_height = avg_height * size_mult
                
                # Check if this position overlaps with any Ref box
                test_box = (x_center, y_center, test_width, test_height)
                
                is_valid = True
                for ref_box in ref_boxes:
                    # ref_box is (class_id, x_center, y_center, width, height)
                    ref_box_coords = (ref_box[1], ref_box[2], ref_box[3], ref_box[4])
                    
                    if compute_iou(test_box, ref_box_coords) > iou_threshold:
                        is_valid = False
                        break
                
                if is_valid:
                    # Check if box is within image bounds
                    x_min = x_center - test_width/2
                    y_min = y_center - test_height/2
                    x_max = x_center + test_width/2
                    y_max = y_center + test_height/2
                    
                    if x_min >= 0 and y_min >= 0 and x_max <= img_width and y_max <= img_height:
                        valid_positions.append({
                            'x_center': float(x_center),
                            'y_center': float(y_center),
                            'width': float(test_width),
                            'height': float(test_height),
                            'size_mult': float(size_mult)
                        })
    
    print(f"[cache] Found {len(valid_positions)} valid positions")
    
    cache_data = {
        'image_path': image_path,
        'image_width': img_width,
        'image_height': img_height,
        'ref_boxes': [
            {
                'x_center': float(box[1]),
                'y_center': float(box[2]),
                'width': float(box[3]),
                'height': float(box[4])
            }
            for box in ref_boxes
        ],
        'avg_ref_width': float(avg_width),
        'avg_ref_height': float(avg_height),
        'valid_positions': valid_positions,
        'num_valid_positions': len(valid_positions)
    }
    
    return cache_data


def process_template(image_path, label_path, split_name):
    """Process a single template and save its cache."""
    # Get template name from image filename
    template_name = os.path.basename(image_path)
    cache_filename = template_name.replace('.jpg', '_positioning.json').replace('.png', '_positioning.json')
    cache_path = os.path.join(CACHE_DIR, cache_filename)
    
    # Check if cache already exists
    if os.path.exists(cache_path):
        print(f"[{split_name}] Cache already exists: {cache_filename}")
        return True
    
    print(f"[{split_name}] Processing: {template_name}")
    
    # Load image to get dimensions
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not load image: {image_path}")
        return False
    
    img_height, img_width = img.shape[:2]
    
    # Parse label file
    boxes = parse_yolo_label(label_path, img_width, img_height)
    
    # Filter for Ref boxes (class 1)
    ref_boxes = [box for box in boxes if box[0] == 1]
    
    if not ref_boxes:
        print(f"[WARNING] No Ref boxes found in {label_path}")
        return False
    
    print(f"[{split_name}] Found {len(ref_boxes)} Ref boxes")
    
    # Build cache
    cache_data = build_positioning_cache(image_path, label_path, ref_boxes)
    
    if cache_data is None:
        return False
    
    # Save cache
    with open(cache_path, 'w') as f:
        json.dump(cache_data, f, indent=2)
    
    print(f"[{split_name}] ✓ Saved cache: {cache_filename}")
    return True


def main():
    """Pre-build all positioning caches."""
    ensure_dir(CACHE_DIR)
    
    print("=" * 70)
    print("Pre-building Positioning Caches")
    print("=" * 70)
    print(f"YouTube Labeled Dir: {YOUTUBE_LABELED_DIR}")
    print(f"Cache Dir: {CACHE_DIR}")
    print()
    
    splits = ['train', 'valid', 'test']
    total_processed = 0
    total_success = 0
    
    for split in splits:
        images_dir = os.path.join(YOUTUBE_LABELED_DIR, split, 'images')
        labels_dir = os.path.join(YOUTUBE_LABELED_DIR, split, 'labels')
        
        if not os.path.exists(images_dir):
            print(f"[WARNING] Split directory not found: {images_dir}")
            continue
        
        # Get all images
        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
        
        print(f"\n=== Processing {split.upper()} split ({len(image_files)} templates) ===\n")
        
        for image_file in sorted(image_files):
            image_path = os.path.join(images_dir, image_file)
            label_file = image_file.replace('.jpg', '.txt').replace('.png', '.txt')
            label_path = os.path.join(labels_dir, label_file)
            
            total_processed += 1
            if process_template(image_path, label_path, split):
                total_success += 1
            print()
    
    print("=" * 70)
    print(f"Pre-building Complete!")
    print(f"Total templates processed: {total_processed}")
    print(f"Caches successfully built: {total_success}")
    print(f"Cache directory: {CACHE_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
