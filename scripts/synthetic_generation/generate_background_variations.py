#!/usr/bin/env python3
"""
Generate background variations with jostled zone labels.
Creates a large pool of backgrounds with randomized zone positions for synthetic data generation.

Usage:
    python generate_background_variations.py --num-variations 10000
"""

import os
import yaml
import random
import argparse
import math
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path
import shutil

# ===================== CONFIGURATION =====================
# Auto-detect base path for Windows/Linux compatibility
import platform
if platform.system() == 'Windows':
    BASE_PATH = Path(r'c:\VS Code\FaB Code')
else:
    # Linux/RunPod - assume script is in scripts/synthetic_generation/
    BASE_PATH = Path(__file__).parent.parent.parent

BASE_IMAGE_PATH = BASE_PATH / 'data' / 'Background Perfecting' / 'images' / 'AI-Edit-High-Res-3_png.rf.b44b3276cacd96d33871b994aaf0e459.jpg'
BASE_LABEL_PATH = BASE_PATH / 'data' / 'Background Perfecting' / 'labels' / 'AI-Edit-High-Res-3_png.rf.b44b3276cacd96d33871b994aaf0e459.txt'
BASE_DATA_YAML = BASE_PATH / 'data' / 'Background Perfecting' / 'data.yaml'
OUTPUT_DIR = BASE_PATH / 'data' / 'synthetic' / 'backgrounds'

# Variation parameters
ROTATION_MAX_DEGREES = 45  # +/- max rotation (bell curve) - NOT USED (YOLO uses axis-aligned boxes)
POSITION_MAX_PIXELS = 60   # +/- max X/Y shift in pixels (bell curve)
SCALE_MAX_PERCENT = 0.10   # +/- max scale (10%)

# Classes to exclude from transformations (Combat Chain 1, 2, and Window)
EXCLUDED_CLASS_IDS = {7, 8, 24}  # Combat Chain 1 (7), Combat Chain 2 (8), Window (24)

# =========================================================

def load_data_yaml(yaml_path):
    """Load the data.yaml file to get image dimensions and class names."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data

def bell_curve_random(min_val, max_val, std_dev_factor=3):
    """
    Generate a random value using a bell curve (normal distribution).
    
    Args:
        min_val: Minimum value (corresponds to -std_dev_factor * sigma)
        max_val: Maximum value (corresponds to +std_dev_factor * sigma)
        std_dev_factor: Number of standard deviations to span the range
    
    Returns:
        Random value from normal distribution, clipped to [min_val, max_val]
    """
    # Calculate sigma such that min_val and max_val are at +/- std_dev_factor sigma
    sigma = (max_val - min_val) / (2 * std_dev_factor)
    mean = (min_val + max_val) / 2
    
    # Generate random value from normal distribution
    value = random.gauss(mean, sigma)
    
    # Clip to range
    return max(min_val, min(max_val, value))

def yolo_to_pixels(cx, cy, w, h, img_width, img_height):
    """Convert YOLO normalized coordinates to pixel coordinates."""
    x_center = cx * img_width
    y_center = cy * img_height
    box_width = w * img_width
    box_height = h * img_height
    
    return x_center, y_center, box_width, box_height

def pixels_to_yolo(x_center, y_center, box_width, box_height, img_width, img_height):
    """Convert pixel coordinates to YOLO normalized coordinates."""
    cx = x_center / img_width
    cy = y_center / img_height
    w = box_width / img_width
    h = box_height / img_height
    
    # Clip to valid range [0, 1]
    cx = max(0, min(1, cx))
    cy = max(0, min(1, cy))
    w = max(0, min(1, w))
    h = max(0, min(1, h))
    
    return cx, cy, w, h

def rotate_point(x, y, cx, cy, angle_degrees):
    """Rotate a point (x, y) around center (cx, cy) by angle_degrees."""
    angle_rad = math.radians(angle_degrees)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    # Translate to origin
    x_translated = x - cx
    y_translated = y - cy
    
    # Rotate
    x_rotated = x_translated * cos_a - y_translated * sin_a
    y_rotated = x_translated * sin_a + y_translated * cos_a
    
    # Translate back
    x_final = x_rotated + cx
    y_final = y_rotated + cy
    
    return x_final, y_final

def jostle_bbox(class_id, cx, cy, w, h, img_width, img_height):
    """
    Apply random jostling to a bounding box.
    
    Args:
        class_id: YOLO class ID
        cx, cy, w, h: YOLO normalized coordinates
        img_width, img_height: Image dimensions in pixels
    
    Returns:
        Jostled (cx, cy, w, h) in YOLO format
    """
    # Convert to pixels for easier manipulation
    x_center, y_center, box_width, box_height = yolo_to_pixels(cx, cy, w, h, img_width, img_height)
    
    # 1. POSITION SHIFT (X and Y independently)
    shift_x = bell_curve_random(-POSITION_MAX_PIXELS, POSITION_MAX_PIXELS)
    shift_y = bell_curve_random(-POSITION_MAX_PIXELS, POSITION_MAX_PIXELS)
    
    x_center += shift_x
    y_center += shift_y
    
    # 2. SCALE (uniform, both width and height)
    scale_factor = 1.0 + bell_curve_random(-SCALE_MAX_PERCENT, SCALE_MAX_PERCENT)
    box_width *= scale_factor
    box_height *= scale_factor
    
    # Convert back to YOLO format (clipped to valid range)
    cx_new, cy_new, w_new, h_new = pixels_to_yolo(
        x_center, y_center, box_width, box_height, img_width, img_height
    )
    
    return cx_new, cy_new, w_new, h_new

def generate_background_variation(base_label_lines, img_width, img_height, variation_idx):
    """
    Generate a single background variation by jostling all zone labels.
    
    Args:
        base_label_lines: List of YOLO label lines from base file
        img_width, img_height: Image dimensions
        variation_idx: Index of this variation (for seeding)
    
    Returns:
        List of jostled label lines
    """
    jostled_lines = []
    
    for line in base_label_lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        
        class_id = int(parts[0])
        cx, cy, w, h = map(float, parts[1:])
        
        # Skip jostling for excluded classes (Combat Chain 1 and 2)
        if class_id in EXCLUDED_CLASS_IDS:
            jostled_lines.append(line)
            continue
        
        # Jostle this bbox
        cx_new, cy_new, w_new, h_new = jostle_bbox(class_id, cx, cy, w, h, img_width, img_height)
        
        # Format as YOLO label
        jostled_line = f"{class_id} {cx_new:.6f} {cy_new:.6f} {w_new:.6f} {h_new:.6f}\n"
        jostled_lines.append(jostled_line)
    
    return jostled_lines

def visualize_labels(image_path, label_lines, output_path, class_names):
    """
    Draw bounding boxes on image for visualization.
    
    Args:
        image_path: Path to image file
        label_lines: List of YOLO label lines
        output_path: Where to save visualization
        class_names: List of class names from data.yaml
    """
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    img_width, img_height = img.size
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Generate colors for each class
    colors = {}
    for i in range(len(class_names)):
        # Generate distinct colors
        hue = (i * 137.5) % 360  # Golden angle for good distribution
        rgb = tuple(int(c * 255) for c in _hsv_to_rgb(hue / 360, 0.8, 0.9))
        colors[i] = rgb
    
    # Draw each bounding box
    for line in label_lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        
        class_id = int(parts[0])
        cx, cy, w, h = map(float, parts[1:])
        
        # Convert YOLO to pixel coordinates
        x_center = cx * img_width
        y_center = cy * img_height
        box_width = w * img_width
        box_height = h * img_height
        
        # Calculate corner coordinates
        x1 = x_center - box_width / 2
        y1 = y_center - box_height / 2
        x2 = x_center + box_width / 2
        y2 = y_center + box_height / 2
        
        # Choose color (red for excluded classes, otherwise class-specific)
        if class_id in EXCLUDED_CLASS_IDS:
            color = (255, 0, 0)  # Red for Combat Chain zones
            thickness = 3
        else:
            color = colors.get(class_id, (0, 255, 0))
            thickness = 2
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
        
        # Draw label
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        label = f"{class_name}"
        
        # Draw text background
        bbox = draw.textbbox((x1, y1 - 20), label, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((x1, y1 - 20), label, fill=(255, 255, 255), font=font)
    
    img.save(output_path)
    print(f"  Saved visualization: {output_path}")

def _hsv_to_rgb(h, s, v):
    """Convert HSV color to RGB."""
    if s == 0.0:
        return v, v, v
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    if i == 5:
        return v, p, q

def main():
    parser = argparse.ArgumentParser(description='Generate background variations with jostled zone labels')
    parser.add_argument('--num-variations', type=int, default=10000, 
                       help='Number of background variations to generate')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations with bounding boxes drawn')
    parser.add_argument('--visualize-count', type=int, default=10,
                       help='Number of samples to visualize (default: 10)')
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    print(f"Generating {args.num_variations} background variations...")
    print(f"Parameters:")
    print(f"  - Position shift: +/- {POSITION_MAX_PIXELS}px (bell curve)")
    print(f"  - Scale: +/- {SCALE_MAX_PERCENT*100}% (bell curve)")
    print(f"  - Excluded classes: Combat Chain 1, Combat Chain 2, Window (no transforms)")
    print()
    
    # Load data.yaml to get image dimensions
    data_yaml = load_data_yaml(str(BASE_DATA_YAML))
    
    # Get image dimensions from the base image
    base_image = Image.open(str(BASE_IMAGE_PATH))
    img_width, img_height = base_image.size
    print(f"Base image dimensions: {img_width}x{img_height}")
    
    # Load base label file
    with open(str(BASE_LABEL_PATH), 'r') as f:
        base_label_lines = f.readlines()
    
    print(f"Base label contains {len(base_label_lines)} zone annotations")
    
    # Create output directories
    output_images_dir = OUTPUT_DIR / 'images'
    output_labels_dir = OUTPUT_DIR / 'labels'
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualization directory if needed
    if args.visualize:
        output_viz_dir = OUTPUT_DIR / 'visualizations'
        output_viz_dir.mkdir(parents=True, exist_ok=True)
        print(f"Visualization enabled: will generate {args.visualize_count} samples")
        print()
    
    # Generate variations
    print(f"\nGenerating variations...")
    for i in range(args.num_variations):
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{args.num_variations} variations...")
        
        # Generate jostled labels
        jostled_labels = generate_background_variation(
            base_label_lines, img_width, img_height, i
        )
        
        # Copy base image (we're not actually modifying the image, just the labels)
        output_image_path = output_images_dir / f'bg_{i:06d}.png'
        shutil.copy(str(BASE_IMAGE_PATH), str(output_image_path))
        
        # Write jostled label file
        output_label_path = output_labels_dir / f'bg_{i:06d}.txt'
        with open(str(output_label_path), 'w') as f:
            f.writelines(jostled_labels)
        
        # Generate visualization if requested
        if args.visualize and i < args.visualize_count:
            output_viz_path = output_viz_dir / f'bg_{i:06d}_viz.png'
            visualize_labels(str(output_image_path), jostled_labels, str(output_viz_path), data_yaml['names'])
    
    # Copy data.yaml to output directory
    output_data_yaml = OUTPUT_DIR / 'data.yaml'
    shutil.copy(str(BASE_DATA_YAML), str(output_data_yaml))
    
    print(f"\n{'='*60}")
    print(f"Generation complete!")
    print(f"Generated {args.num_variations} background variations")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"  - Images: {output_images_dir}")
    print(f"  - Labels: {output_labels_dir}")
    print(f"  - data.yaml: {output_data_yaml}")
    if args.visualize:
        print(f"  - Visualizations: {output_viz_dir} ({args.visualize_count} samples)")
        print(f"\nCombat Chain 1, 2, and Window zones are shown in RED (not transformed)")
        print(f"All other zones are shown in various colors (transformed)")
    print()
    print("These backgrounds are ready to use as templates for synthetic card generation!")

if __name__ == '__main__':
    main()
