"""
Generate Random Playmat Images for YOLO Training
=================================================

Generates synthetic training data with random card placement on solid color backgrounds.
No templates needed - purely random positioning with overlap control.

Features:
- Random solid color backgrounds (table/mat colors)
- 10-20 cards per scene at random positions
- Baseline 150x110px cards, 50% get random ±50% scale
- Completely random rotation (0-359°)
- Max 25% overlap between cards
- Random brightness/darkness per card
- Weighted card selection with popularity bands
- Proper YOLO label tracking
"""

import os
import sys
import glob
import random
import uuid
import json
import yaml
import math
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

# Output directory for generated images
OUTPUT_BASE_DIR = r'c:\VS Code\FaB Code\data\synthetic_2'

# Card image directories
CARD_SET_DIRS = [
    r'c:\VS Code\FaB Code\data\images\WTR',
    r'c:\VS Code\FaB Code\data\images\ARC',
    r'c:\VS Code\FaB Code\data\images\CRU',
    r'c:\VS Code\FaB Code\data\images\HVY',
    r'c:\VS Code\FaB Code\data\images\SEA',
]

# Card popularity weights file
POPULARITY_WEIGHTS_PATH = r'c:\VS Code\FaB Code\data\card_popularity_weights.json'

# Image dimensions (fixed for consistency)
IMG_WIDTH = 1920
IMG_HEIGHT = 1080

# Card baseline dimensions (from analysis of Ref boxes)
# Only width is specified - height calculated automatically to maintain aspect ratio
CARD_BASE_WIDTH = 150

# Cards per scene
CARDS_PER_SCENE_MIN = 10
CARDS_PER_SCENE_MAX = 20

# Maximum overlap allowed (as fraction of smaller card area)
MAX_OVERLAP_RATIO = 0.25

# Split ratios
TRAIN_RATIO = 0.70
VALID_RATIO = 0.20
TEST_RATIO = 0.10

# ============================================================================
# BACKGROUND GENERATION
# ============================================================================

def generate_random_background(width, height):
    """
    Generate a random textured background with lots of variety.
    
    Returns:
        PIL Image with random creative background
    """
    # Realistic table/mat colors
    base_colors = [
        (34, 87, 50),      # Green felt
        (25, 60, 95),      # Blue mat
        (20, 20, 25),      # Black table
        (60, 45, 35),      # Brown wood
        (70, 70, 75),      # Gray mat
        (45, 25, 15),      # Dark brown
        (15, 45, 35),      # Dark teal
        (80, 50, 40),      # Tan/beige
        (30, 50, 70),      # Navy blue
        (50, 40, 30),      # Darker brown
        (90, 30, 30),      # Dark red
        (30, 30, 60),      # Deep blue
        (40, 60, 40),      # Forest green
        (55, 35, 50),      # Purple-brown
        (35, 55, 60),      # Teal-blue
    ]
    
    # Lots of texture types for variety
    texture_type = random.choice([
        'solid', 'noise', 'gradient', 'vignette', 'noise+gradient',
        'wood_grain', 'stripes', 'checkerboard', 'color_slices',
        'random_lines', 'dots', 'cross_hatch', 'radial_gradient',
        'wave_pattern', 'diagonal_stripes', 'heavy_noise'
    ])
    
    if texture_type == 'solid':
        # Simple solid color
        base_color = random.choice(base_colors)
        base_color = tuple(max(0, min(255, c + random.randint(-15, 15))) for c in base_color)
        img = Image.new('RGB', (width, height), base_color)
    
    elif texture_type == 'noise':
        # Add noise texture
        base_color = random.choice(base_colors)
        base_color = tuple(max(0, min(255, c + random.randint(-15, 15))) for c in base_color)
        img_array = np.full((height, width, 3), base_color, dtype=np.int16)
        noise_strength = random.randint(10, 35)
        noise = np.random.randint(-noise_strength, noise_strength, (height, width, 3), dtype=np.int16)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)
    
    elif texture_type == 'heavy_noise':
        # Heavy noise texture
        base_color = random.choice(base_colors)
        img_array = np.full((height, width, 3), base_color, dtype=np.int16)
        noise_strength = random.randint(40, 70)
        noise = np.random.randint(-noise_strength, noise_strength, (height, width, 3), dtype=np.int16)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)
    
    elif texture_type == 'gradient':
        # Gradient background
        base_color = random.choice(base_colors)
        img_array = np.full((height, width, 3), base_color, dtype=np.uint8)
        gradient_direction = random.choice(['horizontal', 'vertical', 'diagonal'])
        gradient_strength = random.uniform(0.3, 0.7)
        
        if gradient_direction == 'horizontal':
            gradient = np.linspace(1 - gradient_strength, 1 + gradient_strength, width)
            gradient = np.tile(gradient, (height, 1))
        elif gradient_direction == 'vertical':
            gradient = np.linspace(1 - gradient_strength, 1 + gradient_strength, height)
            gradient = np.tile(gradient.reshape(-1, 1), (1, width))
        else:  # diagonal
            x = np.linspace(1 - gradient_strength, 1 + gradient_strength, width)
            y = np.linspace(1 - gradient_strength, 1 + gradient_strength, height)
            gradient = (np.tile(x, (height, 1)) + np.tile(y.reshape(-1, 1), (1, width))) / 2
        
        gradient = gradient[:, :, np.newaxis]
        img_array = np.clip(img_array * gradient, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)
    
    elif texture_type == 'radial_gradient':
        # Radial gradient from center
        base_color = random.choice(base_colors)
        img_array = np.full((height, width, 3), base_color, dtype=np.float32)
        
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height / 2, width / 2
        max_dist = np.sqrt(center_x**2 + center_y**2)
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2) / max_dist
        
        gradient_strength = random.uniform(0.4, 0.8)
        gradient = 1.0 + (dist_from_center * gradient_strength - gradient_strength/2)
        img_array = img_array * gradient[:, :, np.newaxis]
        img = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
    
    elif texture_type == 'vignette':
        # Vignette effect
        base_color = random.choice(base_colors)
        img_array = np.full((height, width, 3), base_color, dtype=np.float32)
        
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height / 2, width / 2
        max_dist = np.sqrt(center_x**2 + center_y**2)
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2) / max_dist
        
        vignette_strength = random.uniform(0.3, 0.6)
        vignette = 1.0 - (dist_from_center * vignette_strength)
        vignette = np.clip(vignette, 0, 1)
        
        img_array = img_array * vignette[:, :, np.newaxis]
        img = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
    
    elif texture_type == 'noise+gradient':
        # Combine noise and gradient
        base_color = random.choice(base_colors)
        img_array = np.full((height, width, 3), base_color, dtype=np.int16)
        noise_strength = random.randint(12, 25)
        noise = np.random.randint(-noise_strength, noise_strength, (height, width, 3), dtype=np.int16)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        
        gradient_direction = random.choice(['horizontal', 'vertical', 'diagonal'])
        gradient_strength = random.uniform(0.2, 0.4)
        
        if gradient_direction == 'horizontal':
            gradient = np.linspace(1 - gradient_strength, 1 + gradient_strength, width)
            gradient = np.tile(gradient, (height, 1))
        elif gradient_direction == 'vertical':
            gradient = np.linspace(1 - gradient_strength, 1 + gradient_strength, height)
            gradient = np.tile(gradient.reshape(-1, 1), (1, width))
        else:
            x = np.linspace(1 - gradient_strength, 1 + gradient_strength, width)
            y = np.linspace(1 - gradient_strength, 1 + gradient_strength, height)
            gradient = (np.tile(x, (height, 1)) + np.tile(y.reshape(-1, 1), (1, width))) / 2
        
        gradient = gradient[:, :, np.newaxis]
        img_array = np.clip(img_array * gradient, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)
    
    elif texture_type == 'wood_grain':
        # Simulated wood grain pattern
        base_color = random.choice([(60, 45, 35), (45, 25, 15), (50, 40, 30), (80, 50, 40)])
        img_array = np.full((height, width, 3), base_color, dtype=np.float32)
        
        # Create wood grain lines using sine waves
        frequency = random.uniform(0.01, 0.03)
        amplitude = random.uniform(15, 35)
        x = np.arange(width)
        y = np.arange(height).reshape(-1, 1)
        
        # Create grain pattern (ensure correct shape)
        grain = np.sin(x * frequency * 2 * np.pi + np.sin(y * frequency * 0.5) * 5) * amplitude
        # grain is already (height, width) from broadcasting
        
        img_array = img_array + grain[:, :, np.newaxis]
        img = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
    
    elif texture_type == 'stripes':
        # Horizontal or vertical stripes
        direction = random.choice(['horizontal', 'vertical'])
        num_stripes = random.randint(5, 12)
        colors = random.sample(base_colors, min(3, len(base_colors)))
        
        img_array = np.zeros((height, width, 3), dtype=np.uint8)
        
        if direction == 'horizontal':
            stripe_height = height // num_stripes
            for i in range(num_stripes):
                color = random.choice(colors)
                color = tuple(max(0, min(255, c + random.randint(-20, 20))) for c in color)
                start_y = i * stripe_height
                end_y = (i + 1) * stripe_height if i < num_stripes - 1 else height
                img_array[start_y:end_y, :] = color
        else:  # vertical
            stripe_width = width // num_stripes
            for i in range(num_stripes):
                color = random.choice(colors)
                color = tuple(max(0, min(255, c + random.randint(-20, 20))) for c in color)
                start_x = i * stripe_width
                end_x = (i + 1) * stripe_width if i < num_stripes - 1 else width
                img_array[:, start_x:end_x] = color
        
        img = Image.fromarray(img_array)
    
    elif texture_type == 'diagonal_stripes':
        # Diagonal stripes
        num_stripes = random.randint(8, 20)
        colors = random.sample(base_colors, min(2, len(base_colors)))
        
        img_array = np.zeros((height, width, 3), dtype=np.uint8)
        y, x = np.ogrid[:height, :width]
        
        # Create diagonal pattern
        diagonal = (x + y) // (max(width, height) // num_stripes)
        
        for i in range(num_stripes):
            mask = (diagonal == i)
            color = colors[i % len(colors)]
            color = tuple(max(0, min(255, c + random.randint(-15, 15))) for c in color)
            img_array[mask] = color
        
        img = Image.fromarray(img_array)
    
    elif texture_type == 'checkerboard':
        # Checkerboard pattern
        square_size = random.randint(80, 200)
        colors = random.sample(base_colors, 2)
        
        img_array = np.zeros((height, width, 3), dtype=np.uint8)
        y, x = np.ogrid[:height, :width]
        
        checker = ((x // square_size) + (y // square_size)) % 2
        
        color1 = tuple(max(0, min(255, c + random.randint(-10, 10))) for c in colors[0])
        color2 = tuple(max(0, min(255, c + random.randint(-10, 10))) for c in colors[1])
        
        img_array[checker == 0] = color1
        img_array[checker == 1] = color2
        
        img = Image.fromarray(img_array)
    
    elif texture_type == 'color_slices':
        # Split into random slices with different colors
        num_slices = random.randint(5, 10)
        direction = random.choice(['horizontal', 'vertical'])
        colors = random.sample(base_colors, min(num_slices, len(base_colors)))
        
        img_array = np.zeros((height, width, 3), dtype=np.uint8)
        
        if direction == 'horizontal':
            slice_positions = sorted([random.randint(0, height) for _ in range(num_slices - 1)])
            slice_positions = [0] + slice_positions + [height]
            
            for i in range(len(slice_positions) - 1):
                color = colors[i % len(colors)]
                color = tuple(max(0, min(255, c + random.randint(-25, 25))) for c in color)
                img_array[slice_positions[i]:slice_positions[i+1], :] = color
        else:  # vertical
            slice_positions = sorted([random.randint(0, width) for _ in range(num_slices - 1)])
            slice_positions = [0] + slice_positions + [width]
            
            for i in range(len(slice_positions) - 1):
                color = colors[i % len(colors)]
                color = tuple(max(0, min(255, c + random.randint(-25, 25))) for c in color)
                img_array[:, slice_positions[i]:slice_positions[i+1]] = color
        
        img = Image.fromarray(img_array)
    
    elif texture_type == 'random_lines':
        # Random lines across background
        base_color = random.choice(base_colors)
        img = Image.new('RGB', (width, height), base_color)
        draw = ImageDraw.Draw(img)
        
        num_lines = random.randint(20, 60)
        line_color_base = random.choice(base_colors)
        
        for _ in range(num_lines):
            x1, y1 = random.randint(0, width), random.randint(0, height)
            x2, y2 = random.randint(0, width), random.randint(0, height)
            line_color = tuple(max(0, min(255, c + random.randint(-40, 40))) for c in line_color_base)
            line_width = random.randint(1, 4)
            draw.line([(x1, y1), (x2, y2)], fill=line_color, width=line_width)
    
    elif texture_type == 'dots':
        # Random dots pattern
        base_color = random.choice(base_colors)
        img = Image.new('RGB', (width, height), base_color)
        draw = ImageDraw.Draw(img)
        
        num_dots = random.randint(100, 300)
        dot_color_base = random.choice(base_colors)
        
        for _ in range(num_dots):
            x, y = random.randint(0, width), random.randint(0, height)
            radius = random.randint(2, 8)
            dot_color = tuple(max(0, min(255, c + random.randint(-30, 30))) for c in dot_color_base)
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=dot_color)
    
    elif texture_type == 'cross_hatch':
        # Cross-hatch pattern
        base_color = random.choice(base_colors)
        img = Image.new('RGB', (width, height), base_color)
        draw = ImageDraw.Draw(img)
        
        line_color = tuple(max(0, min(255, c + random.randint(-50, 50))) for c in base_color)
        spacing = random.randint(30, 80)
        
        # Horizontal lines
        for y in range(0, height, spacing):
            draw.line([(0, y), (width, y)], fill=line_color, width=random.randint(1, 3))
        
        # Vertical lines
        for x in range(0, width, spacing):
            draw.line([(x, 0), (x, height)], fill=line_color, width=random.randint(1, 3))
    
    elif texture_type == 'wave_pattern':
        # Wave pattern
        base_color = random.choice(base_colors)
        img_array = np.full((height, width, 3), base_color, dtype=np.float32)
        
        # Create wave pattern
        y = np.arange(height).reshape(-1, 1)
        x = np.arange(width)
        
        frequency = random.uniform(0.005, 0.02)
        amplitude = random.uniform(20, 50)
        
        wave = np.sin(y * frequency * 2 * np.pi) * amplitude
        wave = np.tile(wave, (1, width))
        
        img_array = img_array + wave[:, :, np.newaxis]
        img = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
    
    return img

# ============================================================================
# CARD UTILITIES
# ============================================================================

def load_card_files(card_dirs):
    """Load all card PNG files from specified directories."""
    files = []  # list of (dir, filename)
    for d in card_dirs:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.lower().endswith('.png'):
                files.append((d, f))
    return files

def canonicalize_name(name: str) -> str:
    """
    Return card name unchanged to preserve set codes.
    Set codes distinguish different rarity variants with different popularity.
    The weights file uses format: CardName_SETXXX
    """
    return name

def load_class_index(path):
    """Load or create class index mapping class names to IDs."""
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
            return {name: idx for idx, name in enumerate(data['names'])}
    return {}

def save_class_index(path, name_to_id):
    """Save class index to YAML file."""
    names = [''] * len(name_to_id)
    for name, idx in name_to_id.items():
        names[idx] = name
    
    data = {
        'names': names,
        'nc': len(names),
    }
    
    with open(path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)

def load_popularity_weights(path):
    """Load card popularity weights from JSON file."""
    if not os.path.exists(path):
        print(f"[warning] Popularity weights file not found: {path}")
        return {}
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract weights section
    if 'weights' not in data:
        print(f"[warning] No 'weights' section in popularity file")
        return {}
    
    weights_section = data['weights']
    
    # Convert to dictionary with ranks
    # Sort by total_weight descending to assign ranks
    sorted_cards = sorted(
        weights_section.items(),
        key=lambda x: x[1]['total_weight'],
        reverse=True
    )
    
    weights_dict = {}
    for rank, (card_name, weight_info) in enumerate(sorted_cards, start=1):
        weights_dict[card_name] = {
            'weight': weight_info['total_weight'],
            'rank': rank
        }
    
    return weights_dict

def filter_cards_by_popularity(card_files, weights_dict, min_rank=None, max_rank=None):
    """Filter card files to only include those within popularity rank range."""
    if not weights_dict or (min_rank is None and max_rank is None):
        return card_files
    
    filtered = []
    for set_dir, filename in card_files:
        raw_name = os.path.splitext(filename)[0]
        card_name = canonicalize_name(raw_name)
        
        if card_name in weights_dict:
            rank = weights_dict[card_name]['rank']
            
            # Check if rank is within specified range
            if min_rank is not None and rank < min_rank:
                continue
            if max_rank is not None and rank > max_rank:
                continue
            
            filtered.append((set_dir, filename))
    
    return filtered

def weighted_card_choice(card_files, weights_dict):
    """Select a card using popularity weights."""
    if not weights_dict:
        return random.choice(card_files)
    
    # Build weights list matching card_files order
    weights = []
    for set_dir, filename in card_files:
        raw_name = os.path.splitext(filename)[0]
        card_name = canonicalize_name(raw_name)
        
        if card_name in weights_dict:
            weights.append(weights_dict[card_name]['weight'])
        else:
            weights.append(0.1)  # Small default weight for unranked cards
    
    # Normalize weights
    total = sum(weights)
    if total == 0:
        return random.choice(card_files)
    
    weights = [w / total for w in weights]
    
    # Weighted random choice
    return random.choices(card_files, weights=weights, k=1)[0]

# ============================================================================
# CARD PLACEMENT & OVERLAP CHECKING
# ============================================================================

def calculate_overlap_ratio(box1, box2):
    """
    Calculate overlap ratio between two bounding boxes.
    Returns overlap area as fraction of smaller box area.
    
    Args:
        box1, box2: (x1, y1, x2, y2) tuples
    
    Returns:
        float: overlap ratio (0 = no overlap, 1 = complete overlap of smaller box)
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0  # No overlap
    
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate areas
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Return overlap as fraction of smaller box
    smaller_area = min(area1, area2)
    return intersection_area / smaller_area if smaller_area > 0 else 0.0

def find_valid_position(placed_boxes, card_w, card_h, img_w, img_h, max_overlap, max_attempts=100):
    """
    Find a valid random position for a card with acceptable overlap.
    
    Returns:
        (x1, y1, x2, y2) tuple or None if no valid position found
    """
    for _ in range(max_attempts):
        # Random position (ensure card stays within image bounds)
        x1 = random.randint(0, max(1, img_w - card_w))
        y1 = random.randint(0, max(1, img_h - card_h))
        x2 = x1 + card_w
        y2 = y1 + card_h
        
        new_box = (x1, y1, x2, y2)
        
        # Check overlap with all placed boxes
        valid = True
        for placed_box in placed_boxes:
            overlap = calculate_overlap_ratio(new_box, placed_box)
            if overlap > max_overlap:
                valid = False
                break
        
        if valid:
            return new_box
    
    return None  # Couldn't find valid position

# ============================================================================
# CARD PROCESSING
# ============================================================================

def process_card(card_img, target_width, rotation_angle):
    """
    Process card image: scale to target width (maintaining aspect ratio), rotate, apply random brightness.
    
    Args:
        card_img: PIL Image (RGBA)
        target_width: Target width in pixels (height calculated to maintain aspect ratio)
        rotation_angle: Rotation in degrees
    
    Returns:
        Processed PIL Image (RGBA)
    """
    # Scale card to target width, maintaining aspect ratio
    orig_w, orig_h = card_img.size
    aspect_ratio = orig_h / orig_w
    new_w = target_width
    new_h = int(target_width * aspect_ratio)
    card_resized = card_img.resize((new_w, new_h), Image.LANCZOS)
    
    # Apply random rotation
    card_rotated = card_resized.rotate(
        -rotation_angle,  # Negative for clockwise
        expand=True,
        resample=Image.BICUBIC,
        fillcolor=(0, 0, 0, 0)
    )
    
    # Apply random brightness adjustment (instead of background sampling)
    brightness_factor = random.uniform(0.7, 1.3)  # 70% to 130% brightness
    enhancer = ImageEnhance.Brightness(card_rotated)
    card_brightened = enhancer.enhance(brightness_factor)
    
    # Apply slight blur for realism
    blur_radius = random.uniform(0.3, 1.0)
    card_blurred = card_brightened.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    return card_blurred

# ============================================================================
# YOLO LABEL CONVERSION
# ============================================================================

def box_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """
    Convert pixel box coordinates to YOLO format (normalized center + size).
    
    Returns:
        (cx, cy, w, h) in normalized coordinates [0, 1]
    """
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return cx, cy, w, h

# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def generate_random_playmat_image(
    card_files,
    weights_dict,
    name_to_id,
    num_cards_range=(10, 20),
    output_mode='save'
):
    """
    Generate one random playmat image with cards.
    
    Args:
        card_files: List of (dir, filename) tuples
        weights_dict: Popularity weights dictionary
        name_to_id: Class name to ID mapping
        num_cards_range: (min, max) number of cards per scene
        output_mode: 'save' or 'visualize'
    
    Returns:
        (image, labels) tuple where labels is list of YOLO format strings
    """
    # Generate background
    img = generate_random_background(IMG_WIDTH, IMG_HEIGHT)
    
    # Determine number of cards for this scene
    num_cards = random.randint(num_cards_range[0], num_cards_range[1])
    
    # Track placed boxes and labels
    placed_boxes = []  # List of (x1, y1, x2, y2) tuples
    labels = []  # List of YOLO format strings
    
    for i in range(num_cards):
        # Select card using weighted selection
        if weights_dict:
            set_dir, card_filename = weighted_card_choice(card_files, weights_dict)
        else:
            set_dir, card_filename = random.choice(card_files)
        
        card_path = os.path.join(set_dir, card_filename)
        raw_name = os.path.splitext(card_filename)[0]
        class_name = canonicalize_name(raw_name)
        
        # Get or create class ID
        if class_name not in name_to_id:
            name_to_id[class_name] = len(name_to_id)
        class_id = name_to_id[class_name]
        
        # Load card image
        with Image.open(card_path).convert('RGBA') as card_img:
            # Determine target width: 50% use baseline, 50% use random (±50%)
            if random.random() < 0.5:
                # Baseline width
                target_width = CARD_BASE_WIDTH
            else:
                # Random width (50% to 150% of baseline)
                scale_multiplier = random.uniform(0.5, 1.5)
                target_width = int(CARD_BASE_WIDTH * scale_multiplier)
            
            # Random rotation
            rotation_angle = random.uniform(0, 360)
            
            # Process card (scale to target width, rotate, adjust brightness)
            # Height is calculated automatically to maintain aspect ratio
            card_processed = process_card(card_img, target_width, rotation_angle)
            
            # Get rotated card dimensions
            rotated_w, rotated_h = card_processed.size
            
            # Find valid position with overlap checking
            box = find_valid_position(
                placed_boxes,
                rotated_w,
                rotated_h,
                IMG_WIDTH,
                IMG_HEIGHT,
                MAX_OVERLAP_RATIO,
                max_attempts=100
            )
            
            if box is None:
                # Couldn't find valid position, skip this card
                print(f"[warning] Could not place card {i+1}/{num_cards}, skipping")
                continue
            
            x1, y1, x2, y2 = box
            placed_boxes.append(box)
            
            # Paste card onto background
            img.paste(card_processed, (x1, y1), card_processed)
            
            # Create YOLO label
            cx, cy, w, h = box_to_yolo(x1, y1, x2, y2, IMG_WIDTH, IMG_HEIGHT)
            label = f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
            labels.append(label)
    
    return img, labels

# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_labels(img, labels):
    """
    Draw bounding boxes on image for visualization.
    
    Args:
        img: PIL Image
        labels: List of YOLO format label strings
    
    Returns:
        PIL Image with boxes drawn
    """
    img_vis = img.copy()
    draw = ImageDraw.Draw(img_vis)
    
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255)
    ]
    
    for idx, label in enumerate(labels):
        parts = label.strip().split()
        if len(parts) != 5:
            continue
        
        class_id, cx, cy, w, h = parts
        cx, cy, w, h = float(cx), float(cy), float(w), float(h)
        
        # Convert to pixel coordinates
        x1 = int((cx - w/2) * IMG_WIDTH)
        y1 = int((cy - h/2) * IMG_HEIGHT)
        x2 = int((cx + w/2) * IMG_WIDTH)
        y2 = int((cy + h/2) * IMG_HEIGHT)
        
        # Draw box
        color = colors[idx % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label
        draw.text((x1 + 5, y1 + 5), f"C{class_id}", fill=color)
    
    return img_vis

# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate random playmat images for YOLO training')
    parser.add_argument('--num-images', type=int, default=100, help='Number of images to generate')
    parser.add_argument('--popularity-min', type=int, default=None, help='Minimum popularity rank (inclusive)')
    parser.add_argument('--popularity-max', type=int, default=None, help='Maximum popularity rank (inclusive)')
    parser.add_argument('--use-popularity-weights', action='store_true', help='Use popularity weights for card selection')
    parser.add_argument('--cards-per-scene-min', type=int, default=CARDS_PER_SCENE_MIN, help='Minimum cards per scene')
    parser.add_argument('--cards-per-scene-max', type=int, default=CARDS_PER_SCENE_MAX, help='Maximum cards per scene')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization images with bounding boxes')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # Load card files
    print(f"[init] Loading card files from {len(CARD_SET_DIRS)} directories...")
    card_files = load_card_files(CARD_SET_DIRS)
    print(f"[init] Found {len(card_files)} card images")
    
    # Load popularity weights
    weights_dict = {}
    if args.use_popularity_weights:
        print(f"[init] Loading popularity weights from {POPULARITY_WEIGHTS_PATH}")
        weights_dict = load_popularity_weights(POPULARITY_WEIGHTS_PATH)
        print(f"[init] Loaded {len(weights_dict)} card popularity weights")
    
    # Filter by popularity range
    if args.popularity_min is not None or args.popularity_max is not None:
        original_count = len(card_files)
        card_files = filter_cards_by_popularity(
            card_files,
            weights_dict,
            args.popularity_min,
            args.popularity_max
        )
        print(f"[init] Filtered to {len(card_files)} cards in rank range [{args.popularity_min}, {args.popularity_max}]")
        
        if len(card_files) == 0:
            print("[error] No cards found in specified popularity range!")
            return
    
    # Load or create class index
    class_index_path = os.path.join(OUTPUT_BASE_DIR, 'classes.yaml')
    name_to_id = load_class_index(class_index_path)
    print(f"[init] Loaded {len(name_to_id)} existing classes")
    
    # Create output directories
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(OUTPUT_BASE_DIR, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_BASE_DIR, split, 'labels'), exist_ok=True)
    
    # Generate images
    print(f"[generation] Generating {args.num_images} images...")
    print(f"[generation] Cards per scene: {args.cards_per_scene_min}-{args.cards_per_scene_max}")
    print(f"[generation] Max overlap: {MAX_OVERLAP_RATIO*100:.0f}%")
    
    counts = {'train': 0, 'valid': 0, 'test': 0}
    
    for idx in range(args.num_images):
        # Determine split
        rand = random.random()
        if rand < TRAIN_RATIO:
            split = 'train'
        elif rand < TRAIN_RATIO + VALID_RATIO:
            split = 'valid'
        else:
            split = 'test'
        
        # Generate image
        img, labels = generate_random_playmat_image(
            card_files,
            weights_dict if args.use_popularity_weights else {},
            name_to_id,
            num_cards_range=(args.cards_per_scene_min, args.cards_per_scene_max)
        )
        
        # Generate unique filename
        unique_id = uuid.uuid4().hex[:8]
        base_name = f"random_{idx:06d}_{unique_id}"
        
        # Save image
        img_path = os.path.join(OUTPUT_BASE_DIR, split, 'images', f"{base_name}.png")
        img.save(img_path)
        
        # Save labels
        label_path = os.path.join(OUTPUT_BASE_DIR, split, 'labels', f"{base_name}.txt")
        with open(label_path, 'w') as f:
            f.writelines(labels)
        
        # Save visualization if requested
        if args.visualize:
            img_vis = visualize_labels(img, labels)
            vis_path = os.path.join(OUTPUT_BASE_DIR, split, 'images', f"{base_name}_labeled.png")
            img_vis.save(vis_path)
        
        counts[split] += 1
        
        # Progress update
        if (idx + 1) % 10 == 0 or idx == args.num_images - 1:
            print(f"[progress] {idx+1}/{args.num_images} | train={counts['train']} valid={counts['valid']} test={counts['test']}")
    
    # Save updated class index
    save_class_index(class_index_path, name_to_id)
    print(f"[complete] Saved {len(name_to_id)} classes to {class_index_path}")
    
    # Create data.yaml
    data_yaml_path = os.path.join(OUTPUT_BASE_DIR, 'data.yaml')
    data_yaml = {
        'path': OUTPUT_BASE_DIR.replace('\\', '/'),
        'train': os.path.join(OUTPUT_BASE_DIR, 'train', 'images').replace('\\', '/'),
        'val': os.path.join(OUTPUT_BASE_DIR, 'valid', 'images').replace('\\', '/'),
        'test': os.path.join(OUTPUT_BASE_DIR, 'test', 'images').replace('\\', '/'),
        'nc': len(name_to_id),
        'names': [name for name, _ in sorted(name_to_id.items(), key=lambda x: x[1])]
    }
    
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)
    
    print(f"[complete] Saved data.yaml to {data_yaml_path}")
    print(f"[complete] Total images: {sum(counts.values())}")
    print(f"[complete] Train: {counts['train']}, Valid: {counts['valid']}, Test: {counts['test']}")

if __name__ == '__main__':
    main()
