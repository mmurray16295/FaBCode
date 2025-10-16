"""
Test script to generate one synthetic playmat image using CardSelector.
"""

import random
import math
import json
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from card_selector import CardSelector
from augmentations import apply_blur, apply_glare
from augmentation_config import get_config

# Load card name to class ID mapping for YOLO labels
CARD_NAME_TO_CLASS_ID = {}
try:
    with open('data/card_name_to_class_id.json', 'r', encoding='utf-8') as f:
        CARD_NAME_TO_CLASS_ID = json.load(f)
    print(f"Loaded {len(CARD_NAME_TO_CLASS_ID)} card name mappings for labeling")
except FileNotFoundError:
    print("WARNING: card_name_to_class_id.json not found. Labels will use class ID 0.")

def load_random_background(background_dir):
    """Load a random background image from the directory."""
    background_files = list(Path(background_dir).glob('*.jpg')) + list(Path(background_dir).glob('*.png'))
    if not background_files:
        raise FileNotFoundError(f"No background images found in {background_dir}")
    
    bg_path = random.choice(background_files)
    print(f"   Using background: {bg_path.name}")
    return Image.open(bg_path), bg_path

def load_label_file(background_path, labels_dir):
    """Load the corresponding label file for a background image."""
    label_filename = background_path.stem + '.txt'
    label_path = Path(labels_dir) / label_filename
    
    if not label_path.exists():
        print(f"   WARNING: Label file not found: {label_path}")
        return []
    
    # Zone class ID mapping from data.yaml
    zone_names = {
        0: 'Arms',
        1: 'Arms 2',
        2: 'Banish',
        3: 'Banish 2',
        4: 'Card',
        5: 'Chest',
        6: 'Chest 2',
        7: 'Combat Chain 1',
        8: 'Combat Chain 2',
        9: 'Graveyard',
        10: 'Graveyard 2',
        11: 'Head',
        12: 'Head 2',
        13: 'Hero',
        14: 'Hero 2',
        15: 'Legs',
        16: 'Legs 2',
        17: 'Pitch',
        18: 'Pitch 2',
        19: 'Ref',
        20: 'Weapon',
        21: 'Weapon 2',
        22: 'Weapon or Off-Hand',
        23: 'Weapon or Off-Hand 2',
        24: 'Window'
    }
    
    zones = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                center_x = float(parts[1])
                center_y = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                zones.append({
                    'class_id': class_id,
                    'zone_name': zone_names.get(class_id, f'Unknown_{class_id}'),
                    'center_x': center_x,
                    'center_y': center_y,
                    'width': width,
                    'height': height
                })
    
    print(f"   Loaded {len(zones)} zones from label file")
    return zones

def yolo_to_pixel_coords(zone, img_width, img_height, apply_jitter=True, jitter_range=25):
    """
    Convert YOLO normalized coordinates to pixel coordinates.
    
    Args:
        zone: Zone dict with normalized coordinates
        img_width: Image width in pixels
        img_height: Image height in pixels
        apply_jitter: Whether to apply position jitter (default True)
        jitter_range: Max jitter in pixels (default ±25px)
        
    Returns:
        Tuple of (x, y, width, height) in pixels
    """
    center_x_px = zone['center_x'] * img_width
    center_y_px = zone['center_y'] * img_height
    width_px = zone['width'] * img_width
    height_px = zone['height'] * img_height
    
    # Apply position jitter if enabled and not Window zone
    if apply_jitter and zone['zone_name'] != 'Window':
        # Use triangular distribution for natural center-bias
        # triangular(low, high, mode) where mode is the peak
        jitter_x = random.triangular(-jitter_range, jitter_range, 0)
        jitter_y = random.triangular(-jitter_range, jitter_range, 0)
        
        center_x_px += jitter_x
        center_y_px += jitter_y
    
    # Calculate top-left corner
    x = int(center_x_px - width_px / 2)
    y = int(center_y_px - height_px / 2)
    
    return x, y, int(width_px), int(height_px)

def partition_zones(zones):
    """
    Split zones into hero1/hero2 categories in a single pass.
    Replaces 4 separate list comprehensions with one efficient loop.
    
    Returns:
        Tuple of (hero1_zones, hero2_zones, hero1_available_zones, hero2_available_zones)
    """
    hero1_zones = []
    hero2_zones = []
    hero1_available = []
    hero2_available = []
    
    for z in zones:
        class_id = z['class_id']
        name = z['zone_name']
        ends_with_2 = name.endswith(' 2')
        
        # Hero 1 zones (for counting)
        if class_id not in [14, 24] and name not in ['Hero 2', 'Window'] and not ends_with_2:
            hero1_zones.append(z)
        
        # Hero 2 zones (for counting)
        if ends_with_2 or class_id == 14:
            hero2_zones.append(z)
        
        # Hero 1 available zones (for card selection)
        if class_id not in [13, 14, 24] and not ends_with_2:
            hero1_available.append(z)
        
        # Hero 2 available zones (for card selection)
        if ends_with_2 and class_id not in [13, 14, 24]:
            hero2_available.append(z)
    
    return hero1_zones, hero2_zones, hero1_available, hero2_available

def get_zone_sort_key(zone_name):
    """
    Get sort key for zone ordering. Weapon zones must come before Weapon or Off-Hand zones.
    Uses dict lookup for O(1) instead of conditional checks.
    
    Returns:
        Integer sort key (lower = higher priority)
    """
    # Pre-computed sort keys for known zones
    sort_keys = {
        'Weapon': 0,
        'Weapon 2': 0,
        'Weapon or Off-Hand': 1,
        'Weapon or Off-Hand 2': 1
    }
    return sort_keys.get(zone_name, 2)  # Default to 2 for all other zones

def save_yolo_labels(label_path, card_placements, card_name_to_class_id, img_width, img_height):
    """Save YOLO format label file with normalized coordinates.
    
    Args:
        label_path: Path to save the .txt label file
        card_placements: List of card placement dicts with x, y, width, height, card_name
        card_name_to_class_id: Dict mapping card names to class IDs
        img_width: Image width in pixels
        img_height: Image height in pixels
    """
    with open(label_path, 'w') as f:
        for placement in card_placements:
            card_name = placement.get('card_name', '10,000 Year Reunion')  # Default to first card
            class_id = card_name_to_class_id.get(card_name, 0)  # Default to class 0 if not found
            
            # Convert to center x, center y, width, height (normalized 0-1)
            x_center = (placement['x'] + placement['width'] / 2) / img_width
            y_center = (placement['y'] + placement['height'] / 2) / img_height
            norm_width = placement['width'] / img_width
            norm_height = placement['height'] / img_height
            
            # Clamp to [0, 1] range
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            norm_width = max(0, min(1, norm_width))
            norm_height = max(0, min(1, norm_height))
            
            # YOLO format: class_id x_center y_center width height
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")


def create_data_yaml(output_dir, force_update=False):
    """Check that data.yaml file exists for YOLO training.
    
    NOTE: data.yaml should be pre-generated with generate_card_data_yaml.py
    to include all 2,641 card name classes.
    
    Args:
        output_dir: Base directory (e.g., data/synthetic)
        force_update: Ignored (kept for backward compatibility)
    """
    yaml_path = Path(output_dir) / 'data.yaml'
    
    if yaml_path.exists():
        print(f"   data.yaml exists at {yaml_path}")
    else:
        print(f"   WARNING: data.yaml NOT FOUND at {yaml_path}")
        print(f"   Please run: python scripts/generate_card_data_yaml.py")


def draw_bounding_box_with_label(draw, x, y, width, height, label, color=(0, 255, 0)):
    """Draw a bounding box with label on the image."""
    # Draw rectangle
    draw.rectangle([x, y, x + width, y + height], outline=color, width=3)
    
    # Try to use a nice font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Draw label background
    label_bbox = draw.textbbox((x, y - 20), label, font=font)
    draw.rectangle(label_bbox, fill=color)
    
    # Draw label text
    draw.text((x, y - 20), label, fill=(0, 0, 0), font=font)

def calculate_overlap_percentage(box1, box2):
    """
    Calculate the percentage of overlap between two bounding boxes.
    
    Args:
        box1: Dict with 'x', 'y', 'width', 'height'
        box2: Dict with 'x', 'y', 'width', 'height'
    
    Returns:
        Float: Percentage of overlap (0-100) relative to smaller box area
    """
    # Calculate box boundaries
    x1_min, y1_min = box1['x'], box1['y']
    x1_max, y1_max = x1_min + box1['width'], y1_min + box1['height']
    
    x2_min, y2_min = box2['x'], box2['y']
    x2_max, y2_max = x2_min + box2['width'], y2_min + box2['height']
    
    # Calculate intersection
    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    
    if x_overlap == 0 or y_overlap == 0:
        return 0.0
    
    overlap_area = x_overlap * y_overlap
    
    # Calculate areas
    area1 = box1['width'] * box1['height']
    area2 = box2['width'] * box2['height']
    
    # Return percentage relative to smaller box
    smaller_area = min(area1, area2)
    if smaller_area == 0:
        return 0.0
    
    return (overlap_area / smaller_area) * 100

def find_valid_position_in_zone(zone, card_width, card_height, existing_placements, 
                                  img_width, img_height, max_overlap_pct=25, max_attempts=50):
    """
    Find a valid random position within a zone that doesn't exceed max overlap with existing cards.
    
    Args:
        zone: Zone dict with YOLO coordinates
        card_width: Width of card to place (after rotation)
        card_height: Height of card to place (after rotation)
        existing_placements: List of already placed card dicts
        img_width: Image width in pixels
        img_height: Image height in pixels
        max_overlap_pct: Maximum allowed overlap percentage (default 25%)
        max_attempts: Maximum placement attempts
    
    Returns:
        (x, y) tuple if valid position found, None otherwise
    """
    zone_x, zone_y, zone_w, zone_h = yolo_to_pixel_coords(zone, img_width, img_height)
    
    for attempt in range(max_attempts):
        # Random position within zone boundaries, accounting for card size
        x = random.randint(zone_x, max(zone_x, zone_x + zone_w - card_width))
        y = random.randint(zone_y, max(zone_y, zone_y + zone_h - card_height))
        
        # Check overlap with all existing placements
        new_box = {'x': x, 'y': y, 'width': card_width, 'height': card_height}
        
        valid = True
        for existing in existing_placements:
            overlap = calculate_overlap_percentage(new_box, existing)
            if overlap > max_overlap_pct:
                valid = False
                break
        
        if valid:
            return x, y
    
    return None

def calculate_glare_intensity(card_center_x, card_center_y, light_source_pos, img_width, img_height):
    """
    Calculate glare intensity based on distance from light source.
    Cards closer to the light source get more glare.
    
    Args:
        card_center_x, card_center_y: Center of the card
        light_source_pos: (x, y) tuple of light source position
        img_width, img_height: Image dimensions for normalization
        
    Returns:
        Glare intensity (0.0-1.0), or None if no light source
    """
    if light_source_pos is None:
        return None
    
    # Calculate distance from card to light source
    dx = card_center_x - light_source_pos[0]
    dy = card_center_y - light_source_pos[1]
    distance = math.sqrt(dx**2 + dy**2)
    
    # Normalize by image diagonal (max possible distance)
    max_distance = math.sqrt(img_width**2 + img_height**2)
    normalized_distance = distance / max_distance
    
    # Inverse relationship: closer = more glare
    # Use steeper exponential falloff for more dramatic gradient
    # Increased from -3 to -5 to create faster dropoff from light source
    raw_intensity = math.exp(-5 * normalized_distance)
    
    # Fixed minimum (0.05) with moderate max (0.35)
    # Steeper falloff means only cards very close to light source get high glare
    glare_intensity = max(0.05, min(0.35, raw_intensity))
    
    return glare_intensity


def apply_occluders_to_playmat(playmat, card_placements, tokens_dir, probability_per_card=0.10, second_occluder_probability=0.01):
    """
    Apply token/dice occluders to cards on the playmat.
    
    Args:
        playmat: PIL Image of the playmat with all cards placed
        card_placements: List of card placement dicts with 'x', 'y', 'width', 'height'
        tokens_dir: Directory containing token images
        probability_per_card: Probability of applying occluder to each card (default 10%)
        second_occluder_probability: Probability of applying second occluder to a card (default 1%)
        
    Returns:
        PIL Image with occluders applied
    """
    # Load all available tokens
    tokens_path = Path(tokens_dir)
    token_files = list(tokens_path.glob('*.png'))
    
    if len(token_files) == 0:
        print("   No token files found for occluders!")
        return playmat
    
    # Create a copy to work with
    result = playmat.copy()
    occluders_applied = 0
    
    # Process each card placement
    for placement in card_placements:
        # Skip based on probability
        if random.random() > probability_per_card:
            continue
        
        card_x = placement['x']
        card_y = placement['y']
        card_width = placement['width']
        card_height = placement['height']
        
        # Determine number of occluders for this card (1 or 2)
        num_occluders = 2 if random.random() < second_occluder_probability else 1
        
        for i in range(num_occluders):
            # Select random token
            token_path = random.choice(token_files)
            token = Image.open(token_path)
            
            # Apply augmentations to token (blur and color shift)
            # Convert to numpy array for processing
            token_array = np.array(token)
            
            # Apply slight blur (10-30% intensity)
            blur_intensity = random.uniform(0.10, 0.30)
            kernel_size = max(3, int(token_array.shape[0] * blur_intensity * 0.1))
            if kernel_size % 2 == 0:
                kernel_size += 1
            token_array = cv2.GaussianBlur(token_array, (kernel_size, kernel_size), 0)
            
            # Apply color shift (slight hue/saturation adjustment)
            if token_array.shape[2] == 4:  # RGBA
                # Work with RGB channels only, preserve alpha
                rgb = token_array[:, :, :3]
                alpha = token_array[:, :, 3]
                
                # Convert to HSV for color adjustment
                hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
                
                # Random hue shift (-10 to +10 degrees)
                hue_shift = random.uniform(-10, 10)
                hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
                
                # Random saturation adjustment (0.8 to 1.2)
                sat_adjust = random.uniform(0.8, 1.2)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_adjust, 0, 255)
                
                # Random brightness adjustment (0.9 to 1.1)
                bright_adjust = random.uniform(0.9, 1.1)
                hsv[:, :, 2] = np.clip(hsv[:, :, 2] * bright_adjust, 0, 255)
                
                # Convert back to RGB
                rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
                
                # Recombine with alpha
                token_array = np.dstack([rgb, alpha])
            
            # Convert back to PIL Image
            token = Image.fromarray(token_array)
            
            # Random rotation (0-360 degrees)
            rotation_angle = random.uniform(0, 360)
            token_rotated = token.rotate(rotation_angle, expand=True, resample=Image.BICUBIC)
            
            token_width, token_height = token_rotated.size
            
            # Position occluder more centered on the card (weighted toward center)
            # Use beta distribution for center bias: most positions near center, fewer at edges
            center_bias = 0.3  # Lower = more centered, higher = more spread out
            
            # Generate position relative to card with center bias
            rel_x = random.betavariate(2, 2) - 0.5  # Range: -0.5 to 0.5, centered at 0
            rel_y = random.betavariate(2, 2) - 0.5
            
            # Scale relative position to card size (allow some overhang)
            max_offset_x = card_width * 0.3  # Can be up to 30% off-center
            max_offset_y = card_height * 0.3
            
            # Calculate absolute position (centered on card, with offset)
            x = card_x + (card_width - token_width) // 2 + int(rel_x * max_offset_x)
            y = card_y + (card_height - token_height) // 2 + int(rel_y * max_offset_y)
            
            # Paste token onto playmat using alpha channel as mask
            if token_rotated.mode == 'RGBA':
                result.paste(token_rotated, (x, y), token_rotated)
            else:
                result.paste(token_rotated, (x, y))
            
            occluders_applied += 1
    
    if occluders_applied > 0:
        print(f"   Applied {occluders_applied} occluder(s) to playmat")
    
    return result


def apply_hard_case(card_img):
    """
    Apply a hard case effect to a card image.
    Creates a clearish white halo with extra glare and black edge artifacts to simulate
    a clear plastic hard case (thicker than a sleeve).
    
    Args:
        card_img: PIL Image of the card
        
    Returns:
        PIL Image with hard case effect applied (RGBA with transparent background)
    """
    card_width, card_height = card_img.size
    
    # Hard case is larger than card - 8-12px on each side (thicker and more visible)
    border_size = random.randint(8, 12)
    case_width = card_width + (border_size * 2)
    case_height = card_height + (border_size * 2)
    
    # Create RGBA image with TRANSPARENT background to avoid white corners when rotating
    case_img = Image.new('RGBA', (case_width, case_height), (0, 0, 0, 0))
    
    # Paste the card centered in the case
    if card_img.mode == 'RGBA':
        case_img.paste(card_img, (border_size, border_size), card_img)
    else:
        case_img.paste(card_img.convert('RGBA'), (border_size, border_size))
    
    # Convert to numpy for adding effects (preserve alpha) (RGBA: RGB + Alpha)
    case_array = np.array(case_img)
    
    # Add clearish white halo effect ONLY on the border area (simulating thick clear plastic)
    # Use varied grey/white tones to make it more visible
    border_base_color = random.randint(200, 230)  # Light grey to white range
    
    # Fill border areas with semi-opaque white/grey (making hard case visible)
    for offset in range(border_size):
        # Fade alpha from opaque at edge to semi-transparent near card
        alpha = int(255 * (1.0 - (offset / border_size) * 0.5))  # 100% to 50% opacity
        color_value = border_base_color + random.randint(-10, 10)  # Slight variation
        
        # Top border
        case_array[offset, :, :3] = color_value  # RGB channels
        case_array[offset, :, 3] = alpha  # Alpha channel
        
        # Bottom border
        case_array[case_height - 1 - offset, :, :3] = color_value
        case_array[case_height - 1 - offset, :, 3] = alpha
        
        # Left border (avoid corners already set)
        case_array[border_size:case_height-border_size, offset, :3] = color_value
        case_array[border_size:case_height-border_size, offset, 3] = alpha
        
        # Right border (avoid corners already set)
        case_array[border_size:case_height-border_size, case_width - 1 - offset, :3] = color_value
        case_array[border_size:case_height-border_size, case_width - 1 - offset, 3] = alpha
    
    # Add extra glare spots (simulating light reflection on plastic) - BIGGER and BRIGHTER
    num_glare_spots = random.randint(3, 6)
    for _ in range(num_glare_spots):
        # Random position, prefer edges and corners where hard case is visible
        if random.random() < 0.7:  # 70% chance of edge placement
            # Place on the border area
            edge = random.choice(['top', 'bottom', 'left', 'right'])
            if edge == 'top':
                gx = random.randint(0, case_width - 1)
                gy = random.randint(0, border_size)
            elif edge == 'bottom':
                gx = random.randint(0, case_width - 1)
                gy = random.randint(case_height - border_size, case_height - 1)
            elif edge == 'left':
                gx = random.randint(0, border_size)
                gy = random.randint(0, case_height - 1)
            else:  # right
                gx = random.randint(case_width - border_size, case_width - 1)
                gy = random.randint(0, case_height - 1)
        else:
            # Random position anywhere on card
            gx = random.randint(border_size, case_width - border_size)
            gy = random.randint(border_size, case_height - border_size)
        
        # Create circular glare spot (LARGER radius for visibility)
        glare_radius = random.randint(20, 40)
        for dy in range(-glare_radius, glare_radius + 1):
            for dx in range(-glare_radius, glare_radius + 1):
                if dx*dx + dy*dy <= glare_radius*glare_radius:
                    px, py = gx + dx, gy + dy
                    if 0 <= px < case_width and 0 <= py < case_height:
                        # Only apply to opaque pixels (where alpha > 0)
                        if case_array[py, px, 3] > 0:
                            distance_ratio = np.sqrt(dx*dx + dy*dy) / glare_radius
                            glare_boost = int(80 * (1 - distance_ratio))  # MUCH stronger glare
                            case_array[py, px, :3] = np.clip(case_array[py, px, :3] + glare_boost, 0, 255)
    
    # Add black lines/artifacts parallel to edges (simulating case seams/edges)
    # RGBA format: set RGB to dark and alpha to fully opaque
    line_color = (random.randint(0, 40), random.randint(0, 40), random.randint(0, 40), 255)  # Dark with full opacity
    
    # Top edge - horizontal lines (2-3 lines for visibility)
    num_top_lines = random.randint(2, 3)
    for i in range(num_top_lines):
        line_y = random.randint(1, border_size - 1)
        line_thickness = random.randint(1, 3)  # Thicker lines
        for t in range(line_thickness):
            if line_y + t < case_height:
                case_array[line_y + t, :] = line_color
    
    # Bottom edge - horizontal lines
    num_bottom_lines = random.randint(2, 3)
    for i in range(num_bottom_lines):
        line_y = case_height - random.randint(2, border_size)
        line_thickness = random.randint(1, 3)
        for t in range(line_thickness):
            if line_y + t < case_height:
                case_array[line_y + t, :] = line_color
    
    # Left edge - vertical lines
    num_left_lines = random.randint(2, 3)
    for i in range(num_left_lines):
        line_x = random.randint(1, border_size - 1)
        line_thickness = random.randint(1, 3)
        for t in range(line_thickness):
            if line_x + t < case_width:
                case_array[:, line_x + t] = line_color
    
    # Right edge - vertical lines
    num_right_lines = random.randint(2, 3)
    for i in range(num_right_lines):
        line_x = case_width - random.randint(2, border_size)
        line_thickness = random.randint(1, 3)
        for t in range(line_thickness):
            if line_x + t < case_width:
                case_array[:, line_x + t] = line_color
    
    # Apply Gaussian blur to smooth and soften the black edge lines
    # Blur only the RGB channels, preserve alpha
    case_bgr = case_array[:, :, :3][:, :, ::-1]  # Convert RGBA to BGR for OpenCV
    case_bgr_blurred = cv2.GaussianBlur(case_bgr, (5, 5), 1.5)  # Moderate blur with sigma=1.5
    case_array[:, :, :3] = case_bgr_blurred[:, :, ::-1]  # Convert back to RGB
    
    # Add some random small artifacts (scratches, dust) on the case surface
    num_artifacts = random.randint(5, 12)
    for _ in range(num_artifacts):
        ax = random.randint(0, case_width - 1)
        ay = random.randint(0, case_height - 1)
        artifact_len = random.randint(3, 10)
        artifact_color = (random.randint(160, 200), random.randint(160, 200), random.randint(160, 200), 200)  # RGBA
        
        # Draw small line artifact
        direction = random.choice(['h', 'v'])
        for i in range(artifact_len):
            if direction == 'h' and ax + i < case_width and case_array[ay, ax + i, 3] > 0:
                case_array[ay, ax + i] = artifact_color
            elif direction == 'v' and ay + i < case_height and case_array[ay + i, ax, 3] > 0:
                case_array[ay + i, ax] = artifact_color
    
    return Image.fromarray(case_array, 'RGBA')


def apply_card_augmentations(card_img, augmentation_config, blur_intensity=None, glare_intensity=None, glare_pattern=None, color_params=None, sleeve_color=None, card_types=None, use_hard_case=False):
    """
    Apply all augmentations to a card image.
    Converts PIL → NumPy → apply augmentations → PIL
    
    Args:
        card_img: PIL Image
        augmentation_config: AugmentationConfig instance
        blur_intensity: Fixed blur intensity for this image (0.0-1.0)
        glare_intensity: Fixed glare intensity for this card (0.0-1.0)
        glare_pattern: List of dicts with glare spot positions (x_ratio, y_ratio, radius_ratio)
        color_params: Dict with color adjustment parameters (brightness, contrast, saturation, hue_shift)
        sleeve_color: Tuple (R, G, B) for sleeve color. If None, no sleeve is applied.
        card_types: List of card types to determine if color strip degradation should be applied
        use_hard_case: Boolean, if True applies hard case effect instead of sleeve
        
    Returns:
        Augmented PIL Image (with sleeve or hard case if applied)
    """
    # Preserve alpha channel if present
    has_alpha = card_img.mode == 'RGBA'
    
    # Convert PIL to NumPy array (BGR format for OpenCV) for initial processing
    if has_alpha:
        card_array = np.array(card_img)
        alpha_channel = card_array[:, :, 3].copy()  # Save alpha
        card_array = card_array[:, :, :3]  # RGB only for now
    else:
        card_array = np.array(card_img.convert('RGB'))
    card_array = card_array[:, :, ::-1]  # RGB → BGR for OpenCV
    
    # Degrade color strip at top to make pitch value less obvious (85% probability)
    # Apply THIS FIRST, before sleeve, so only the card's color strip is degraded
    # Only apply to action/attack/defense/instant cards (NOT equipment, weapons, or heroes)
    should_degrade_color = False
    if card_types is not None:
        # Convert types to lowercase for case-insensitive comparison
        types_lower = [t.lower() for t in card_types]
        # Exclude Equipment, Weapon, and Hero cards
        excluded_types = {'equipment', 'weapon', 'hero'}
        # Check if card has any excluded types
        has_excluded_type = any(t in excluded_types for t in types_lower)
        # Only degrade if it doesn't have excluded types
        should_degrade_color = not has_excluded_type
    
    if should_degrade_color:
        from augmentations import apply_color_strip_degradation
        card_array = apply_color_strip_degradation(card_array, strip_height_ratio=0.08, probability=0.85)
    
    # Convert back to PIL for sleeve/hard case application
    card_array_rgb = card_array[:, :, ::-1]  # BGR → RGB
    if has_alpha:
        # Restore alpha channel
        card_array_rgba = np.dstack([card_array_rgb, alpha_channel])
        card_img = Image.fromarray(card_array_rgba, 'RGBA')
    else:
        card_img = Image.fromarray(card_array_rgb)
    
    # Apply hard case OR sleeve AFTER color strip degradation (mutually exclusive)
    if use_hard_case:
        # Apply hard case effect (for Equipment, Weapon, Hero cards)
        card_img = apply_hard_case(card_img)
        has_alpha = True  # Hard case NOW creates RGBA image with transparent background
    elif sleeve_color is not None:
        # Create a sleeve background that's 3px larger on each side
        card_width, card_height = card_img.size
        sleeve_width = card_width + 6  # 3px on each side
        sleeve_height = card_height + 6  # 3px on each side
        
        # Create the sleeve as a solid color rectangle
        sleeve_img = Image.new('RGB', (sleeve_width, sleeve_height), sleeve_color)
        
        # Paste the card onto the sleeve (centered, so 3px border on all sides)
        if card_img.mode == 'RGBA':
            # Use alpha channel as mask for proper transparency
            sleeve_img.paste(card_img, (3, 3), card_img)
        else:
            sleeve_img.paste(card_img, (3, 3))
        
        # Use the sleeved card for all subsequent transformations
        card_img = sleeve_img
        has_alpha = False  # Sleeve creates RGB image
    
    # Convert PIL to NumPy array (BGR format for OpenCV) for remaining augmentations
    if has_alpha:
        card_array = np.array(card_img)
        alpha_channel = card_array[:, :, 3].copy()  # Save alpha
        card_array = card_array[:, :, :3]  # RGB only
    else:
        card_array = np.array(card_img.convert('RGB'))
    card_array = card_array[:, :, ::-1]  # RGB → BGR for OpenCV
    
    # Apply uniform greyish yellow mask (10% transparent) to simulate sleeve tint/aging
    # Greyish yellow in BGR: (120, 180, 180) - more grey, less yellow tone
    greyish_yellow = np.array([120, 180, 180], dtype=np.float32)
    mask_intensity = 0.10  # 10% opacity
    card_array = card_array.astype(np.float32)
    card_array = card_array * (1.0 - mask_intensity) + greyish_yellow * mask_intensity
    card_array = np.clip(card_array, 0, 255).astype(np.uint8)
    
    # Apply all augmentations (blur, glare, color adjustments)
    card_array = apply_blur(card_array, augmentation_config.blur, blur_intensity)
    card_array = apply_glare(card_array, augmentation_config.glare, glare_intensity, glare_pattern)
    if color_params is not None:
        from augmentations import apply_color_adjustment
        card_array = apply_color_adjustment(card_array, augmentation_config.color, **color_params)
    
    # Convert back to PIL (BGR → RGB)
    card_array = card_array[:, :, ::-1]  # BGR → RGB
    if has_alpha:
        # Restore alpha channel
        card_array_rgba = np.dstack([card_array, alpha_channel])
        card_img_augmented = Image.fromarray(card_array_rgba, 'RGBA')
    else:
        card_img_augmented = Image.fromarray(card_array)
    
    return card_img_augmented

def apply_window_color_transformation(playmat: Image.Image, window_zone: dict) -> Image.Image:
    """
    Apply massive random color transformations to the window area of the background.
    
    This creates variety in lighting conditions, weather, time of day, etc.
    
    Args:
        playmat: PIL Image of the playmat background
        window_zone: Dictionary with window zone info (center_x, center_y, width, height in normalized coords)
        
    Returns:
        PIL Image with transformed window area
    """
    # Convert to NumPy for processing
    playmat_array = np.array(playmat.convert('RGB'))
    playmat_array = playmat_array[:, :, ::-1]  # RGB → BGR for OpenCV
    
    img_height, img_width = playmat_array.shape[:2]
    
    # Calculate window bounding box in pixels
    center_x = int(window_zone['center_x'] * img_width)
    center_y = int(window_zone['center_y'] * img_height)
    w = int(window_zone['width'] * img_width)
    h = int(window_zone['height'] * img_height)
    
    # Get bbox coordinates
    x_min = max(0, center_x - w // 2)
    x_max = min(img_width, center_x + w // 2)
    y_min = max(0, center_y - h // 2)
    y_max = min(img_height, center_y + h // 2)
    
    # Extract window region
    window_region = playmat_array[y_min:y_max, x_min:x_max].copy()
    
    # Convert to HSV for easier color manipulation
    window_hsv = cv2.cvtColor(window_region, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # Apply MASSIVE random transformations
    transformation_type = random.choice([
        'hue_shift',      # Different time of day / colored lighting
        'saturation',     # Overcast vs vibrant
        'brightness',     # Lighting intensity
        'color_tint',     # Strong color overlay (sunset, blue hour, etc.)
        'extreme_combo',  # Combination of multiple effects
        'invert_colors',  # Artistic/unusual lighting
        'posterize',      # Simplified colors
    ])
    
    if transformation_type == 'hue_shift':
        # Massive hue shift (simulate different colored lighting)
        hue_shift = random.uniform(-90, 90)  # Can shift entire color spectrum
        window_hsv[:, :, 0] = (window_hsv[:, :, 0] + hue_shift) % 180
        print(f"   Window transformation: Hue shift {hue_shift:.1f}°")
    
    elif transformation_type == 'saturation':
        # Extreme saturation changes (grey overcast to super vibrant)
        sat_mult = random.choice([
            random.uniform(0.1, 0.4),   # Very desaturated (foggy/overcast)
            random.uniform(1.5, 3.0)    # Hyper saturated (vibrant/artificial)
        ])
        window_hsv[:, :, 1] = np.clip(window_hsv[:, :, 1] * sat_mult, 0, 255)
        print(f"   Window transformation: Saturation {sat_mult:.2f}×")
    
    elif transformation_type == 'brightness':
        # Extreme brightness (dark storm to bright sunny day)
        bright_mult = random.choice([
            random.uniform(0.3, 0.6),   # Very dark
            random.uniform(1.4, 2.2)    # Very bright
        ])
        window_hsv[:, :, 2] = np.clip(window_hsv[:, :, 2] * bright_mult, 0, 255)
        print(f"   Window transformation: Brightness {bright_mult:.2f}×")
    
    elif transformation_type == 'color_tint':
        # Strong color overlay (golden hour, blue hour, green/red/purple lighting)
        tint_colors = [
            (15, 200),   # Orange/golden
            (105, 200),  # Blue
            (60, 200),   # Green
            (0, 200),    # Red
            (140, 200),  # Purple
            (30, 200),   # Yellow
        ]
        tint_hue, tint_sat = random.choice(tint_colors)
        overlay_strength = random.uniform(0.3, 0.7)
        
        # Blend towards the tint color
        window_hsv[:, :, 0] = window_hsv[:, :, 0] * (1 - overlay_strength) + tint_hue * overlay_strength
        window_hsv[:, :, 1] = np.clip(window_hsv[:, :, 1] * (1 - overlay_strength) + tint_sat * overlay_strength, 0, 255)
        
        tint_names = {15: 'Golden', 105: 'Blue', 60: 'Green', 0: 'Red', 140: 'Purple', 30: 'Yellow'}
        print(f"   Window transformation: {tint_names.get(tint_hue, 'Color')} tint {overlay_strength:.0%}")
    
    elif transformation_type == 'extreme_combo':
        # Combine multiple effects for really varied lighting
        hue_shift = random.uniform(-60, 60)
        sat_mult = random.uniform(0.4, 2.0)
        bright_mult = random.uniform(0.5, 1.8)
        
        window_hsv[:, :, 0] = (window_hsv[:, :, 0] + hue_shift) % 180
        window_hsv[:, :, 1] = np.clip(window_hsv[:, :, 1] * sat_mult, 0, 255)
        window_hsv[:, :, 2] = np.clip(window_hsv[:, :, 2] * bright_mult, 0, 255)
        print(f"   Window transformation: Combo (H{hue_shift:.0f}° S{sat_mult:.1f}× B{bright_mult:.1f}×)")
    
    elif transformation_type == 'invert_colors':
        # Invert colors for dramatic effect
        window_hsv[:, :, 0] = (window_hsv[:, :, 0] + 90) % 180  # Shift hue by 90° (complementary colors)
        window_hsv[:, :, 2] = 255 - window_hsv[:, :, 2]  # Invert brightness
        print(f"   Window transformation: Inverted colors")
    
    elif transformation_type == 'posterize':
        # Reduce colors to create stylized look
        levels = random.choice([4, 6, 8])
        window_hsv[:, :, 0] = (window_hsv[:, :, 0] // (180 / levels)) * (180 / levels)
        window_hsv[:, :, 1] = (window_hsv[:, :, 1] // (255 / levels)) * (255 / levels)
        window_hsv[:, :, 2] = (window_hsv[:, :, 2] // (255 / levels)) * (255 / levels)
        print(f"   Window transformation: Posterized ({levels} levels)")
    
    # Convert back to BGR
    window_transformed = cv2.cvtColor(window_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # Create smooth gradient mask at edges for natural blending (OPTIMIZED: vectorized operations)
    blend_size = min(30, w // 10, h // 10)  # Blend over 30px or 10% of window size
    mask = np.ones((y_max - y_min, x_max - x_min), dtype=np.float32)
    
    # Apply gradient falloff at edges using vectorized NumPy operations (30-60x faster)
    if blend_size > 0 and mask.shape[0] > 0 and mask.shape[1] > 0:
        # Ensure blend_size doesn't exceed mask dimensions
        actual_blend_h = min(blend_size, mask.shape[0] // 2)
        actual_blend_w = min(blend_size, mask.shape[1] // 2)
        
        if actual_blend_h > 0:
            gradient_h = np.linspace(0, 1, actual_blend_h, dtype=np.float32)
            # Apply to top and bottom edges
            mask[:actual_blend_h, :] *= gradient_h[:, np.newaxis]
            mask[-actual_blend_h:, :] *= gradient_h[::-1, np.newaxis]
        
        if actual_blend_w > 0:
            gradient_w = np.linspace(0, 1, actual_blend_w, dtype=np.float32)
            # Apply to left and right edges (already has top/bottom gradient applied)
            mask[:, :actual_blend_w] *= gradient_w
            mask[:, -actual_blend_w:] *= gradient_w[::-1]
    
    # Blend transformed window back into playmat
    mask_3ch = np.stack([mask] * 3, axis=2)
    playmat_array[y_min:y_max, x_min:x_max] = (
        window_transformed * mask_3ch + 
        playmat_array[y_min:y_max, x_min:x_max] * (1 - mask_3ch)
    ).astype(np.uint8)
    
    # Convert back to PIL
    playmat_array_rgb = playmat_array[:, :, ::-1]  # BGR → RGB
    return Image.fromarray(playmat_array_rgb)

def replace_outside_window_area(playmat: Image.Image, window_zone: dict, probability: float = 0.75) -> Image.Image:
    """
    Replace the area outside the window bbox with a random color or texture.
    
    This creates variety in backgrounds and focuses attention on the playmat area.
    Applied 75% of the time to add background diversity.
    
    Args:
        playmat: PIL Image of the playmat background
        window_zone: Dictionary with window zone info (center_x, center_y, width, height in normalized coords)
        probability: Probability of applying this effect (default 0.75)
        
    Returns:
        PIL Image with replaced outside area
    """
    # Apply with specified probability
    if random.random() > probability:
        return playmat
    
    # Convert to NumPy for processing
    playmat_array = np.array(playmat.convert('RGB'))
    playmat_array = playmat_array[:, :, ::-1]  # RGB → BGR for OpenCV
    
    img_height, img_width = playmat_array.shape[:2]
    
    # Calculate window bounding box in pixels
    center_x = int(window_zone['center_x'] * img_width)
    center_y = int(window_zone['center_y'] * img_height)
    w = int(window_zone['width'] * img_width)
    h = int(window_zone['height'] * img_height)
    
    # Get bbox coordinates
    x_min = max(0, center_x - w // 2)
    x_max = min(img_width, center_x + w // 2)
    y_min = max(0, center_y - h // 2)
    y_max = min(img_height, center_y + h // 2)
    
    # Create mask for window area (1 = keep, 0 = replace)
    window_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    window_mask[y_min:y_max, x_min:x_max] = 1
    
    # Choose replacement type
    replacement_type = random.choice([
        'solid_color',      # Solid color background
        'gradient',         # Color gradient
        'noise',           # Textured noise
        'wood_texture',    # Wood-like texture
        'fabric_texture',  # Fabric-like texture
    ])
    
    if replacement_type == 'solid_color':
        # Random solid color (darker tones for realistic table surfaces)
        color_choice = random.choice([
            np.array([random.randint(20, 60), random.randint(20, 60), random.randint(20, 60)]),  # Dark neutral
            np.array([random.randint(15, 40), random.randint(40, 80), random.randint(10, 30)]),  # Dark green
            np.array([random.randint(10, 30), random.randint(10, 30), random.randint(40, 80)]),  # Dark red/brown
            np.array([random.randint(40, 80), random.randint(30, 60), random.randint(15, 40)]),  # Dark blue
        ])
        replacement = np.full_like(playmat_array, color_choice, dtype=np.uint8)
        print(f"   Outside window: Solid color BGR{tuple(color_choice)}")
    
    elif replacement_type == 'gradient':
        # Linear gradient across the image
        angle = random.uniform(0, 360)
        color1 = np.array([random.randint(20, 80), random.randint(20, 80), random.randint(20, 80)])
        color2 = np.array([random.randint(20, 80), random.randint(20, 80), random.randint(20, 80)])
        
        # Create gradient
        y_coords, x_coords = np.ogrid[:img_height, :img_width]
        angle_rad = np.radians(angle)
        gradient_axis = (x_coords * np.cos(angle_rad) + y_coords * np.sin(angle_rad))
        gradient_axis = (gradient_axis - gradient_axis.min()) / (gradient_axis.max() - gradient_axis.min())
        
        replacement = np.zeros_like(playmat_array)
        for c in range(3):
            replacement[:, :, c] = (color1[c] * (1 - gradient_axis) + color2[c] * gradient_axis).astype(np.uint8)
        print(f"   Outside window: Gradient at {angle:.0f}°")
    
    elif replacement_type == 'noise':
        # Perlin-like noise texture
        base_color = np.array([random.randint(30, 70), random.randint(30, 70), random.randint(30, 70)])
        noise = np.random.normal(0, 25, (img_height, img_width, 3))
        replacement = np.clip(base_color + noise, 0, 255).astype(np.uint8)
        print(f"   Outside window: Noise texture")
    
    elif replacement_type == 'wood_texture':
        # Wood-like grain texture
        base_brown = np.array([random.randint(15, 35), random.randint(35, 65), random.randint(60, 100)])
        
        # Create wood grain pattern with sine waves
        y_coords, x_coords = np.ogrid[:img_height, :img_width]
        grain_frequency = random.uniform(0.01, 0.03)
        grain_pattern = np.sin(x_coords * grain_frequency + np.random.normal(0, 0.5, (img_height, img_width)))
        grain_pattern = ((grain_pattern + 1) / 2 * 40 - 20)  # Range: -20 to +20
        
        replacement = np.zeros_like(playmat_array)
        for c in range(3):
            replacement[:, :, c] = np.clip(base_brown[c] + grain_pattern, 0, 255).astype(np.uint8)
        print(f"   Outside window: Wood texture")
    
    elif replacement_type == 'fabric_texture':
        # Fabric/felt-like texture
        base_color = np.array([random.randint(20, 60), random.randint(50, 90), random.randint(20, 60)])
        
        # Create fabric weave pattern
        fine_noise = np.random.normal(0, 15, (img_height, img_width, 3))
        replacement = np.clip(base_color + fine_noise, 0, 255).astype(np.uint8)
        
        # Add subtle directional pattern
        y_coords, x_coords = np.ogrid[:img_height, :img_width]
        weave = (np.sin(x_coords * 0.1) * 5 + np.sin(y_coords * 0.1) * 5)
        for c in range(3):
            replacement[:, :, c] = np.clip(replacement[:, :, c] + weave, 0, 255).astype(np.uint8)
        print(f"   Outside window: Fabric texture")
    
    # Blend replacement with original using mask
    # Expand mask to 3 channels
    mask_3ch = np.stack([window_mask] * 3, axis=2).astype(np.float32)
    
    # Blend: keep window area, replace outside
    result = (playmat_array * mask_3ch + replacement * (1 - mask_3ch)).astype(np.uint8)
    
    # Convert back to PIL
    result_rgb = result[:, :, ::-1]  # BGR → RGB
    return Image.fromarray(result_rgb)

def place_card_on_playmat(playmat, card_img, x, y, rotation=0, scale=1.0):
    """Place a card image on the playmat at given position with anti-aliased rotation. Returns final dimensions."""
    # Convert to RGBA if not already
    if card_img.mode != 'RGBA':
        card_img = card_img.convert('RGBA')
    
    # Resize card
    new_width = int(card_img.width * scale)
    new_height = int(card_img.height * scale)
    card_resized = card_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Rotate if needed with anti-aliasing using super-sampling
    if rotation != 0:
        # Super-sample for smooth edges: 4x larger, rotate, then downsample
        supersample_scale = 4
        large_width = card_resized.width * supersample_scale
        large_height = card_resized.height * supersample_scale
        
        # Upsample
        card_large = card_resized.resize((large_width, large_height), Image.Resampling.LANCZOS)
        
        # Rotate with high-quality bicubic
        card_large_rotated = card_large.rotate(rotation, expand=True, resample=Image.Resampling.BICUBIC)
        
        # Downsample with smooth filter for anti-aliasing
        final_width = card_large_rotated.width // supersample_scale
        final_height = card_large_rotated.height // supersample_scale
        card_resized = card_large_rotated.resize((final_width, final_height), Image.Resampling.LANCZOS)
    
    # Get final dimensions after rotation
    final_width = card_resized.width
    final_height = card_resized.height
    
    # Paste with alpha channel
    playmat.paste(card_resized, (x, y), card_resized)
    
    return final_width, final_height

def place_hero_card(hero_img_path, hero_zone, hero_card, image_cache, playmat, 
                    aug_config, blur_intensity, glare_pattern, color_params, 
                    sleeve_color, light_source_pos, img_width, img_height, uniform_scale_factor=1.0):
    """Helper function to place hero card with augmentations and rotation (OPTIMIZATION: eliminates code duplication)."""
    x, y, zone_w, zone_h = yolo_to_pixel_coords(hero_zone, img_width, img_height)
    hero_img = image_cache[str(hero_img_path)]  # OPTIMIZATION: use cached image
    
    if aug_config is not None:
        card_center_x = hero_zone['center_x'] * img_width
        card_center_y = hero_zone['center_y'] * img_height
        glare_intensity = calculate_glare_intensity(card_center_x, card_center_y, light_source_pos, img_width, img_height)
        # Heroes have 50% chance of getting hard case (per card decision)
        use_hard_case = random.random() < 0.5
        hero_img = apply_card_augmentations(hero_img, aug_config, blur_intensity, glare_intensity, glare_pattern, color_params, sleeve_color, card_types=['Hero'], use_hard_case=use_hard_case)
    
    # Apply base rotation based on X position
    target_width = 140
    target_height = 100
    
    if hero_zone['center_x'] < 0.5:  # Left half - rotate 90° clockwise
        base_rotation = -90
    else:  # Right half - rotate 90° counter-clockwise
        base_rotation = 90
    
    # Calculate scale for target size (after rotation: height->width, width->height)
    # Apply uniform scale factor to maintain consistent size across all cards in image
    scale = min(target_width / hero_img.height, target_height / hero_img.width) * uniform_scale_factor
    rotation = base_rotation + random.uniform(-3, 3)
    
    width, height = place_card_on_playmat(playmat, hero_img, x, y, rotation=rotation, scale=scale)
    return {
        'x': x, 'y': y, 'width': width, 'height': height,
        'label': f"{hero_zone['zone_name']}: {hero_card['name']}", 
        'zone_name': hero_zone['zone_name'],
        'card_name': hero_card['name']
    }

def place_standard_card(card, img_path, zone, image_cache, playmat, aug_config, 
                       blur_intensity, glare_pattern, color_params, sleeve_color, 
                       light_source_pos, img_width, img_height, uniform_scale_factor=1.0):
    """Helper function to place standard zone card with augmentations and rotation (OPTIMIZATION: eliminates code duplication)."""
    x, y, zone_w, zone_h = yolo_to_pixel_coords(zone, img_width, img_height)
    card_img = image_cache[str(img_path)]  # OPTIMIZATION: use cached image
    
    if aug_config is not None:
        card_center_x = zone['center_x'] * img_width
        card_center_y = zone['center_y'] * img_height
        glare_intensity = calculate_glare_intensity(card_center_x, card_center_y, light_source_pos, img_width, img_height)
        
        # Check if this card type is eligible for hard case (Equipment, Weapon, or Off-Hand)
        card_types = card.get('types', [])
        types_lower = [t.lower() for t in card_types]
        is_eligible = any(t in ['equipment', 'weapon', 'off-hand'] for t in types_lower)
        # Each eligible card has 50% chance of getting hard case (per card decision)
        should_use_hard_case = is_eligible and (random.random() < 0.5)
        
        card_img = apply_card_augmentations(card_img, aug_config, blur_intensity, glare_intensity, glare_pattern, color_params, sleeve_color, card_types=card_types, use_hard_case=should_use_hard_case)
    
    # Standard zones: target 140x100px after rotation
    target_width = 140
    target_height = 100
    
    # Apply base rotation based on X position
    if zone['center_x'] < 0.5:  # Left half - rotate 90° clockwise
        base_rotation = -90
    else:  # Right half - rotate 90° counter-clockwise
        base_rotation = 90
    
    # Calculate scale to achieve target size (after rotation: height->width, width->height)
    # Apply uniform scale factor to maintain consistent size across all cards in image
    scale = min(target_width / card_img.height, target_height / card_img.width) * uniform_scale_factor
    
    # Add small random rotation on top of base rotation
    rotation = base_rotation + random.uniform(-3, 3)
    card_width, card_height = place_card_on_playmat(playmat, card_img, x, y, rotation=rotation, scale=scale)
    
    return {
        'x': x, 'y': y, 'width': card_width, 'height': card_height,
        'label': f"{zone['zone_name']}: {card['name']}", 
        'zone_name': zone['zone_name'],
        'card_name': card['name']
    }

def place_combat_chain_card(card, img_path, zone, chain_name, zone_id, card_placements, 
                            image_cache, aug_config, blur_intensity, glare_pattern, 
                            color_params, sleeve_color, light_source_pos, img_width, img_height, playmat, uniform_scale_factor=1.0):
    """Helper function to place combat chain card with rotation and overlap detection (OPTIMIZATION: eliminates code duplication)."""
    card_img_original = image_cache[str(img_path)]  # OPTIMIZATION: use cached image
    rotation = random.uniform(-180, 180)
    target_width, target_height = 140, 100
    # Apply uniform scale factor to maintain consistent size across all cards in image
    scale = min(target_width / card_img_original.height, target_height / card_img_original.width) * uniform_scale_factor
    
    # Calculate bounding box estimate
    scaled_width = int(card_img_original.width * scale)
    scaled_height = int(card_img_original.height * scale)
    max_dim = int(math.sqrt(scaled_width**2 + scaled_height**2))
    
    # Find valid position
    position = find_valid_position_in_zone(zone, max_dim, max_dim, card_placements,
                                           img_width, img_height, max_overlap_pct=50, max_attempts=50)
    
    if not position:
        return None
    
    x, y = position
    
    # Apply augmentations (combat chain cards don't get hard cases, only sleeves)
    if aug_config is not None:
        card_center_x = x + max_dim / 2
        card_center_y = y + max_dim / 2
        glare_intensity = calculate_glare_intensity(card_center_x, card_center_y, light_source_pos, img_width, img_height)
        card_img = apply_card_augmentations(card_img_original, aug_config, blur_intensity, glare_intensity, glare_pattern, color_params, sleeve_color, card_types=card.get('types', []), use_hard_case=False)
    else:
        card_img = card_img_original
    
    final_width, final_height = place_card_on_playmat(playmat, card_img, x, y, rotation=rotation, scale=scale)
    print(f"      Placed: {card['name']} at ({x}, {y}) with rotation {rotation:.1f}°")
    
    return {
        'x': x, 'y': y, 'width': final_width, 'height': final_height,
        'label': f"{chain_name}: {card['name']}", 
        'zone_name': chain_name,
        'card_name': card['name']
    }

def main(enable_augmentations=True, draw_bboxes=True, preset_name=None):
    # Initialize timing dictionary
    timings = {}
    script_start = time.time()
    
    print("Testing synthetic playmat generation...")
    print("=" * 60)
    
    # Initialize CardSelector
    t0 = time.time()
    card_json_path = Path(r'c:\VS Code\FaB Code\data\card.json')
    weights_path = Path(r'c:\VS Code\FaB Code\data\card_popularity_weights_by_hero.json')
    card_dir = Path(r'c:\VS Code\FaB Code\data\images')
    background_dir = Path(r'c:\VS Code\FaB Code\data\Background Perfecting\images')
    labels_dir = Path(r'c:\VS Code\FaB Code\data\Background Perfecting\labels')
    
    selector = CardSelector(str(card_json_path), str(weights_path))
    timings['initialization'] = time.time() - t0
    
    # Load augmentation config (optionally from preset)
    aug_config = get_config() if enable_augmentations else None
    blur_intensity = None
    glare_pattern = None
    color_params = None
    sleeve_color = None
    preset = None
    
    # Load preset if specified
    if preset_name is not None:
        try:
            from augmentation_presets import get_preset, get_preset_ranges
            preset = get_preset(preset_name)
            print(f"   Using preset: {preset.name}")
        except (ImportError, ValueError) as e:
            print(f"   Warning: Could not load preset '{preset_name}': {e}")
            preset = None
    
    if enable_augmentations:
        # Select blur intensity once for entire image (15-50%)
        # Use preset value if available, otherwise random
        if preset is not None:
            blur_intensity = preset.blur_intensity
            print(f"   Augmentations enabled (blur intensity: {blur_intensity:.1%} from preset)")
        else:
            blur_intensity = random.uniform(*aug_config.blur.probability_range)
            print(f"   Augmentations enabled (blur intensity: {blur_intensity:.1%})")
        
        # Generate glare pattern once for entire image (consistent across all cards)
        if aug_config.glare.enabled:
            if preset is not None:
                num_spots = preset.num_glare_spots
                print(f"   Glare pattern: {num_spots} spot(s) (from preset) at consistent relative positions")
            else:
                num_spots = random.randint(*aug_config.glare.num_spots_range)
                print(f"   Glare pattern: {num_spots} spot(s) at consistent relative positions")
            
            glare_pattern = []
            for _ in range(num_spots):
                glare_pattern.append({
                    'x_ratio': random.random(),  # 0.0-1.0, scales to card width
                    'y_ratio': random.random(),  # 0.0-1.0, scales to card height
                    # Very wide range: 20-160% of card size for large washout coverage with moderate intensity
                    'radius_ratio': random.uniform(0.20, 1.60)
                })
        
        # Generate color adjustment parameters once for entire image (consistent across all cards)
        if aug_config.color.enabled:
            if preset is not None:
                # Use preset values
                color_params = {
                    'brightness': preset.brightness,
                    'contrast': preset.contrast,
                    'saturation': preset.saturation,
                    'hue_shift': preset.hue_shift,
                    'tint_color': preset.tint_color,
                    'tint_intensity': preset.tint_intensity
                }
                if preset.tint_color:
                    print(f"   Color adjustment (preset): brightness={color_params['brightness']:.2f}, contrast={color_params['contrast']:.2f}, saturation={color_params['saturation']:.2f}, hue_shift={color_params['hue_shift']}°, tint={color_params['tint_color']}@{color_params['tint_intensity']:.1%}")
                else:
                    print(f"   Color adjustment (preset): brightness={color_params['brightness']:.2f}, contrast={color_params['contrast']:.2f}, saturation={color_params['saturation']:.2f}, hue_shift={color_params['hue_shift']}°")
            else:
                # Random values from config ranges
                color_params = {
                    'brightness': random.uniform(*aug_config.color.brightness_range),
                    'contrast': random.uniform(*aug_config.color.contrast_range),
                    'saturation': random.uniform(*aug_config.color.saturation_range),
                    'hue_shift': random.randint(*aug_config.color.hue_shift_range)
                }
                # Add tint parameters with probability (consistent across all cards in image)
                if random.random() < aug_config.color.tint_probability:
                    color_params['tint_color'] = random.choice(list(aug_config.color.tint_colors.keys()))
                    color_params['tint_intensity'] = random.uniform(*aug_config.color.tint_intensity_range)
                    print(f"   Color adjustment: brightness={color_params['brightness']:.2f}, contrast={color_params['contrast']:.2f}, saturation={color_params['saturation']:.2f}, hue_shift={color_params['hue_shift']}°, tint={color_params['tint_color']}@{color_params['tint_intensity']:.1%}")
                else:
                    # No tint - set to None explicitly so all cards get the same treatment
                    color_params['tint_color'] = None
                    color_params['tint_intensity'] = None
                    print(f"   Color adjustment: brightness={color_params['brightness']:.2f}, contrast={color_params['contrast']:.2f}, saturation={color_params['saturation']:.2f}, hue_shift={color_params['hue_shift']}°")
        
        # Generate sleeve color once for entire image (consistent across all cards)
        # Common sleeve colors: black, white, blue, red, green, purple, clear (None)
        sleeve_colors = [
            None,  # No sleeve (50% chance when selected)
            (0, 0, 0),  # Black
            (255, 255, 255),  # White
            (50, 50, 200),  # Blue
            (200, 50, 50),  # Red
            (50, 150, 50),  # Green
            (150, 50, 150),  # Purple
            (200, 150, 50),  # Gold
            (100, 100, 100),  # Grey
        ]
        sleeve_color = random.choice(sleeve_colors)
        
        if sleeve_color is not None:
            print(f"   Sleeve color: RGB{sleeve_color}")
        else:
            print(f"   Sleeve: None (bare cards)")
        print(f"   Hard cases: 50% chance per Equipment/Weapon/Hero/Off-Hand card")
        
        # Generate uniform scale factor for entire image (consistent across all cards)
        # Random scale between 0.93 and 1.07 (±7%)
        uniform_scale_factor = random.uniform(0.93, 1.07)
        print(f"   Uniform card scale: {uniform_scale_factor:.3f}× ({(uniform_scale_factor - 1.0) * 100:+.1f}%)")
    else:
        uniform_scale_factor = 1.0  # No scaling when augmentations disabled
            
    if not draw_bboxes:
        print(f"   Bounding box visualization disabled")

    # Load background and labels
    t0 = time.time()
    print("\n1. Loading background and labels...")
    playmat, bg_path = load_random_background(str(background_dir))
    zones = load_label_file(bg_path, str(labels_dir))
    
    img_width, img_height = playmat.size
    print(f"   Image size: {img_width}x{img_height}")
    timings['load_background_labels'] = time.time() - t0
    
    # Apply massive color transformations to window area if augmentations enabled
    t0 = time.time()
    if enable_augmentations:
        # Find window zone
        window_zone = None
        for zone in zones:
            if zone['zone_name'] == 'Window':
                window_zone = zone
                break
        
        if window_zone:
            # Apply random color transformation to window area
            playmat = apply_window_color_transformation(playmat, window_zone)
            
            # Replace outside window area with random color/texture (75% probability)
            playmat = replace_outside_window_area(playmat, window_zone, probability=0.75)
    
    timings['window_transformations'] = time.time() - t0
    
    # Select light source position for glare (near window zone if it exists)
    t0 = time.time()
    light_source_pos = None
    if enable_augmentations and aug_config.glare.enabled:
        # Find window zone
        window_zone = None
        for zone in zones:
            if zone['zone_name'] == 'Window':
                window_zone = zone
                break
        
        if window_zone:
            # Pick a random point along the window border (simulating off-screen light)
            window_center_x = window_zone['center_x'] * img_width
            window_center_y = window_zone['center_y'] * img_height
            window_w = window_zone['width'] * img_width
            window_h = window_zone['height'] * img_height
            
            # Randomly select a point on the window perimeter
            side = random.choice(['top', 'bottom', 'left', 'right'])
            if side == 'top':
                light_source_pos = (window_center_x + random.uniform(-window_w/2, window_w/2), 
                                   window_center_y - window_h/2)
            elif side == 'bottom':
                light_source_pos = (window_center_x + random.uniform(-window_w/2, window_w/2),
                                   window_center_y + window_h/2)
            elif side == 'left':
                light_source_pos = (window_center_x - window_w/2,
                                   window_center_y + random.uniform(-window_h/2, window_h/2))
            else:  # right
                light_source_pos = (window_center_x + window_w/2,
                                   window_center_y + random.uniform(-window_h/2, window_h/2))
            
            print(f"   Light source at: ({light_source_pos[0]:.0f}, {light_source_pos[1]:.0f})")
    
    timings['light_source_setup'] = time.time() - t0
    
    # Create zone lookup dictionaries for O(1) access
    zones_by_class_id = {z['class_id']: z for z in zones}
    zones_by_name = {z['zone_name']: z for z in zones}
    
    # Select two heroes
    t0 = time.time()
    print("\n2. Selecting heroes...")
    hero1_key, hero1_card, hero1_weights = selector.select_random_hero()
    print(f"   Hero 1: {hero1_card['name']} ({hero1_key})")
    
    hero2_key, hero2_card, hero2_weights = selector.select_random_hero()
    print(f"   Hero 2: {hero2_card['name']} ({hero2_key})")
    
    # Find hero images
    hero1_img_path = selector.find_card_image(hero1_card, str(card_dir))
    if not hero1_img_path:
        print(f"   ERROR: Could not find hero 1 image!")
        return
    print(f"   Hero 1 image: {hero1_img_path}")
    
    hero2_img_path = selector.find_card_image(hero2_card, str(card_dir))
    if not hero2_img_path:
        print(f"   ERROR: Could not find hero 2 image!")
        return
    print(f"   Hero 2 image: {hero2_img_path}")
    
    # Select format for this image (CC 80%, LL 15%, Blitz 5%)
    from card_selector import select_format
    format = select_format()
    format_names = {'cc': 'Classic Constructed', 'll': 'Living Legend', 'blitz': 'Blitz'}
    print(f"\n   Format: {format_names[format]}")
    
    # Build card pools for both heroes
    print("\n3. Building card pools...")
    hero1_pools = selector.build_card_pools(hero1_card, hero1_weights, format)
    print(f"   Hero 1 - Weighted: {len(hero1_pools['weighted'])}, Generic: {len(hero1_pools['generic'])}, Class-only: {len(hero1_pools['class_only'])}, Talent-only: {len(hero1_pools['talent_only'])}, Both: {len(hero1_pools['both'])}")
    
    hero2_pools = selector.build_card_pools(hero2_card, hero2_weights, format)
    print(f"   Hero 2 - Weighted: {len(hero2_pools['weighted'])}, Generic: {len(hero2_pools['generic'])}, Class-only: {len(hero2_pools['class_only'])}, Talent-only: {len(hero2_pools['talent_only'])}, Both: {len(hero2_pools['both'])}")
    
    # Pre-build complete card pool lists (optimization: avoid repeated concatenation)
    all_hero1_cards = (hero1_pools['weighted'] + hero1_pools['generic'] + 
                       hero1_pools['class_only'] + hero1_pools['talent_only'] + 
                       hero1_pools['both'])
    all_hero2_cards = (hero2_pools['weighted'] + hero2_pools['generic'] + 
                       hero2_pools['class_only'] + hero2_pools['talent_only'] + 
                       hero2_pools['both'])
    
    # Partition zones in single pass (optimization: 4 loops → 1 loop)
    hero1_zones, hero2_zones, hero1_available_zones, hero2_available_zones = partition_zones(zones)
    
    print(f"   Hero 1 zones: {len(hero1_zones)}, Hero 2 zones: {len(hero2_zones)}")
    
    # Select cards for each hero (with zone-specific filtering)
    print("\n4. Selecting cards for playmat...")
    
    # Sort zones so Weapon comes before Weapon or Off-Hand (optimization: O(1) dict lookup)
    hero1_available_zones.sort(key=lambda z: get_zone_sort_key(z['zone_name']))
    hero2_available_zones.sort(key=lambda z: get_zone_sort_key(z['zone_name']))
    
    # Select cards for Hero 1 based on zones (excluding combat chain zones)
    hero1_cards = []
    hero1_zones_used = []  # Track which zones were used
    hero1_weapon_state = {'weapon_is_2h': False}
    
    for zone in hero1_available_zones:
        # Skip combat chain zones - they're handled separately
        if zone['zone_name'] in ['Combat Chain 1', 'Combat Chain 2']:
            continue
        
        # Use card selector's zone-aware selection method
        card = selector.select_card_for_zone(
            hero1_pools, 
            zone['zone_name'], 
            all_hero1_cards, 
            hero1_weapon_state,
            pitch_weighting=True
        )
        
        if card:
                
            hero1_cards.append(card)
            hero1_zones_used.append(zone)  # Track this zone was used
            
            # Debug: print card types for weapon and equipment zones
            if 'Weapon' in zone['zone_name'] or zone['zone_name'] in ['Head', 'Chest', 'Arms', 'Legs']:
                print(f"   Hero 1 {zone['zone_name']}: {card['name']} - Types: {card.get('types', [])}")
            
            # Track if we selected a 2H weapon in the main weapon slot (uppercase!)
            if zone['zone_name'] == 'Weapon' and '2H' in card.get('types', []):
                hero1_weapon_state['weapon_is_2h'] = True
        else:
            if zone['zone_name'] in ['Weapon or Off-Hand', 'Weapon or Off-Hand 2'] and hero1_weapon_state['weapon_is_2h']:
                print(f"   Hero 1: Skipping {zone['zone_name']} (2H weapon equipped)")
            else:
                print(f"   WARNING: No valid cards for Hero 1 zone {zone['zone_name']}")
    
    # Select cards for Hero 2 based on zones (excluding combat chain zones)
    hero2_cards = []
    hero2_zones_used = []  # Track which zones were used
    hero2_weapon_state = {'weapon_is_2h': False}
    
    for zone in hero2_available_zones:
        # Skip combat chain zones - they're handled separately
        if zone['zone_name'] in ['Combat Chain 1', 'Combat Chain 2']:
            continue
        
        # Use card selector's zone-aware selection method
        card = selector.select_card_for_zone(
            hero2_pools, 
            zone['zone_name'], 
            all_hero2_cards, 
            hero2_weapon_state,
            pitch_weighting=True
        )
        
        if card:
                
            hero2_cards.append(card)
            hero2_zones_used.append(zone)  # Track this zone was used
            
            # Debug: print card types for weapon and equipment zones
            if 'Weapon' in zone['zone_name'] or zone['zone_name'] in ['Head 2', 'Chest 2', 'Arms 2', 'Legs 2']:
                print(f"   Hero 2 {zone['zone_name']}: {card['name']} - Types: {card.get('types', [])}")
            
            # Track if we selected a 2H weapon in the main weapon slot (uppercase!)
            if zone['zone_name'] == 'Weapon 2' and '2H' in card.get('types', []):
                hero2_weapon_state['weapon_is_2h'] = True
        else:
            if zone['zone_name'] in ['Weapon or Off-Hand', 'Weapon or Off-Hand 2'] and hero2_weapon_state['weapon_is_2h']:
                print(f"   Hero 2: Skipping {zone['zone_name']} (2H weapon equipped)")
            else:
                print(f"   WARNING: No valid cards for Hero 2 zone {zone['zone_name']}")
    
    print(f"   Selected {len(hero1_cards)} cards for Hero 1")
    print(f"   Selected {len(hero2_cards)} cards for Hero 2")
    
    timings['hero_card_selection'] = time.time() - t0

    # Select combat chain cards (0-15 total, split between both combat chains)
    t0 = time.time()
    print("\n   Selecting combat chain cards...")
    combat_chain_total = random.randint(0, 15)
    print(f"   Total combat chain cards: {combat_chain_total}")
    
    # Find combat chain zones using lookup dict
    combat_chain1_zone = zones_by_class_id.get(7)  # Combat Chain 1
    combat_chain2_zone = zones_by_class_id.get(8)  # Combat Chain 2
    
    combat_chain1_cards = []
    combat_chain2_cards = []
    
    if combat_chain_total > 0 and (combat_chain1_zone or combat_chain2_zone):
        # Pre-filter valid combat chain cards once (optimization: avoid repeated filtering in loop)
        valid_cc1_cards = selector.filter_cards_for_zone(all_hero1_cards, 'Combat Chain 1', None)
        valid_cc2_cards = selector.filter_cards_for_zone(all_hero2_cards, 'Combat Chain 2', None)
        
        # Pre-filter tokens using tokens_only parameter (optimization: single efficient filter)
        valid_cc1_tokens = selector.filter_cards_for_zone(all_hero1_cards, 'Combat Chain 1', None, tokens_only=True)
        valid_cc2_tokens = selector.filter_cards_for_zone(all_hero2_cards, 'Combat Chain 2', None, tokens_only=True)
        
        # Determine if we should include tokens (5% chance)
        include_tokens = random.random() < 0.05
        num_tokens = 0
        if include_tokens and combat_chain_total > 0:
            # 1-2 tokens
            num_tokens = random.randint(1, min(2, combat_chain_total))
            print(f"   Including {num_tokens} token(s)")
        
        tokens_added = 0
        
        # Split cards between both combat chains (randomly distribute)
        for i in range(combat_chain_total):
            # Decide if this card should be a token
            should_be_token = (tokens_added < num_tokens)
            
            # Randomly assign to hero 1 or hero 2 combat chain
            if random.random() < 0.5 and combat_chain1_zone:
                # Hero 1 combat chain
                if should_be_token and valid_cc1_tokens:
                    # Select directly from token pool
                    card = random.choice(valid_cc1_tokens)
                    combat_chain1_cards.append(card)
                    tokens_added += 1
                    print(f"      Added token: {card['name']}")
                elif valid_cc1_cards:
                    # Regular card selection
                    card = selector.select_card(hero1_pools)
                    if card in valid_cc1_cards:
                        combat_chain1_cards.append(card)
            elif combat_chain2_zone:
                # Hero 2 combat chain
                if should_be_token and valid_cc2_tokens:
                    # Select directly from token pool
                    card = random.choice(valid_cc2_tokens)
                    combat_chain2_cards.append(card)
                    tokens_added += 1
                    print(f"      Added token: {card['name']}")
                elif valid_cc2_cards:
                    # Regular card selection
                    card = selector.select_card(hero2_pools)
                    if card in valid_cc2_cards:
                        combat_chain2_cards.append(card)
        
        print(f"   Combat Chain 1 cards: {len(combat_chain1_cards)}")
        print(f"   Combat Chain 2 cards: {len(combat_chain2_cards)}")
        if tokens_added > 0:
            print(f"   Tokens added: {tokens_added}")
    
    timings['combat_chain_selection'] = time.time() - t0
    
    selected_cards = hero1_cards + hero2_cards + combat_chain1_cards + combat_chain2_cards
    
    # Find all card images in one batch operation (optimization: single pass instead of 4 loops)
    t0 = time.time()
    print("\n5. Finding card images...")
    
    # Create list of all cards with their group labels for organized output
    all_cards_to_find = (
        [('Hero 1', i, card) for i, card in enumerate(hero1_cards)] +
        [('Hero 2', i, card) for i, card in enumerate(hero2_cards)] +
        [('Combat Chain 1', i, card) for i, card in enumerate(combat_chain1_cards)] +
        [('Combat Chain 2', i, card) for i, card in enumerate(combat_chain2_cards)]
    )
    
    # Batch find all images
    card_image_lookup = {}  # Map card name -> (card, img_path)
    for group, idx, card in all_cards_to_find:
        img_path = selector.find_card_image(card, str(card_dir))
        if img_path:
            card_image_lookup[id(card)] = (card, img_path)  # Use id() to handle duplicate card names
            print(f"   {group} [{idx+1}/{len([c for g,i,c in all_cards_to_find if g == group])}] Found: {card['name']}")
        else:
            print(f"   {group} [{idx+1}/{len([c for g,i,c in all_cards_to_find if g == group])}] MISSING: {card['name']}")
    
    # Distribute found images back to separate lists (maintains original order)
    hero1_card_images = [(card, card_image_lookup[id(card)][1]) for card in hero1_cards if id(card) in card_image_lookup]
    hero2_card_images = [(card, card_image_lookup[id(card)][1]) for card in hero2_cards if id(card) in card_image_lookup]
    combat_chain1_card_images = [(card, card_image_lookup[id(card)][1]) for card in combat_chain1_cards if id(card) in card_image_lookup]
    combat_chain2_card_images = [(card, card_image_lookup[id(card)][1]) for card in combat_chain2_cards if id(card) in card_image_lookup]
    
    total_found = len(hero1_card_images) + len(hero2_card_images) + len(combat_chain1_card_images) + len(combat_chain2_card_images)
    total_cards = len(hero1_cards) + len(hero2_cards) + len(combat_chain1_cards) + len(combat_chain2_cards)
    print(f"\n   Success rate: {total_found}/{total_cards} ({100*total_found/total_cards:.1f}%)")
    
    # Load all images into memory cache (optimization: load once, reuse many times)
    print("\n   Loading images into memory...")
    image_cache = {}
    
    # Cache hero images
    image_cache[str(hero1_img_path)] = Image.open(hero1_img_path)
    image_cache[str(hero2_img_path)] = Image.open(hero2_img_path)
    
    # Cache all card images
    for card, img_path in hero1_card_images + hero2_card_images + combat_chain1_card_images + combat_chain2_card_images:
        path_str = str(img_path)
        if path_str not in image_cache:
            image_cache[path_str] = Image.open(img_path)
    
    print(f"   Loaded {len(image_cache)} unique images into cache")
    
    timings['load_images_to_cache'] = time.time() - t0
    
    # Place cards in zones
    t0 = time.time()
    print("\n6. Placing cards in zones...")
    card_placements = []
    
    if zones:
        # Find hero zones using lookup dict (class_id 13 for Hero, 14 for Hero 2)
        hero1_zone = zones_by_class_id.get(13)
        hero2_zone = zones_by_class_id.get(14)
        
        # Place Hero 1 (OPTIMIZATION: using helper function)
        if hero1_zone:
            placement = place_hero_card(hero1_img_path, hero1_zone, hero1_card, image_cache, playmat,
                                       aug_config, blur_intensity, glare_pattern, color_params, 
                                       sleeve_color, light_source_pos, img_width, img_height, uniform_scale_factor)
            card_placements.append(placement)
        
        # Place Hero 2 (OPTIMIZATION: using helper function)
        if hero2_zone:
            placement = place_hero_card(hero2_img_path, hero2_zone, hero2_card, image_cache, playmat,
                                       aug_config, blur_intensity, glare_pattern, color_params, 
                                       sleeve_color, light_source_pos, img_width, img_height, uniform_scale_factor)
            card_placements.append(placement)
        
        # Get hero 1 zones (no number or ending without ' 2')
        hero1_available_zones = [z for z in zones if z['class_id'] not in [13, 14, 24] and not z['zone_name'].endswith(' 2')]
        
        # Get hero 2 zones (ending with ' 2')
        hero2_available_zones = [z for z in zones if z['zone_name'].endswith(' 2') and z['class_id'] not in [13, 14, 24]]
        
        # Place Hero 1 cards (OPTIMIZATION: using helper function)
        for (card, img_path), zone in zip(hero1_card_images, hero1_zones_used):
            placement = place_standard_card(card, img_path, zone, image_cache, playmat, aug_config,
                                          blur_intensity, glare_pattern, color_params, sleeve_color,
                                          light_source_pos, img_width, img_height, uniform_scale_factor)
            card_placements.append(placement)
        
        # Place Hero 2 cards (OPTIMIZATION: using helper function)
        for (card, img_path), zone in zip(hero2_card_images, hero2_zones_used):
            print(f"   DEBUG: Placing {card['name']} (types: {card.get('types', [])}) in zone {zone['zone_name']}")
            placement = place_standard_card(card, img_path, zone, image_cache, playmat, aug_config,
                                          blur_intensity, glare_pattern, color_params, sleeve_color,
                                          light_source_pos, img_width, img_height, uniform_scale_factor)
            card_placements.append(placement)
        
        # Place Combat Chain 1 cards (OPTIMIZATION: using helper function)
        if combat_chain1_card_images and combat_chain1_zone:
            print(f"\n   Placing {len(combat_chain1_card_images)} cards in Combat Chain 1...")
            for card, img_path in combat_chain1_card_images:
                placement = place_combat_chain_card(card, img_path, combat_chain1_zone, "Combat Chain 1", 7,
                                                   card_placements, image_cache, aug_config, blur_intensity,
                                                   glare_pattern, color_params, sleeve_color, light_source_pos,
                                                   img_width, img_height, playmat, uniform_scale_factor)
                if placement:
                    card_placements.append(placement)
                else:
                    print(f"      WARNING: Could not find valid position for {card['name']} (overlap limit reached)")
        
        # Place Combat Chain 2 cards (OPTIMIZATION: using helper function)
        if combat_chain2_card_images and combat_chain2_zone:
            print(f"\n   Placing {len(combat_chain2_card_images)} cards in Combat Chain 2...")
            for card, img_path in combat_chain2_card_images:
                placement = place_combat_chain_card(card, img_path, combat_chain2_zone, "Combat Chain 2", 8,
                                                   card_placements, image_cache, aug_config, blur_intensity,
                                                   glare_pattern, color_params, sleeve_color, light_source_pos,
                                                   img_width, img_height, playmat, uniform_scale_factor)
                if placement:
                    card_placements.append(placement)
                else:
                    print(f"      WARNING: Could not find valid position for {card['name']} (overlap limit reached)")
    else:
        # Fallback to grid if no zones
        print("   No zones found, using grid layout...")
        # Place heroes (OPTIMIZATION: use cached images)
        hero1_img = image_cache[str(hero1_img_path)]
        if aug_config is not None:
            glare_intensity = calculate_glare_intensity(50, 50, light_source_pos, img_width, img_height)
            # Heroes have 50% chance of getting hard case (per card decision)
            use_hard_case_hero1 = random.random() < 0.5
            hero1_img = apply_card_augmentations(hero1_img, aug_config, blur_intensity, glare_intensity, glare_pattern, color_params, sleeve_color, card_types=['Hero'], use_hard_case=use_hard_case_hero1)
        hero1_width, hero1_height = place_card_on_playmat(playmat, hero1_img, x=50, y=50, scale=0.6 * uniform_scale_factor)
        card_placements.append({'x': 50, 'y': 50, 'width': hero1_width, 'height': hero1_height, 'label': f"Hero 1: {hero1_card['name']}", 'zone_name': 'Hero', 'card_name': hero1_card['name']})
        
        hero2_img = image_cache[str(hero2_img_path)]
        if aug_config is not None:
            glare_intensity = calculate_glare_intensity(50, 500, light_source_pos, img_width, img_height)
            # Heroes have 50% chance of getting hard case (per card decision)
            use_hard_case_hero2 = random.random() < 0.5
            hero2_img = apply_card_augmentations(hero2_img, aug_config, blur_intensity, glare_intensity, glare_pattern, color_params, sleeve_color, card_types=['Hero'], use_hard_case=use_hard_case_hero2)
        hero2_width, hero2_height = place_card_on_playmat(playmat, hero2_img, x=50, y=500, scale=0.6 * uniform_scale_factor)
        card_placements.append({'x': 50, 'y': 500, 'width': hero2_width, 'height': hero2_height, 'label': f"Hero 2: {hero2_card['name']}", 'zone_name': 'Hero 2', 'card_name': hero2_card['name']})
        
        # Grid placement for cards
        card_scale = 0.4
        cards_per_row = 5
        card_spacing_x = 250
        card_spacing_y = 350
        start_x = 400
        start_y = 50
        
        all_card_images = hero1_card_images + hero2_card_images
        for i, (card, img_path) in enumerate(all_card_images):
            row = i // cards_per_row
            col = i % cards_per_row
            
            x = start_x + col * card_spacing_x
            y = start_y + row * card_spacing_y
            
            rotation = random.uniform(-5, 5)
            
            card_img = image_cache[str(img_path)]  # OPTIMIZATION: use cached image
            
            # Apply augmentations if enabled
            if aug_config is not None:
                # Check if this card type is eligible for hard case (Equipment, Weapon, or Off-Hand)
                card_types = card.get('types', [])
                types_lower = [t.lower() for t in card_types]
                is_eligible = any(t in ['equipment', 'weapon', 'off-hand'] for t in types_lower)
                # Each eligible card has 50% chance of getting hard case (per card decision)
                should_use_hard_case = is_eligible and (random.random() < 0.5)
                
                # Calculate glare based on position (for fallback grid layout)
                glare_intensity = calculate_glare_intensity(x, y, light_source_pos, img_width, img_height)
                card_img = apply_card_augmentations(card_img, aug_config, blur_intensity, glare_intensity, glare_pattern, color_params, sleeve_color, card_types=card_types, use_hard_case=should_use_hard_case)
            
            card_width, card_height = place_card_on_playmat(playmat, card_img, x, y, rotation=rotation, scale=card_scale * uniform_scale_factor)
            card_placements.append({'x': x, 'y': y, 'width': card_width, 'height': card_height, 'label': card['name'], 'zone_name': 'Card', 'card_name': card['name']})
    
    timings['place_cards'] = time.time() - t0
    
    # Apply occluders to playmat (after all cards placed, before bounding boxes)
    t0 = time.time()
    if enable_augmentations:
        print("\n7. Applying occluders to playmat...")
        tokens_dir = Path(__file__).parent.parent / "data" / "tokens"
        playmat = apply_occluders_to_playmat(playmat, card_placements, tokens_dir, probability_per_card=0.10, second_occluder_probability=0.01)
    
    timings['apply_occluders'] = time.time() - t0
    
    # Draw bounding boxes with labels (if enabled)
    t0 = time.time()
    if draw_bboxes:
        print("\n8. Drawing bounding boxes and labels...")
        draw = ImageDraw.Draw(playmat)
        for placement in card_placements:
            draw_bounding_box_with_label(
                draw, 
                placement['x'], 
                placement['y'], 
                placement['width'], 
                placement['height'], 
                placement['label']
            )
    else:
        print("\n8. Skipping bounding box visualization (draw_bboxes=False)")
    
    timings['draw_bboxes'] = time.time() - t0
    
    # Save output with proper train/val/test split
    t0 = time.time()
    base_dir = Path(r'c:\VS Code\FaB Code\data\synthetic')
    
    # Determine split (70% train, 20% val, 10% test)
    split_rand = random.random()
    if split_rand < 0.70:
        split = 'train'
    elif split_rand < 0.90:
        split = 'valid'
    else:
        split = 'test'
    
    # Create directories
    images_dir = base_dir / split / 'images'
    labels_dir = base_dir / split / 'labels'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # milliseconds
    filename = f'playmat_{timestamp}'
    
    # Save image
    output_path = images_dir / f'{filename}.jpg'
    playmat.save(output_path, quality=95)
    print(f"\n9. Saved image to: {output_path}")
    
    # Save YOLO labels using card name mapping
    label_path = labels_dir / f'{filename}.txt'
    save_yolo_labels(label_path, card_placements, CARD_NAME_TO_CLASS_ID, img_width, img_height)
    print(f"   Saved labels to: {label_path}")
    print(f"   Split: {split} ({len(card_placements)} cards labeled)")
    
    # Verify data.yaml exists (should be pre-generated with generate_card_data_yaml.py)
    create_data_yaml(base_dir, force_update=False)
    
    timings['save_outputs'] = time.time() - t0
    
    # Print timing breakdown
    total_time = time.time() - script_start
    timings['total'] = total_time
    
    print("\n" + "=" * 60)
    print("TIMING BREAKDOWN:")
    print("-" * 60)
    for key, duration in timings.items():
        if key != 'total':
            percent = (duration / total_time) * 100
            print(f"  {key:.<35} {duration:>6.3f}s ({percent:>5.1f}%)")
    print("-" * 60)
    print(f"  {'TOTAL':.<35} {total_time:>6.3f}s (100.0%)")
    print("=" * 60)
    print("SUCCESS! Playmat generated with labels.")

if __name__ == '__main__':
    # For testing: disable bbox drawing to see cards more clearly
    main(enable_augmentations=True, draw_bboxes=False)

