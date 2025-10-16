"""
Test script to generate one synthetic playmat image using CardSelector.
"""

import random
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from card_selector import CardSelector

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

def yolo_to_pixel_coords(zone, img_width, img_height):
    """Convert YOLO normalized coordinates to pixel coordinates."""
    center_x_px = zone['center_x'] * img_width
    center_y_px = zone['center_y'] * img_height
    width_px = zone['width'] * img_width
    height_px = zone['height'] * img_height
    
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

def place_card_on_playmat(playmat, card_img, x, y, rotation=0, scale=1.0):
    """Place a card image on the playmat at given position. Returns final dimensions."""
    # Convert to RGBA if not already
    if card_img.mode != 'RGBA':
        card_img = card_img.convert('RGBA')
    
    # Resize card
    new_width = int(card_img.width * scale)
    new_height = int(card_img.height * scale)
    card_resized = card_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Rotate if needed (expand=True creates transparent background)
    if rotation != 0:
        card_resized = card_resized.rotate(rotation, expand=True, resample=Image.Resampling.BICUBIC)
    
    # Get final dimensions after rotation
    final_width = card_resized.width
    final_height = card_resized.height
    
    # Paste with alpha channel
    playmat.paste(card_resized, (x, y), card_resized)
    
    return final_width, final_height

def main():
    print("Testing synthetic playmat generation...")
    print("=" * 60)
    
    # Initialize CardSelector
    card_json_path = Path(r'c:\VS Code\FaB Code\data\card.json')
    weights_path = Path(r'c:\VS Code\FaB Code\data\card_popularity_weights_by_hero.json')
    card_dir = Path(r'c:\VS Code\FaB Code\data\images')
    background_dir = Path(r'c:\VS Code\FaB Code\data\Background Perfecting\images')
    labels_dir = Path(r'c:\VS Code\FaB Code\data\Background Perfecting\labels')
    
    selector = CardSelector(str(card_json_path), str(weights_path))
    
    # Load background and labels
    print("\n1. Loading background and labels...")
    playmat, bg_path = load_random_background(str(background_dir))
    zones = load_label_file(bg_path, str(labels_dir))
    
    img_width, img_height = playmat.size
    print(f"   Image size: {img_width}x{img_height}")
    
    # Create zone lookup dictionaries for O(1) access
    zones_by_class_id = {z['class_id']: z for z in zones}
    zones_by_name = {z['zone_name']: z for z in zones}
    
    # Select two heroes
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
    
    # Select combat chain cards (0-15 total, split between both combat chains)
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
        
        # Determine if we should include tokens (50% chance)
        include_tokens = random.random() < 0.5
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
                    # Try to select a token using weighted selection
                    card = selector.select_card(hero1_pools)
                    if card in valid_cc1_tokens:
                        combat_chain1_cards.append(card)
                        tokens_added += 1
                        print(f"      Added token: {card['name']}")
                        continue
                
                # Regular card selection (or fallback if no tokens found)
                if valid_cc1_cards:
                    card = selector.select_card(hero1_pools)
                    if card in valid_cc1_cards:
                        combat_chain1_cards.append(card)
            elif combat_chain2_zone:
                # Hero 2 combat chain
                if should_be_token and valid_cc2_tokens:
                    # Try to select a token using weighted selection
                    card = selector.select_card(hero2_pools)
                    if card in valid_cc2_tokens:
                        combat_chain2_cards.append(card)
                        tokens_added += 1
                        print(f"      Added token: {card['name']}")
                        continue
                
                # Regular card selection (or fallback if no tokens found)
                if valid_cc2_cards:
                    card = selector.select_card(hero2_pools)
                    if card in valid_cc2_cards:
                        combat_chain2_cards.append(card)
        
        print(f"   Combat Chain 1 cards: {len(combat_chain1_cards)}")
        print(f"   Combat Chain 2 cards: {len(combat_chain2_cards)}")
        if tokens_added > 0:
            print(f"   Tokens added: {tokens_added}")
    
    selected_cards = hero1_cards + hero2_cards + combat_chain1_cards + combat_chain2_cards
    
    # Find all card images in one batch operation (optimization: single pass instead of 4 loops)
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
    
    # Place cards in zones
    print("\n6. Placing cards in zones...")
    card_placements = []
    
    if zones:
        # Find hero zones using lookup dict (class_id 13 for Hero, 14 for Hero 2)
        hero1_zone = zones_by_class_id.get(13)
        hero2_zone = zones_by_class_id.get(14)
        
        # Place Hero 1
        if hero1_zone:
            x, y, zone_w, zone_h = yolo_to_pixel_coords(hero1_zone, img_width, img_height)
            hero1_img = Image.open(hero1_img_path)
            
            # Apply base rotation based on X position
            target_width = 140
            target_height = 100
            
            if hero1_zone['center_x'] < 0.5:  # Left half - rotate 90° clockwise
                base_rotation = -90
            else:  # Right half - rotate 90° counter-clockwise
                base_rotation = 90
            
            # Calculate scale for target size (after rotation: height->width, width->height)
            scale = min(target_width / hero1_img.height, target_height / hero1_img.width)
            rotation = base_rotation + random.uniform(-3, 3)
            
            hero1_width, hero1_height = place_card_on_playmat(playmat, hero1_img, x, y, rotation=rotation, scale=scale)
            card_placements.append({
                'x': x, 'y': y, 'width': hero1_width, 'height': hero1_height,
                'label': f"{hero1_zone['zone_name']}: {hero1_card['name']}", 'zone_id': hero1_zone['class_id']
            })
        
        # Place Hero 2
        if hero2_zone:
            x, y, zone_w, zone_h = yolo_to_pixel_coords(hero2_zone, img_width, img_height)
            hero2_img = Image.open(hero2_img_path)
            
            # Apply base rotation based on X position
            target_width = 140
            target_height = 100
            
            if hero2_zone['center_x'] < 0.5:  # Left half - rotate 90° clockwise
                base_rotation = -90
            else:  # Right half - rotate 90° counter-clockwise
                base_rotation = 90
            
            # Calculate scale for target size (after rotation: height->width, width->height)
            scale = min(target_width / hero2_img.height, target_height / hero2_img.width)
            rotation = base_rotation + random.uniform(-3, 3)
            
            hero2_width, hero2_height = place_card_on_playmat(playmat, hero2_img, x, y, rotation=rotation, scale=scale)
            card_placements.append({
                'x': x, 'y': y, 'width': hero2_width, 'height': hero2_height,
                'label': f"{hero2_zone['zone_name']}: {hero2_card['name']}", 'zone_id': hero2_zone['class_id']
            })
        
        # Get hero 1 zones (no number or ending without ' 2')
        hero1_available_zones = [z for z in zones if z['class_id'] not in [13, 14, 24] and not z['zone_name'].endswith(' 2')]
        
        # Get hero 2 zones (ending with ' 2')
        hero2_available_zones = [z for z in zones if z['zone_name'].endswith(' 2') and z['class_id'] not in [13, 14, 24]]
        
        # Place Hero 1 cards
        for (card, img_path), zone in zip(hero1_card_images, hero1_zones_used):
            x, y, zone_w, zone_h = yolo_to_pixel_coords(zone, img_width, img_height)
            card_img = Image.open(img_path)
            
            # Standard zones: target 140x100px after rotation (zone is 140x100px)
            # After 90° rotation, card height becomes width and card width becomes height
            target_width = 140  # Target width after rotation
            target_height = 100  # Target height after rotation
            
            # Apply base rotation based on X position
            if zone['center_x'] < 0.5:  # Left half - rotate 90° clockwise
                base_rotation = -90
            else:  # Right half - rotate 90° counter-clockwise
                base_rotation = 90
            
            # Calculate scale to achieve target size without stretching
            # After rotation: original height -> width, original width -> height
            scale = min(target_width / card_img.height, target_height / card_img.width)
            
            # Add small random rotation on top of base rotation
            rotation = base_rotation + random.uniform(-3, 3)
            card_width, card_height = place_card_on_playmat(playmat, card_img, x, y, rotation=rotation, scale=scale)
            card_placements.append({
                'x': x, 'y': y, 'width': card_width, 'height': card_height,
                'label': f"{zone['zone_name']}: {card['name']}", 'zone_id': zone['class_id']
            })
        
        # Place Hero 2 cards
        for (card, img_path), zone in zip(hero2_card_images, hero2_zones_used):
            print(f"   DEBUG: Placing {card['name']} (types: {card.get('types', [])}) in zone {zone['zone_name']}")
            x, y, zone_w, zone_h = yolo_to_pixel_coords(zone, img_width, img_height)
            card_img = Image.open(img_path)
            
            # Standard zones: target 140x100px after rotation (zone is 140x100px)
            # After 90° rotation, card height becomes width and card width becomes height
            target_width = 140  # Target width after rotation
            target_height = 100  # Target height after rotation
            
            # Apply base rotation based on X position
            if zone['center_x'] < 0.5:  # Left half - rotate 90° clockwise
                base_rotation = -90
            else:  # Right half - rotate 90° counter-clockwise
                base_rotation = 90
            
            # Calculate scale to achieve target size without stretching
            # After rotation: original height -> width, original width -> height
            scale = min(target_width / card_img.height, target_height / card_img.width)
            
            # Add small random rotation on top of base rotation
            rotation = base_rotation + random.uniform(-3, 3)
            card_width, card_height = place_card_on_playmat(playmat, card_img, x, y, rotation=rotation, scale=scale)
            card_placements.append({
                'x': x, 'y': y, 'width': card_width, 'height': card_height,
                'label': f"{zone['zone_name']}: {card['name']}", 'zone_id': zone['class_id']
            })
        
        # Place Combat Chain 1 cards with scatter and overlap detection
        if combat_chain1_card_images and combat_chain1_zone:
            print(f"\n   Placing {len(combat_chain1_card_images)} cards in Combat Chain 1...")
            for card, img_path in combat_chain1_card_images:
                card_img = Image.open(img_path)
                
                # Random rotation for variety
                rotation = random.uniform(-180, 180)
                
                # Target 140x100px (same as standard zones)
                target_width = 140
                target_height = 100
                
                # Scale to target size (same calculation as standard zones at 90°)
                scale = min(target_width / card_img.height, target_height / card_img.width)
                
                # Calculate estimated dimensions after scaling AND rotating
                # For overlap detection, we need to estimate the bounding box after rotation
                scaled_width = int(card_img.width * scale)
                scaled_height = int(card_img.height * scale)
                # After arbitrary rotation, worst case is diagonal
                max_dim = int(math.sqrt(scaled_width**2 + scaled_height**2))
                card_width_estimate = max_dim
                card_height_estimate = max_dim
                
                # Find valid position with overlap detection
                position = find_valid_position_in_zone(
                    combat_chain1_zone, 
                    card_width_estimate, 
                    card_height_estimate, 
                    card_placements,
                    img_width, 
                    img_height,
                    max_overlap_pct=25,
                    max_attempts=50
                )
                
                if position:
                    x, y = position
                    final_width, final_height = place_card_on_playmat(playmat, card_img, x, y, rotation=rotation, scale=scale)
                    card_placements.append({
                        'x': x, 'y': y, 'width': final_width, 'height': final_height,
                        'label': f"Combat Chain 1: {card['name']}", 'zone_id': 7
                    })
                    print(f"      Placed: {card['name']} at ({x}, {y}) with rotation {rotation:.1f}°")
                else:
                    print(f"      WARNING: Could not find valid position for {card['name']} (overlap limit reached)")
        
        # Place Combat Chain 2 cards with scatter and overlap detection
        if combat_chain2_card_images and combat_chain2_zone:
            print(f"\n   Placing {len(combat_chain2_card_images)} cards in Combat Chain 2...")
            for card, img_path in combat_chain2_card_images:
                card_img = Image.open(img_path)
                
                # Random rotation for variety
                rotation = random.uniform(-180, 180)
                
                # Target 140x100px (same as standard zones)
                target_width = 140
                target_height = 100
                
                # Scale to target size (same calculation as standard zones at 90°)
                scale = min(target_width / card_img.height, target_height / card_img.width)
                
                # Calculate estimated dimensions after scaling AND rotating
                # For overlap detection, we need to estimate the bounding box after rotation
                scaled_width = int(card_img.width * scale)
                scaled_height = int(card_img.height * scale)
                # After arbitrary rotation, worst case is diagonal
                max_dim = int(math.sqrt(scaled_width**2 + scaled_height**2))
                card_width_estimate = max_dim
                card_height_estimate = max_dim
                
                # Find valid position with overlap detection
                position = find_valid_position_in_zone(
                    combat_chain2_zone, 
                    card_width_estimate, 
                    card_height_estimate, 
                    card_placements,
                    img_width, 
                    img_height,
                    max_overlap_pct=25,
                    max_attempts=50
                )
                
                if position:
                    x, y = position
                    final_width, final_height = place_card_on_playmat(playmat, card_img, x, y, rotation=rotation, scale=scale)
                    card_placements.append({
                        'x': x, 'y': y, 'width': final_width, 'height': final_height,
                        'label': f"Combat Chain 2: {card['name']}", 'zone_id': 8
                    })
                    print(f"      Placed: {card['name']} at ({x}, {y}) with rotation {rotation:.1f}°")
                else:
                    print(f"      WARNING: Could not find valid position for {card['name']} (overlap limit reached)")
    else:
        # Fallback to grid if no zones
        print("   No zones found, using grid layout...")
        # Place heroes
        hero1_img = Image.open(hero1_img_path)
        hero1_width, hero1_height = place_card_on_playmat(playmat, hero1_img, x=50, y=50, scale=0.6)
        card_placements.append({'x': 50, 'y': 50, 'width': hero1_width, 'height': hero1_height, 'label': f"Hero 1: {hero1_card['name']}"})
        
        hero2_img = Image.open(hero2_img_path)
        hero2_width, hero2_height = place_card_on_playmat(playmat, hero2_img, x=50, y=500, scale=0.6)
        card_placements.append({'x': 50, 'y': 500, 'width': hero2_width, 'height': hero2_height, 'label': f"Hero 2: {hero2_card['name']}"})
        
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
            
            card_img = Image.open(img_path)
            card_width, card_height = place_card_on_playmat(playmat, card_img, x, y, rotation=rotation, scale=card_scale)
            card_placements.append({'x': x, 'y': y, 'width': card_width, 'height': card_height, 'label': card['name']})
    
    # Draw bounding boxes with labels
    print("\n7. Drawing bounding boxes and labels...")
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
    
    # Save output
    output_dir = Path(r'c:\VS Code\FaB Code\data\Background Perfecting\test')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # milliseconds
    output_path = output_dir / f'test_playmat_{timestamp}.jpg'
    playmat.save(output_path, quality=95)
    print(f"\n8. Saved playmat to: {output_path}")
    print("\n" + "=" * 60)
    print("SUCCESS! Playmat generated.")

if __name__ == '__main__':
    main()
