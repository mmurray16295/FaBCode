# ===================== PATH & RUN SETTINGS =====================
# Change these paths to select which sets and backgrounds to use

# Folder containing background images (playmat screenshots) with Card/Ref labels
TEMPLATE_BASE = r'C:\VS Code\FaB Code\data\images\YouTube_Labeled'

# Base folder containing all card image sets
CARD_IMAGES_BASE = r'c:\VS Code\FaB Code\data\images'

# Folder to save synthetic output (will create train, test, valid subfolders)
OUTPUT_BASE_DIR = r'c:\VS Code\FaB Code\data\synthetic_2'

# Folder to save positioning cache files (saves card placement calculations)
POSITIONING_CACHE_DIR = r'c:\VS Code\FaB Code\data\positioning_cache'

# Number of synthetic images to generate (set this for each run)
NUM_SYNTHETIC_IMAGES = 20  # <--- Change this value for your trial runs

# Path to card popularity weights JSON file
POPULARITY_WEIGHTS_PATH = r'c:\VS Code\FaB Code\data\card_popularity_weights.json'

# Class IDs for Card and Ref (from YOLO labels)
CLASS_ID_CARD = 0
CLASS_ID_REF = 1

# ==============================================================

import os
import glob
import random
import uuid
from PIL import Image, ImageFilter, ImageDraw
import yaml
import argparse
import re
import json
import io
import numpy as np
import math
from collections import defaultdict

import pathlib

# ========================================================

# Automatically discover all card set directories (subdirectories in CARD_IMAGES_BASE)
# Excludes YouTube and YouTube_Labeled folders
CARD_SET_DIRS = []
if os.path.isdir(CARD_IMAGES_BASE):
    for item in os.listdir(CARD_IMAGES_BASE):
        item_path = os.path.join(CARD_IMAGES_BASE, item)
        # Include only directories, exclude YouTube folders
        if os.path.isdir(item_path) and 'YouTube' not in item:
            CARD_SET_DIRS.append(item_path)
    CARD_SET_DIRS.sort()  # Sort alphabetically for consistent ordering

print(f"[init] Discovered {len(CARD_SET_DIRS)} card set directories")

# ========================================================

# ========================================================

def jpeg_round_trip(pil_img, q1=(30,70), q2=None):
    """One or two JPEG compressions."""
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=random.randint(*q1))
    buf.seek(0)
    img = Image.open(buf).convert(pil_img.mode)
    if q2:
        buf2 = io.BytesIO()
        img.save(buf2, format="JPEG", quality=random.randint(*q2))
        buf2.seek(0)
        img = Image.open(buf2).convert(pil_img.mode)
    return img

def down_up_sample(pil_img, min_s=0.5, max_s=0.95):
    """Downscale by a random non-integer factor then upscale back."""
    w,h = pil_img.size
    s = random.uniform(min_s, max_s)
    new = pil_img.resize((max(1,int(w*s)), max(1,int(h*s))), random.choice([
        Image.BILINEAR, Image.BICUBIC, Image.LANCZOS
    ]))
    return new.resize((w,h), random.choice([Image.BILINEAR, Image.BICUBIC]))

def gamma_jitter(pil_img, lo=0.85, hi=1.15):
    g = random.uniform(lo, hi)
    arr = np.asarray(pil_img).astype(np.float32) / 255.0
    arr = np.clip(arr ** g, 0, 1)
    return Image.fromarray((arr*255).astype(np.uint8), mode=pil_img.mode)

def chroma_subsample(pil_img, sigma=(0.8, 2.0)):
    """Blur color channels to mimic 4:2:0."""
    ycbcr = pil_img.convert("YCbCr")
    Y, Cb, Cr = ycbcr.split()
    rad = random.uniform(*sigma)
    Cb = Cb.filter(ImageFilter.GaussianBlur(radius=rad))
    Cr = Cr.filter(ImageFilter.GaussianBlur(radius=rad))
    return Image.merge("YCbCr", (Y,Cb,Cr)).convert(pil_img.mode)

def tiny_blur_or_unsharp(pil_img):
    if random.random() < 0.5:
        return pil_img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.2)))
    # unsharp: blur then blend
    blur = pil_img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.6, 1.6)))
    return Image.blend(pil_img, blur, alpha=random.uniform(0.15, 0.35))

def add_letterbox_and_ui(pil_img, p_bars=0.5, p_ui=0.5):
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    w,h = img.size
    # bars
    if random.random() < p_bars:
        t = random.randint(2, max(2, h//20))   # top
        b = random.randint(2, max(2, h//20))   # bottom
        l = random.randint(0, max(0, w//50))   # left (pillarbox sometimes 0)
        r = random.randint(0, max(0, w//50))
        col = random.choice([(0,0,0), (8,8,8), (16,16,16), (24,24,24)])
        if t: draw.rectangle([0,0,w,t], fill=col)
        if b: draw.rectangle([0,h-b,w,h], fill=col)
        if l: draw.rectangle([0,0,l,h], fill=col)
        if r: draw.rectangle([w-r,0,w,h], fill=col)
    # UI hairlines
    if random.random() < p_ui:
        y = random.randint(0, h-1)
        draw.line([(0,y),(w-1,y)], fill=(random.randint(160,220),)*3, width=1)
        if random.random() < 0.3:
            x = random.randint(0, w-1)
            draw.line([(x,0),(x,h-1)], fill=(random.randint(160,220),)*3, width=1)
    return img

def add_cursor_or_mask(pil_img, p=0.15):
    if random.random() >= p: return pil_img
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    w,h = img.size
    # rectangle mask (like your overlay)
    if random.random() < 0.5:
        mw, mh = random.randint(w//12, w//6), random.randint(h//12, h//6)
        x = random.randint(0, max(0, w-mw)); y = random.randint(0, max(0, h-mh))
        draw.rectangle([x,y,x+mw,y+mh], fill=(0,0,0))
    else:
        # simple cursor triangle
        cx = random.randint(0, w-20); cy = random.randint(0, h-20)
        pts = [(cx,cy),(cx+14,cy+6),(cx+6,cy+14)]
        draw.polygon(pts, fill=(255,255,255))
    return img

def simulate_screen_capture(pil_img):
    """Compose several capture-like artifacts without changing geometry."""
    img = pil_img
    if random.random() < 0.85:
        img = down_up_sample(img, 0.55, 0.95)
    if random.random() < 0.8:
        img = tiny_blur_or_unsharp(img)
    if random.random() < 0.8:
        img = jpeg_round_trip(img, q1=(28,60), q2=(35,75) if random.random()<0.5 else None)
    if random.random() < 0.8:
        img = gamma_jitter(img, 0.85, 1.20)
    if random.random() < 0.6:
        img = chroma_subsample(img, sigma=(0.8, 1.8))
    if random.random() < 0.6:
        img = add_letterbox_and_ui(img, p_bars=0.7, p_ui=0.6)
    if random.random() < 0.2:
        img = add_cursor_or_mask(img, p=1.0)
    return img

# ========================================================
# SYNTHETIC OCCLUDERS (arms, dice, tokens, etc.)
# ========================================================

def _yolo_to_xyxy(lbl, W, H):
    """Convert YOLO format label to pixel coordinates."""
    cls, cx, cy, w, h = lbl.strip().split()
    cx, cy, w, h = float(cx), float(cy), float(w), float(h)
    bw, bh = int(w * W), int(h * H)
    x1 = int((cx - w/2) * W); y1 = int((cy - h/2) * H)
    x2 = x1 + bw; y2 = y1 + bh
    return int(cls), max(0,x1), max(0,y1), min(W-1,x2), min(H-1,y2)

def _xyxy_to_yolo(cls, x1, y1, x2, y2, W, H):
    """Convert pixel coordinates to YOLO format."""
    bw = (x2 - x1) / W; bh = (y2 - y1) / H
    cx = (x1 + x2) / (2*W); cy = (y1 + y2) / (2*H)
    return f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"

def _rand_skin_or_sleeve():
    """Random skin tones or sleeve/cloth colors."""
    skin = [(207,176,149),(183,143,110),(144,105,78),(97,67,46),(230,195,170)]
    cloth= [(30,30,30),(60,60,80),(90,30,30),(30,80,60),(120,120,120),(40,40,100)]
    dice = [(200,30,30),(30,200,30),(30,30,200),(255,255,255),(20,20,20)]
    return random.choice(skin + cloth + dice)

# Chaos mode removed - now implemented in separate script: generate_chaos_mode.py

def apply_occluders(pil_img, yolo_labels, coverage_drop=0.85,
                    p_apply=0.40, occl_per_img=(1,4),
                    per_box_cov=(0.20,0.60), blur=(0.8,2.0), shadow=True):
    """
    Adds synthetic occluders (arms/sleeves/dice/tokens) on top of the composed image.
    Labels are kept unless a card is > coverage_drop occluded, then that label is removed.
    
    Args:
        pil_img: PIL Image to add occluders to
        yolo_labels: List of YOLO format label strings
        coverage_drop: Drop label if card is >this fraction occluded (default 0.85)
        p_apply: Probability to apply occluders (default 0.40 = 40%)
        occl_per_img: Tuple (min, max) number of cards to occlude
        per_box_cov: Tuple (min, max) fraction of each card to occlude
        blur: Tuple (min, max) blur radius for occluders
        shadow: Whether to add subtle shadows
    
    Returns:
        (occluded_image, filtered_labels)
    """
    if random.random() > p_apply or not yolo_labels:
        return pil_img, yolo_labels

    W, H = pil_img.size
    # Parse labels → boxes
    boxes = [_yolo_to_xyxy(lbl, W, H) for lbl in yolo_labels if len(lbl.strip().split()) == 5]
    if not boxes:
        return pil_img, yolo_labels

    # Build an occlusion mask
    occ_layer = Image.new("RGBA", (W, H), (0,0,0,0))
    draw = ImageDraw.Draw(occ_layer, "RGBA")

    # Choose targets (some boxes will be occluded)
    idxs = list(range(len(boxes)))
    random.shuffle(idxs)
    K = random.randint(occl_per_img[0], min(occl_per_img[1], len(boxes)))
    targets = idxs[:K]

    for ti in targets:
        cls, x1, y1, x2, y2 = boxes[ti]
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0: 
            continue

        # desired coverage area inside this box
        cov = random.uniform(per_box_cov[0], per_box_cov[1])
        occ_area = int(bw * bh * cov)

        # pick a primitive: rounded-rect (forearm/sleeve), ellipse (die/token), or polygon (hand/finger)
        shape = random.choices(["rrect", "ellipse", "polygon"], weights=[0.5, 0.3, 0.2])[0]

        # size the occluder to roughly match occ_area
        if shape == "ellipse":
            rw = int(math.sqrt(occ_area * random.uniform(0.8,1.2)))
            rh = int(rw * random.uniform(0.7,1.3))
        elif shape == "polygon":
            rw = int(math.sqrt(occ_area * random.uniform(0.9,1.1)))
            rh = int(rw * random.uniform(0.8,1.5))
        else:  # rrect
            rw = int(math.sqrt(occ_area * random.uniform(0.9,1.1)) * random.uniform(1.2,2.0))
            rh = max(6, int(occ_area / max(1,rw)))  # elongated
        rw = max(6, min(rw, int(bw*1.3)))
        rh = max(6, min(rh, int(bh*1.3)))

        # position: prefer edges/corners for more natural placement
        edge_bias = random.random() < 0.6
        if edge_bias:
            # Place near an edge of the card
            edge = random.choice(['top', 'bottom', 'left', 'right', 'corner'])
            if edge == 'top':
                cx = random.randint(x1, x2)
                cy = random.randint(y1 - int(0.3*bh), y1 + int(0.2*bh))
            elif edge == 'bottom':
                cx = random.randint(x1, x2)
                cy = random.randint(y2 - int(0.2*bh), y2 + int(0.3*bh))
            elif edge == 'left':
                cx = random.randint(x1 - int(0.3*bw), x1 + int(0.2*bw))
                cy = random.randint(y1, y2)
            elif edge == 'right':
                cx = random.randint(x2 - int(0.2*bw), x2 + int(0.3*bw))
                cy = random.randint(y1, y2)
            else:  # corner
                cx = random.choice([x1 - int(0.2*bw), x2 + int(0.2*bw)])
                cy = random.choice([y1 - int(0.2*bh), y2 + int(0.2*bh)])
        else:
            # Random placement with some spill outside
            cx = random.randint(x1 - int(0.15*bw), x2 + int(0.15*bw))
            cy = random.randint(y1 - int(0.15*bh), y2 + int(0.15*bh))
        
        ox1 = max(0, cx - rw//2); oy1 = max(0, cy - rh//2)
        ox2 = min(W-1, ox1 + rw); oy2 = min(H-1, oy1 + rh)
        
        # Ensure valid coordinates after clamping
        if ox2 <= ox1: ox2 = ox1 + 1
        if oy2 <= oy1: oy2 = oy1 + 1

        col = _rand_skin_or_sleeve()
        # Vary opacity: some translucent, most opaque
        alpha = random.randint(140, 255) if random.random() < 0.3 else random.randint(200, 255)

        # optional tiny shadow for realism
        if shadow and random.random() < 0.8:
            sh = Image.new("RGBA",(W,H),(0,0,0,0))
            sd = ImageDraw.Draw(sh, "RGBA")
            s_off = random.randint(2,5)
            if shape == "ellipse":
                sd.ellipse([ox1+s_off,oy1+s_off,ox2+s_off,oy2+s_off],
                          fill=(0,0,0,random.randint(40,80)))
            elif shape == "polygon":
                # Simple quad for shadow
                pts = [(ox1+s_off,oy1+s_off),(ox2+s_off,oy1+s_off),
                       (ox2+s_off,oy2+s_off),(ox1+s_off,oy2+s_off)]
                sd.polygon(pts, fill=(0,0,0,random.randint(40,80)))
            else:  # rrect
                r = max(6, int(min(rw, rh) * 0.3))
                sd.rounded_rectangle([ox1+s_off,oy1+s_off,ox2+s_off,oy2+s_off],
                                    radius=r, fill=(0,0,0,random.randint(40,80)))
            sh = sh.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.0,2.5)))
            occ_layer = Image.alpha_composite(occ_layer, sh)

        # Draw the occluder with optional rotation for realism
        if shape == "ellipse":
            draw.ellipse([ox1,oy1,ox2,oy2], fill=(*col, alpha))
        elif shape == "polygon":
            # Irregular quad (hand/finger-ish)
            jitter = int(min(rw, rh) * 0.15)
            pts = [
                (ox1 + random.randint(-jitter, jitter), oy1 + random.randint(-jitter, jitter)),
                (ox2 + random.randint(-jitter, jitter), oy1 + random.randint(-jitter, jitter)),
                (ox2 + random.randint(-jitter, jitter), oy2 + random.randint(-jitter, jitter)),
                (ox1 + random.randint(-jitter, jitter), oy2 + random.randint(-jitter, jitter))
            ]
            draw.polygon(pts, fill=(*col, alpha))
        else:  # rrect with optional rotation
            r = max(6, int(min(rw, rh) * 0.3))  # corner radius
            # For rotation, create a temp layer and rotate it
            if random.random() < 0.6:  # 60% get rotated for arm-like angles
                temp = Image.new("RGBA", (rw+20, rh+20), (0,0,0,0))
                td = ImageDraw.Draw(temp, "RGBA")
                td.rounded_rectangle([10,10,10+rw,10+rh], radius=r, fill=(*col, alpha))
                angle = random.randint(-45, 45)
                temp = temp.rotate(angle, expand=False, fillcolor=(0,0,0,0))
                occ_layer.paste(temp, (ox1-10, oy1-10), temp)
            else:
                draw.rounded_rectangle([ox1,oy1,ox2,oy2], radius=r, fill=(*col, alpha))

    # soften occluders a touch to blend
    occ_layer = occ_layer.filter(ImageFilter.GaussianBlur(radius=random.uniform(*blur)))
    out = Image.alpha_composite(pil_img.convert("RGBA"), occ_layer).convert("RGB")

    # Compute coverage per box to optionally drop labels that are > coverage_drop occluded
    occ_mask = np.array(occ_layer.split()[-1]) > 0
    keep_labels = []
    for lbl, b in zip(yolo_labels, boxes):
        cls, x1, y1, x2, y2 = b
        x1c, y1c, x2c, y2c = max(0,x1), max(0,y1), min(W-1,x2), min(H-1,y2)
        if x2c <= x1c or y2c <= y1c:
            continue
        card_area = (x2c-x1c) * (y2c-y1c)
        occ_area_px = int(occ_mask[y1c:y2c, x1c:x2c].sum())
        cov_ratio = occ_area_px / max(1, card_area)
        if cov_ratio < coverage_drop:
            keep_labels.append(lbl)  # keep supervision
        # else drop label (too occluded to learn from)
    return out, keep_labels

# ========================================================

# Build separate template pairs for each split
template_pairs_by_split = {'train': [], 'valid': [], 'test': []}
for split in ['train', 'valid', 'test']:
    img_dir = os.path.join(TEMPLATE_BASE, split, 'images')
    label_dir = os.path.join(TEMPLATE_BASE, split, 'labels')
    images = glob.glob(os.path.join(img_dir, '*.png')) + glob.glob(os.path.join(img_dir, '*.jpg'))
    for img_path in images:
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, base + '.txt')
        if os.path.exists(label_path):
            template_pairs_by_split[split].append((img_path, label_path))

# Utilities for handling card sets and persistent class index

def load_card_files(card_dirs):
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
    Set codes (e.g., WTR173, WTR174, WTR175) distinguish different rarity variants
    of the same card, which have different popularity weights.
    
    Previously this function stripped set codes to collapse reprints, but this
    caused red/yellow/blue variants to be treated as identical.
    
    Examples (now preserved):
      'Enlightened_Strike_WTR159' -> 'Enlightened_Strike_WTR159'
      'Sigil_of_Solace_WTR173' -> 'Sigil_of_Solace_WTR173'
    """
    return name


def load_popularity_weights(weights_path):
    """
    Load popularity weights from JSON file.
    Returns:
        - weights_dict: dict mapping canonicalized card names to their total weights
        - rank_dict: dict mapping canonicalized card names to their popularity rank (1-based)
    """
    if not os.path.exists(weights_path):
        print(f"[warning] Popularity weights file not found: {weights_path}")
        return {}, {}
    
    try:
        with open(weights_path, 'r') as f:
            data = json.load(f)
        
        raw_weights = data.get('weights', {})
        
        # Build dict of canonicalized names -> total weights
        weights_dict = {}
        for card_key, weight_data in raw_weights.items():
            canon_name = canonicalize_name(card_key)
            total_weight = weight_data.get('total_weight', 0.0)
            
            # Accumulate weights for cards with multiple printings
            if canon_name in weights_dict:
                weights_dict[canon_name] += total_weight
            else:
                weights_dict[canon_name] = total_weight
        
        # Sort by weight descending to create rank mapping
        sorted_cards = sorted(weights_dict.items(), key=lambda x: x[1], reverse=True)
        rank_dict = {card: rank + 1 for rank, (card, _) in enumerate(sorted_cards)}
        
        print(f"[init] Loaded popularity weights for {len(weights_dict)} cards (ranked 1-{len(rank_dict)})")
        return weights_dict, rank_dict
    
    except Exception as e:
        print(f"[error] Failed to load popularity weights: {e}")
        return {}, {}


def filter_cards_by_popularity_rank(card_files, rank_dict, min_rank=None, max_rank=None):
    """
    Filter card files to only include those within the specified popularity rank range.
    
    Args:
        card_files: list of (dir, filename) tuples
        rank_dict: dict mapping canonicalized card names to ranks
        min_rank: minimum rank (inclusive), e.g., 1 for top cards
        max_rank: maximum rank (inclusive), e.g., 300 for top 300
    
    Returns:
        filtered list of (dir, filename) tuples
    """
    if not rank_dict or (min_rank is None and max_rank is None):
        return card_files  # No filtering
    
    filtered = []
    for set_dir, card_filename in card_files:
        raw_name = os.path.splitext(card_filename)[0]
        canon_name = canonicalize_name(raw_name)
        
        rank = rank_dict.get(canon_name)
        if rank is None:
            # Card not in popularity data - skip by default when filtering
            continue
        
        # Check if rank is within range
        if min_rank is not None and rank < min_rank:
            continue
        if max_rank is not None and rank > max_rank:
            continue
        
        filtered.append((set_dir, card_filename))
    
    print(f"[init] Filtered {len(card_files)} cards to {len(filtered)} within rank range [{min_rank or 1}, {max_rank or 'end'}]")
    return filtered


def weighted_card_choice(card_files, weights_dict, class_to_files=None, target_class=None):
    """
    Choose a card file using popularity weights with independent sampling.
    Cards can be selected multiple times (realistic for gameplay scenarios).
    
    Args:
        card_files: list of (dir, filename) tuples to choose from
        weights_dict: dict mapping canonicalized card names to weights
        class_to_files: optional dict for coverage-guided selection
        target_class: optional specific class to select from (for coverage-guided)
    
    Returns:
        (set_dir, card_filename) tuple
    """
    if target_class and class_to_files:
        # Coverage-guided: choose from specific class
        candidates = class_to_files.get(target_class, [])
        if not candidates:
            candidates = card_files
    else:
        candidates = card_files
    
    if not candidates:
        raise RuntimeError("No card files available for selection")
    
    # If no weights available, fall back to uniform random
    if not weights_dict:
        return random.choice(candidates)
    
    # Build weights list for candidates
    candidate_weights = []
    for set_dir, card_filename in candidates:
        raw_name = os.path.splitext(card_filename)[0]
        canon_name = canonicalize_name(raw_name)
        weight = weights_dict.get(canon_name, 0.0)
        
        # Use a minimum weight to ensure all cards have some chance
        weight = max(weight, 0.001)
        candidate_weights.append(weight)
    
    # Normalize weights to probabilities
    total_weight = sum(candidate_weights)
    if total_weight <= 0:
        return random.choice(candidates)
    
    probabilities = [w / total_weight for w in candidate_weights]
    
    # Sample using weights (independent for each call)
    chosen_idx = random.choices(range(len(candidates)), weights=probabilities, k=1)[0]
    return candidates[chosen_idx]


def load_or_create_class_index(index_path):
    if os.path.exists(index_path):
        with open(index_path, 'r') as f:
            data = yaml.safe_load(f) or {}
        raw_names = data.get('names', [])
        if not isinstance(raw_names, list):
            raw_names = []
        # Canonicalize existing names and dedupe preserving order (first wins)
        names = []
        seen = set()
        for n in raw_names:
            cn = canonicalize_name(n)
            if cn not in seen:
                names.append(cn)
                seen.add(cn)
    else:
        names = []
    return names


def save_class_index(index_path, names):
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, 'w') as f:
        yaml.dump({'names': names}, f, default_flow_style=False)


# ===================== POSITIONING CACHE SYSTEM =====================

def get_positioning_cache_path(template_path, cache_dir):
    """
    Generate a cache file path for a template image.
    Cache files are stored with the same name as the template but in the cache directory.
    """
    template_name = os.path.basename(template_path)
    cache_name = os.path.splitext(template_name)[0] + '_positioning.json'
    return os.path.join(cache_dir, cache_name)


def load_positioning_cache(cache_path):
    """
    Load cached positioning data if it exists.
    Returns None if cache doesn't exist or is invalid.
    """
    if not os.path.exists(cache_path):
        return None
    
    try:
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
        return cache_data
    except Exception as e:
        print(f"[warning] Failed to load positioning cache {cache_path}: {e}")
        return None


def save_positioning_cache(cache_path, positioning_data):
    """
    Save positioning data to cache file.
    positioning_data format:
    {
        'template_size': [width, height],
        'ref_bbox_stats': {
            'avg_bbox_width': float,
            'avg_bbox_height': float,
            'avg_area': float,
            'highlighted_card_threshold': float
        },
        'boxes': [
            {
                'class_id': int,
                'bbox': [cx, cy, bw, bh],  # normalized coordinates
                'best_angle': int,  # 0-359 degrees
                'crop_corner': str,  # 'tl', 'tr', 'bl', 'br', 'center'
                'visible_area': int  # pixels
            },
            ...
        ]
    }
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    try:
        with open(cache_path, 'w') as f:
            json.dump(positioning_data, f, indent=2)
    except Exception as e:
        print(f"[warning] Failed to save positioning cache {cache_path}: {e}")


def build_positioning_cache(template_img, label_lines, avg_bbox_width, avg_bbox_height, avg_area, highlighted_threshold):
    """
    Build complete positioning cache for a template by calculating optimal rotations.
    Uses a dummy card image to determine rotations (geometry-dependent, not card-dependent).
    
    Returns cache data structure with all box-specific positioning info.
    """
    W, H = template_img.size
    
    # Create a dummy card image with proper FaB card aspect ratio (~63mm x ~88mm ≈ 0.714 aspect)
    # Use a standard card size that will be scaled appropriately
    CARD_ASPECT = 0.714  # width / height for FaB cards
    dummy_h = 400  # Standard height in pixels
    dummy_w = int(dummy_h * CARD_ASPECT)
    dummy_card = Image.new('RGBA', (dummy_w, dummy_h), (255, 255, 255, 255))
    
    # Calculate aspect ratio stats for this template
    aspect_ratios = []
    for line in label_lines:
        parts = line.strip().split()
        if len(parts) == 5:
            _, cx, cy, bw, bh = parts
            bw, bh = float(bw), float(bh)
            box_w_px = bw * W
            box_h_px = bh * H
            if box_w_px > 0:
                aspect_ratios.append(box_h_px / box_w_px)
    
    aspect_ratios_sorted = sorted(aspect_ratios)
    n = len(aspect_ratios_sorted)
    lower = int(n * 0.25)
    upper = int(n * 0.75)
    trimmed = aspect_ratios_sorted[lower:upper] if upper > lower else aspect_ratios_sorted
    ASPECT_SCALE = (1920.0 / 1080.0) / (640.0 / 640.0)  # = 1.778
    avg_aspect = (sum(trimmed) / len(trimmed) if trimmed else 1.0) * ASPECT_SCALE
    high_threshold = 1.5 * ASPECT_SCALE
    high_aspects = [ar * ASPECT_SCALE for ar in aspect_ratios if ar >= 1.5]
    high_avg_aspect = sum(high_aspects) / len(high_aspects) if high_aspects else avg_aspect * 2.4
    
    cache_data = {
        'template_name': os.path.basename(template_img.filename) if hasattr(template_img, 'filename') else 'unknown',
        'template_size': [W, H],
        'ref_stats': {
            'avg_bbox_width': float(avg_bbox_width),
            'avg_bbox_height': float(avg_bbox_height),
            'avg_area': float(avg_area),
            'highlighted_card_threshold': float(highlighted_threshold)
        },
        'aspect_stats': {
            'avg_aspect': float(avg_aspect),
            'high_avg_aspect': float(high_avg_aspect)
        },
        'boxes': []
    }
    
    # Calculate optimal rotation for each box
    print(f"[cache] Building positioning cache for {len(label_lines)} boxes...")
    for idx, line in enumerate(label_lines):
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        
        class_id_str, cx, cy, bw, bh = parts
        class_id = int(class_id_str)
        cx, cy, bw, bh = float(cx), float(cy), float(bw), float(bh)
        
        box_w = int(bw * W)
        box_h = int(bh * H)
        box_x = int((cx - bw/2) * W)
        box_y = int((cy - bh/2) * H)
        
        # Determine if this is a highlighted card
        bbox_area = bw * bh
        is_highlighted = highlighted_threshold and bbox_area > highlighted_threshold
        
        # Calculate optimal rotation using paste_card's rotation search
        # We use the dummy card and extract the best_angle by running the rotation logic
        # For efficiency, we'll call paste_card but only care about the rotation
        temp_img = template_img.copy()
        
        # Use paste_card to calculate best rotation (it will return the rotated result)
        # but we just need to extract the angle it chose
        # Actually, let's inline the rotation search here for clarity
        
        bbox_data = [cx, cy, bw, bh]
        
        # Calculate rotation using the same logic as paste_card
        # (This duplicates some code but keeps the cache building self-contained)
        import numpy as np
        
        # Scale dummy card based on whether it's highlighted
        orig_w, orig_h = dummy_card.size
        if is_highlighted:
            scale_w = box_w / orig_w
            scale_h = box_h / orig_h
            card_scale = min(scale_w, scale_h)
        else:
            scale_w = avg_bbox_width / orig_w
            scale_h = avg_bbox_height / orig_h
            card_scale = (scale_w + scale_h) / 2.0
        
        # Apply 3.5% reduction
        card_scale = card_scale * 0.965
        
        scaled_w = int(orig_w * card_scale)
        scaled_h = int(orig_h * card_scale)
        card_scaled = dummy_card.resize((scaled_w, scaled_h), Image.LANCZOS)
        
        # NEW SIMPLIFIED ROTATION ALGORITHM (same as paste_card)
        card_arr = np.array(card_scaled)
        if len(card_arr.shape) == 3 and card_arr.shape[2] == 4:
            total_card_pixels = np.sum(card_arr[:, :, 3] > 10)
        else:
            total_card_pixels = scaled_w * scaled_h
        
        # STEP 1: Find angle with maximum visible surface area
        best_angle_area = 0
        best_visible_area = 0
        
        for angle in range(0, 360, 1):  # Test every 1 degree
            test_card = card_scaled.rotate(-angle, expand=True, resample=Image.BICUBIC, fillcolor=(0, 0, 0, 0))
            rot_w, rot_h = test_card.size
            
            overlap_w = min(rot_w, box_w)
            overlap_h = min(rot_h, box_h)
            overlap_region = test_card.crop((0, 0, overlap_w, overlap_h))
            test_arr = np.array(overlap_region)
            
            if len(test_arr.shape) == 3 and test_arr.shape[2] == 4:
                visible_pixels = np.sum(test_arr[:, :, 3] > 10)
            else:
                visible_pixels = overlap_w * overlap_h
            
            if visible_pixels > best_visible_area:
                best_visible_area = visible_pixels
                best_angle_area = angle
        
        # STEP 2: Test vertically mirrored angle
        mirrored_angle = (180 - best_angle_area) % 360
        
        def count_bright_pixels_cache(angle):
            test_card = card_scaled.rotate(-angle, expand=True, resample=Image.BICUBIC, fillcolor=(0, 0, 0, 0))
            rot_w, rot_h = test_card.size
            overlap_w = min(rot_w, box_w)
            overlap_h = min(rot_h, box_h)
            overlap_region = test_card.crop((0, 0, overlap_w, overlap_h))
            
            if overlap_region.mode == 'RGBA':
                arr = np.array(overlap_region)
                rgb = arr[:, :, :3]
                alpha = arr[:, :, 3]
                brightness = np.mean(rgb, axis=2)
                bright_pixels = np.sum((brightness > 128) & (alpha > 10))
            else:
                gray = overlap_region.convert('L')
                arr = np.array(gray)
                bright_pixels = np.sum(arr > 128)
            
            return bright_pixels
        
        bright_original = count_bright_pixels_cache(best_angle_area)
        bright_mirrored = count_bright_pixels_cache(mirrored_angle)
        
        if bright_mirrored > bright_original:
            best_angle = mirrored_angle
        else:
            best_angle = best_angle_area
        
        # STEP 3: If right half of screen, rotate 180° more
        img_center_x = W / 2
        if box_x + box_w/2 >= img_center_x:
            best_angle = (best_angle + 180) % 360
        
        cache_data['boxes'].append({
            'index': idx,
            'class_id': class_id,
            'bbox': [cx, cy, bw, bh],
            'is_highlighted': bool(is_highlighted),
            'best_rotation': int(best_angle),
            'pixel_coords': [box_x, box_y, box_w, box_h]
        })
    
    print(f"[cache] Positioning cache built successfully")
    return cache_data


# =====================================================================


def best_fit_to_bbox(card_img, box_w, box_h, avg_area=None):
    # Crop for aspect ratio < 0.75, or for horizontal cards with high aspect ratio and small area
    # Native resolution - use original thresholds without adjustment
    aspect_ratio = box_h / box_w if box_w > 0 else 1.0
    area = box_w * box_h
    crop_condition = aspect_ratio < 0.75
    if avg_area is not None and aspect_ratio > 1.5 and area < 0.65 * avg_area:
        crop_condition = True
    if crop_condition:
        rw, rh = card_img.size
        # Instead of center-crop, randomly choose edge(s) to show (like stacked cards in play)
        # Randomly decide whether to crop horizontally or vertically based on aspect mismatch
        crop_horizontal = rw > box_w
        crop_vertical = rh > box_h
        
        if crop_horizontal and crop_vertical:
            # Need to crop both dimensions - pick edge alignment
            h_align = random.choice(['left', 'right', 'center'])
            v_align = random.choice(['top', 'bottom', 'center'])
        elif crop_horizontal:
            h_align = random.choice(['left', 'right', 'center'])
            v_align = 'center'
        elif crop_vertical:
            h_align = 'center'
            v_align = random.choice(['top', 'bottom', 'center'])
        else:
            h_align = v_align = 'center'
        
        # Calculate crop positions based on alignment
        if h_align == 'left':
            left = 0
        elif h_align == 'right':
            left = max(0, rw - box_w)
        else:  # center
            left = max(0, (rw - box_w) // 2)
        
        if v_align == 'top':
            upper = 0
        elif v_align == 'bottom':
            upper = max(0, rh - box_h)
        else:  # center
            upper = max(0, (rh - box_h) // 2)
        
        right = min(rw, left + box_w)
        lower = min(rh, upper + box_h)
        card_cropped = card_img.crop((left, upper, right, lower))
        return card_cropped
    else:
        return card_img.resize((box_w, box_h), Image.LANCZOS)
def guess_rotation_direction(template_img, box_x, box_y, box_w, box_h):
    # Crop bounding box region from template
    bbox_crop = template_img.crop((box_x, box_y, box_x + box_w, box_y + box_h)).convert('L')
    w, h = bbox_crop.size
    left_half = bbox_crop.crop((0, 0, w // 2, h))
    right_half = bbox_crop.crop((w // 2, 0, w, h))
    left_brightness = sum(left_half.getdata()) / left_half.size[0] / left_half.size[1]
    right_brightness = sum(right_half.getdata()) / right_half.size[0] / right_half.size[1]
    # If left is brighter, rotate one way; else, the other
    return 1 if left_brightness > right_brightness else -1

def paste_card(template_img, card_img, bbox, avg_aspect, high_avg_aspect, avg_area, target_bbox_width, target_bbox_height, highlighted_threshold, cached_rotation=None, skip_bg_tint=False):
    """
    Place card in bbox with uniform scale across all cards:
    1. Detect if this is a "highlighted" card (much larger bbox)
    2. For normal cards: use uniform scale across scene
    3. For highlighted cards: scale independently to fit bbox
    4. Test all 360 rotation angles to maximize visible card area (or use cached_rotation if provided)
    5. Crop to bbox if card extends beyond
    6. Apply background color tint and other post-processing
    
    Args:
        cached_rotation: Optional pre-calculated optimal rotation angle (0-359). If provided, skips rotation search.
        skip_bg_tint: If True, skips background color sampling/tinting (useful for chaos mode with synthetic backgrounds)
    """
    from PIL import ImageFilter, ImageEnhance
    import numpy as np
    
    w, h = template_img.size
    cx, cy, bw, bh = [float(x) for x in bbox]
    box_w, box_h = int(bw * w), int(bh * h)
    box_x, box_y = int((cx - bw/2) * w), int((cy - bh/2) * h)
    
    # Get original card dimensions
    orig_w, orig_h = card_img.size
    
    # Check if this is a "highlighted" card (blown up for broadcast overlay)
    bbox_area = bw * bh
    is_highlighted = highlighted_threshold and bbox_area > highlighted_threshold
    
    if is_highlighted:
        # Highlighted card: scale to fit its bbox independently (maintain aspect ratio)
        scale_w = box_w / orig_w
        scale_h = box_h / orig_h
        card_scale = min(scale_w, scale_h)  # Fit within bbox without stretching
    else:
        # Normal card: use uniform scale based on target bbox dimensions
        # Use average of width/height scales to balance filling the bbox
        scale_w = target_bbox_width / orig_w
        scale_h = target_bbox_height / orig_h
        card_scale = (scale_w + scale_h) / 2.0  # Average scale - balances both dimensions
    
    # Apply 3.5% reduction to prevent cards from being slightly too large
    card_scale = card_scale * 0.965
    
    # Apply uniform scale to card (same for all cards in scene)
    scaled_w = int(orig_w * card_scale)
    scaled_h = int(orig_h * card_scale)
    card_scaled = card_img.resize((scaled_w, scaled_h), Image.LANCZOS)
    
    # Use cached rotation if available, otherwise calculate optimal rotation
    if cached_rotation is not None:
        best_angle = cached_rotation
    else:
        # NEW SIMPLIFIED ROTATION ALGORITHM
        # Step 1: Find angle that maximizes visible surface area (test every degree, 0-359)
        # Step 2: Test mirrored angle and choose based on bright pixel coverage
        # Step 3: If right half of screen, rotate 180° more
        
        card_arr = np.array(card_scaled)
        if len(card_arr.shape) == 3 and card_arr.shape[2] == 4:
            total_card_pixels = np.sum(card_arr[:, :, 3] > 10)
        else:
            total_card_pixels = scaled_w * scaled_h
        
        # STEP 1: Find angle with maximum visible surface area
        best_angle_area = 0
        best_visible_area = 0
        
        for angle in range(0, 360, 1):  # Test every 1 degree
            test_card = card_scaled.rotate(-angle, expand=True, resample=Image.BICUBIC, fillcolor=(0, 0, 0, 0))
            rot_w, rot_h = test_card.size
            
            # Calculate visible area (intersection with bbox)
            overlap_w = min(rot_w, box_w)
            overlap_h = min(rot_h, box_h)
            overlap_region = test_card.crop((0, 0, overlap_w, overlap_h))
            test_arr = np.array(overlap_region)
            
            if len(test_arr.shape) == 3 and test_arr.shape[2] == 4:
                visible_pixels = np.sum(test_arr[:, :, 3] > 10)
            else:
                visible_pixels = overlap_w * overlap_h
            
            if visible_pixels > best_visible_area:
                best_visible_area = visible_pixels
                best_angle_area = angle
        
        # STEP 2: Test vertically mirrored angle and choose based on bright pixels
        # Mirror formula: if angle is X, mirror is 180 - X (for vertical mirror across horizontal axis)
        # But we want to maintain orientation, so we test small deviations around the found angle
        mirrored_angle = (180 - best_angle_area) % 360
        
        # Test both angles and count bright pixels in the overlap region
        def count_bright_pixels(angle):
            test_card = card_scaled.rotate(-angle, expand=True, resample=Image.BICUBIC, fillcolor=(0, 0, 0, 0))
            rot_w, rot_h = test_card.size
            overlap_w = min(rot_w, box_w)
            overlap_h = min(rot_h, box_h)
            overlap_region = test_card.crop((0, 0, overlap_w, overlap_h))
            
            # Convert to grayscale to measure brightness
            if overlap_region.mode == 'RGBA':
                # Only consider non-transparent pixels
                arr = np.array(overlap_region)
                rgb = arr[:, :, :3]
                alpha = arr[:, :, 3]
                brightness = np.mean(rgb, axis=2)
                # Count bright pixels (>128) that are not transparent
                bright_pixels = np.sum((brightness > 128) & (alpha > 10))
            else:
                gray = overlap_region.convert('L')
                arr = np.array(gray)
                bright_pixels = np.sum(arr > 128)
            
            return bright_pixels
        
        bright_original = count_bright_pixels(best_angle_area)
        bright_mirrored = count_bright_pixels(mirrored_angle)
        
        # Choose the angle with more bright pixels
        if bright_mirrored > bright_original:
            best_angle = mirrored_angle
        else:
            best_angle = best_angle_area
        
        # STEP 3: If right half of screen, rotate 180° more
        img_center_x = w / 2
        if box_x + box_w/2 >= img_center_x:
            best_angle = (best_angle + 180) % 360
    
    # Apply best rotation WITH expansion
    if best_angle != 0:
        card_rotated = card_scaled.rotate(-best_angle, expand=True, resample=Image.BICUBIC, fillcolor=(0, 0, 0, 0))
    else:
        card_rotated = card_scaled
    
    # Crop or center card in bbox WITH RANDOM OFFSET for variety
    card_w, card_h = card_rotated.size
    
    # ALWAYS apply random alignment to simulate partial occlusion from all directions
    # Choose random alignment independently for X and Y axes
    h_align = random.choice(['left', 'center', 'right'])
    v_align = random.choice(['top', 'center', 'bottom'])
    
    # Calculate the "ideal" position based on alignment
    if h_align == 'left':
        ideal_x = 0
    elif h_align == 'right':
        ideal_x = max(0, card_w - box_w)
    else:  # center
        ideal_x = max(0, (card_w - box_w) // 2)
    
    if v_align == 'top':
        ideal_y = 0
    elif v_align == 'bottom':
        ideal_y = max(0, card_h - box_h)
    else:  # center
        ideal_y = max(0, (card_h - box_h) // 2)
    
    # Apply random jitter to create more variation (±10% of card size)
    jitter_x = int(random.uniform(-0.1, 0.1) * card_w)
    jitter_y = int(random.uniform(-0.1, 0.1) * card_h)
    
    crop_x = max(0, min(card_w - box_w, ideal_x + jitter_x)) if card_w > box_w else 0
    crop_y = max(0, min(card_h - box_h, ideal_y + jitter_y)) if card_h > box_h else 0
    
    if card_w > box_w or card_h > box_h:
        # Card is larger - crop with randomized position
        card_final = card_rotated.crop((crop_x, crop_y, crop_x + box_w, crop_y + box_h))
    else:
        # Card is smaller - but still apply offset instead of perfect centering
        card_final = Image.new('RGBA', (box_w, box_h), (0, 0, 0, 0))
        
        # Add random offset to paste position (±20% of available space)
        available_x = box_w - card_w
        available_y = box_h - card_h
        
        base_x = available_x // 2  # Start with center
        base_y = available_y // 2
        
        offset_x = int(random.uniform(-0.4, 0.4) * available_x) if available_x > 0 else 0
        offset_y = int(random.uniform(-0.4, 0.4) * available_y) if available_y > 0 else 0
        
        paste_x = max(0, min(available_x, base_x + offset_x))
        paste_y = max(0, min(available_y, base_y + offset_y))
        card_final.paste(card_rotated, (paste_x, paste_y), card_rotated)
    
    # Post-processing: sample background properties and apply to card
    # Skip background tinting in chaos mode (synthetic backgrounds cause extreme tints)
    if not skip_bg_tint:
        # Sample bounding box region from template
        bbox_crop = template_img.crop((box_x, box_y, box_x + box_w, box_y + box_h)).convert('RGB')
        # Calculate average brightness
        arr = np.array(bbox_crop)
        avg_brightness = np.mean(arr)
        # Calculate average color for tint overlay
        avg_color = tuple(np.mean(arr, axis=(0, 1)).astype(int))
        
        # Estimate blur by variance of Laplacian (simple proxy)
        gray = np.mean(arr, axis=2)
        laplacian = np.abs(np.gradient(gray)[0]) + np.abs(np.gradient(gray)[1])
        blur_level = max(0.5, min(2.5, 2.5 - np.var(laplacian) / 50))  # scale to [0.5, 2.5]
        
        # Apply blur to card
        card_post = card_final.filter(ImageFilter.GaussianBlur(radius=blur_level))
        # Match brightness
        card_post = ImageEnhance.Brightness(card_post).enhance(avg_brightness / 128)
        
        # Apply translucent color tint from background, preserving alpha
        from PIL import Image as PILImage
        tint_layer = PILImage.new('RGB', card_post.size, avg_color)
        card_rgb = card_post.convert('RGB')
        card_tinted = PILImage.blend(card_rgb, tint_layer, alpha=0.15)
        # Restore alpha channel
        card_post_final = PILImage.new('RGBA', card_post.size)
        card_post_final.paste(card_tinted, (0, 0))
        if card_post.mode == 'RGBA':
            card_post_final.putalpha(card_post.split()[3])  # Copy original alpha
    else:
        # Chaos mode: minimal processing - just slight blur
        card_post_final = card_final.filter(ImageFilter.GaussianBlur(radius=0.8))
    
    # Paste processed card at bbox location with alpha transparency
    template_img.paste(card_post_final, (box_x, box_y), card_post_final)
    return template_img

def refined_rotation(ratio, high_avg_aspect):
    # Native resolution - no aspect adjustment needed
    vertical_angles = [0, 180]
    default_angles = [90, 270]
    if ratio >= high_avg_aspect:
        return random.choice(vertical_angles)
    elif ratio < 1.5:
        return random.choice(default_angles)
    else:
        interp = (ratio - 1.5) / (high_avg_aspect - 1.5)
        if random.random() < 0.5:
            return int(90 * (1 - interp) + 0 * interp)
        else:
            return int(270 * (1 - interp) + 180 * interp)


# Parse command-line arguments
parser = argparse.ArgumentParser(description='Generate synthetic FaB card images.')
parser.add_argument('--num-images', type=int, default=None, help='Number of synthetic images to generate')
parser.add_argument('--card-dirs', nargs='*', default=None, help='One or more directories containing card PNG files')
parser.add_argument('--coverage-guided', action='store_true', help='Select cards prioritizing underrepresented classes per split')
parser.add_argument('--coverage-file', type=str, default=None, help='Path to coverage tracking YAML (defaults to <OUTPUT_BASE_DIR>/coverage.yaml)')
parser.add_argument('--seed', type=int, default=None, help='Optional RNG seed for reproducibility')
parser.add_argument('--makeup-min', type=int, default=0, help='If >0, ensure at least this many labels per class in the specified split (best effort within num-images)')
parser.add_argument('--makeup-split', type=str, choices=['train','valid','test'], default=None, help='Split to apply makeup-min target to; defaults to the split being generated each iteration')
parser.add_argument('--popularity-min', type=int, default=None, help='Minimum popularity rank (inclusive) - e.g., 1 for top cards, 301 for cards after top 300')
parser.add_argument('--popularity-max', type=int, default=None, help='Maximum popularity rank (inclusive) - e.g., 300 for top 300, 600 for top 600')
parser.add_argument('--use-popularity-weights', action='store_true', help='Use popularity weights for weighted card sampling (biases towards more popular cards)')
parser.add_argument('--popularity-weights-path', type=str, default=None, help='Path to card popularity weights JSON file')
# Chaos mode removed - see generate_chaos_mode.py for synthetic background generation
args = parser.parse_args()

# Number of synthetic images to generate
NUM_SYNTHETIC_IMAGES = args.num_images if args.num_images is not None else NUM_SYNTHETIC_IMAGES

# Override card set directories from CLI if provided
if args.card_dirs:
    CARD_SET_DIRS = args.card_dirs

# Load popularity weights if available
popularity_weights_path = args.popularity_weights_path or POPULARITY_WEIGHTS_PATH
weights_dict, rank_dict = load_popularity_weights(popularity_weights_path)

# Validate popularity rank arguments
if args.popularity_min is not None and args.popularity_max is not None:
    if args.popularity_min > args.popularity_max:
        raise ValueError(f"--popularity-min ({args.popularity_min}) cannot be greater than --popularity-max ({args.popularity_max})")
    if args.popularity_min < 1:
        raise ValueError(f"--popularity-min must be at least 1")

# Load card files for this run
card_files = load_card_files(CARD_SET_DIRS)

# Filter cards by popularity rank if specified
if args.popularity_min is not None or args.popularity_max is not None:
    card_files = filter_cards_by_popularity_rank(card_files, rank_dict, args.popularity_min, args.popularity_max)
    if not card_files:
        raise RuntimeError(f"No cards found within popularity rank range [{args.popularity_min or 1}, {args.popularity_max or 'end'}]")

# Load and update persistent class index with any new card names from this run
class_index_path = os.path.join(OUTPUT_BASE_DIR, 'classes.yaml')
data_yaml_path = os.path.join(OUTPUT_BASE_DIR, 'data.yaml')
class_names = load_or_create_class_index(class_index_path)
existing = set(class_names)
new_classes_found = False
for set_dir, card_filename in card_files:
    raw_name = os.path.splitext(card_filename)[0]
    name = canonicalize_name(raw_name)
    if name not in existing:
        class_names.append(name)
        existing.add(name)
        new_classes_found = True

# If we found new classes, immediately save to prevent loss on crash
if new_classes_found:
    print(f"[init] Found new classes, total now: {len(class_names)}. Saving immediately...")
    save_class_index(class_index_path, class_names)
    # Also write data.yaml immediately
    data_yaml_content = {
        'train': os.path.join(OUTPUT_BASE_DIR, 'train', 'images').replace('\\', '/'),
        'val': os.path.join(OUTPUT_BASE_DIR, 'valid', 'images').replace('\\', '/'),
        'test': os.path.join(OUTPUT_BASE_DIR, 'test', 'images').replace('\\', '/'),
        'nc': len(class_names),
        'names': class_names,
    }
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_yaml_content, f, default_flow_style=False)
    print(f"[init] Saved classes.yaml and data.yaml with {len(class_names)} classes")

# Build mapping name -> id
name_to_id = {name: idx for idx, name in enumerate(class_names)}

# Build mapping from class name -> list of available image files (set_dir, filename)
class_to_files = {}
for set_dir, card_filename in card_files:
    raw = os.path.splitext(card_filename)[0]
    cname = canonicalize_name(raw)
    class_to_files.setdefault(cname, []).append((set_dir, card_filename))

# Optional reproducibility
if args.seed is not None:
    random.seed(args.seed)

"""
Dynamic split selection helpers: keep overall dataset close to 70/20/10 across runs.
We choose the split with the lowest progress towards its target ratio based on current
image counts on disk, biased by template availability per split.
"""

TARGET_RATIOS = {'train': 0.7, 'valid': 0.2, 'test': 0.1}

def count_images_in_split(base_dir, split):
    img_dir = os.path.join(base_dir, split, 'images')
    if not os.path.isdir(img_dir):
        return 0
    cnt = 0
    try:
        for f in os.listdir(img_dir):
            fl = f.lower()
            if fl.endswith('.png') or fl.endswith('.jpg') or fl.endswith('.jpeg'):
                cnt += 1
    except FileNotFoundError:
        return 0
    return cnt

def get_existing_counts(base_dir):
    return {
        'train': count_images_in_split(base_dir, 'train'),
        'valid': count_images_in_split(base_dir, 'valid'),
        'test': count_images_in_split(base_dir, 'test'),
    }

def choose_split(current_counts, ratios, templates_available):
    # Compute progress score = current_count / ratio; pick the lowest score
    # Shuffle order to avoid always picking the same on ties
    order = ['train', 'valid', 'test']
    random.shuffle(order)
    best_split = None
    best_score = None
    for s in order:
        if not templates_available.get(s, False):
            continue
        r = ratios.get(s, 0)
        if r <= 0:
            continue
        score = current_counts.get(s, 0) / r
        if best_score is None or score < best_score:
            best_score = score
            best_split = s
    # Fallback: if no split has templates, just return 'train' to avoid crash
    return best_split or 'train'

# Precompute which splits have templates available
templates_available = {s: bool(template_pairs_by_split.get(s)) for s in ['train', 'valid', 'test']}

# Initialize counts from existing dataset on disk so we balance across runs
counts = get_existing_counts(OUTPUT_BASE_DIR)

# ----------------- Coverage tracking (per class per split) -----------------
def _zero_counts_dict(names_list):
    return {n: 0 for n in names_list}

def load_or_init_coverage(coverage_path, names_list):
    cov = {
        'classes': names_list[:],
        'counts': {
            'train': _zero_counts_dict(names_list),
            'valid': _zero_counts_dict(names_list),
            'test': _zero_counts_dict(names_list),
            'total': _zero_counts_dict(names_list),
        }
    }
    if coverage_path and os.path.exists(coverage_path):
        try:
            with open(coverage_path, 'r') as f:
                existing_cov = yaml.safe_load(f) or {}
            ex_counts = existing_cov.get('counts', {})
            for k in ['train', 'valid', 'test', 'total']:
                if isinstance(ex_counts.get(k), dict):
                    for n, v in ex_counts[k].items():
                        if n in cov['counts'][k]:
                            try:
                                cov['counts'][k][n] = int(v)
                            except Exception:
                                pass
        except Exception:
            # If malformed, start fresh
            pass
    return cov

def save_coverage(coverage_path, cov):
    if not coverage_path:
        return
    os.makedirs(os.path.dirname(coverage_path), exist_ok=True)
    with open(coverage_path, 'w') as f:
        yaml.dump(cov, f, default_flow_style=False)

def choose_class_for_split(cov, split, available_class_names, makeup_min=0):
    # Optimized: Choose the class with the smallest (split_count, total_count), break ties randomly
    # Prioritize classes with 0 in the target split
    split_counts = cov['counts'][split]
    total_counts = cov['counts']['total']
    
    # Fast path: if makeup_min is specified, prioritize classes below threshold
    if makeup_min and makeup_min > 0:
        backlog = [c for c in available_class_names if split_counts.get(c, 0) < makeup_min]
        if backlog:
            # Only sort the top candidates to avoid full sort
            if len(backlog) <= 20:
                backlog.sort(key=lambda c: (split_counts.get(c,0), total_counts.get(c,0)))
                return random.choice(backlog[:min(10, len(backlog))])
            else:
                # For large backlogs, just pick from lowest 20 by split count
                backlog.sort(key=lambda c: split_counts.get(c,0))
                return random.choice(backlog[:20])
    
    # Find minimum split count efficiently
    min_count = min((split_counts.get(c, 0) for c in available_class_names), default=0)
    
    # Get all classes at minimum count
    zero_bucket = [c for c in available_class_names if split_counts.get(c, 0) == min_count]
    
    if not zero_bucket:
        return random.choice(list(available_class_names))
    
    # For large buckets, avoid sorting all - just sample from lowest total counts
    if len(zero_bucket) <= 20:
        zero_bucket.sort(key=lambda c: total_counts.get(c, 0))
        pick = random.choice(zero_bucket[:min(10, len(zero_bucket))])
    else:
        # For very large buckets, just pick randomly to save time
        pick = random.choice(zero_bucket)
    
    return pick

coverage_path = args.coverage_file or os.path.join(OUTPUT_BASE_DIR, 'coverage.yaml')
coverage = load_or_init_coverage(coverage_path, class_names)

# ===================== IMAGE GENERATION =====================
idx = 0
for _ in range(NUM_SYNTHETIC_IMAGES):
    # Pick split that needs images the most relative to its target ratio
    split_folder = choose_split(counts, TARGET_RATIOS, templates_available)
    template_pairs = template_pairs_by_split.get(split_folder, [])
    if not template_pairs:
        # As a last resort, try any split with available templates
        for alt in ['train', 'valid', 'test']:
            if template_pairs_by_split.get(alt):
                split_folder = alt
                template_pairs = template_pairs_by_split[alt]
                break
        if not template_pairs:
            print("Error: No template pairs available in any split. Aborting generation.")
            break
    
    # Generate one image for the chosen split
    # Randomly select a template image and its label from the current split
    template_path, label_path = random.choice(template_pairs)
    with Image.open(template_path).convert('RGB') as template_img:
        with open(label_path, 'r') as lf:
            lines = lf.readlines()
        
        # Copy template for processing
        img_copy = template_img.copy()
        
        label_copy = []
        
        # Calculate average bbox dimensions using ONLY "Ref" class (class_id==1)
        # This provides accurate sizing since Ref boxes are perfectly aligned
        ref_bbox_widths = []
        ref_bbox_heights = []
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id, cx, cy, bw, bh = parts
                class_id = int(class_id)
                
                # Only use Ref class (class_id==1) for sizing calculations
                if class_id == CLASS_ID_REF:
                    bw = float(bw)
                    bh = float(bh)
                    # Convert normalized coords to pixels
                    ref_bbox_widths.append(bw * template_img.size[0])
                    ref_bbox_heights.append(bh * template_img.size[1])
        
        # Calculate median dimensions from Ref boxes
        if ref_bbox_widths and ref_bbox_heights:
            ref_bbox_widths_sorted = sorted(ref_bbox_widths)
            ref_bbox_heights_sorted = sorted(ref_bbox_heights)
            n_ref = len(ref_bbox_widths_sorted)
            
            if n_ref % 2 == 0:
                avg_bbox_width = (ref_bbox_widths_sorted[n_ref//2 - 1] + ref_bbox_widths_sorted[n_ref//2]) / 2
                avg_bbox_height = (ref_bbox_heights_sorted[n_ref//2 - 1] + ref_bbox_heights_sorted[n_ref//2]) / 2
            else:
                avg_bbox_width = ref_bbox_widths_sorted[n_ref//2]
                avg_bbox_height = ref_bbox_heights_sorted[n_ref//2]
            
            avg_area = avg_bbox_width * avg_bbox_height
            # Set threshold to detect "highlighted" cards (normalized area > 2.5x average)
            # These are oversized cards in broadcast overlays that should scale independently
            highlighted_card_threshold = 2.5 * (avg_area / (template_img.size[0] * template_img.size[1]))
            
            print(f"[sizing] Using {n_ref} Ref boxes: avg_width={avg_bbox_width:.1f}px, avg_height={avg_bbox_height:.1f}px")
            
            # Try to load positioning cache
            positioning_cache = None
            cache_path = get_positioning_cache_path(template_path, POSITIONING_CACHE_DIR)
            positioning_cache = load_positioning_cache(cache_path)
            
            if positioning_cache is None:
                # Cache doesn't exist - build it now
                print(f"[cache] No cache found, building for template: {os.path.basename(template_path)}")
                positioning_cache = build_positioning_cache(
                    template_img, lines, avg_bbox_width, avg_bbox_height, avg_area, highlighted_card_threshold
                )
                save_positioning_cache(cache_path, positioning_cache)
            else:
                print(f"[cache] Loaded cache for template: {os.path.basename(template_path)}")
        else:
            # Fallback if no Ref boxes found (shouldn't happen with new data)
            print("[warning] No Ref boxes found, using fallback sizing")
            avg_area = None
            avg_bbox_width = 300
            avg_bbox_height = 420
            highlighted_card_threshold = None
        
        # Calculate uniform card scale
        # We'll pass the target bbox dimensions to paste_card
        # and it will calculate the scale based on actual card dimensions
        
        # Calculate aspect ratios for all bounding boxes in this image
        aspect_ratios = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                _, cx, cy, bw, bh = parts
                bw = float(bw)
                bh = float(bh)
                if bw > 0:
                    aspect_ratios.append(bh / bw)
        aspect_ratios_sorted = sorted(aspect_ratios)
        n = len(aspect_ratios_sorted)
        lower = int(n * 0.25)
        upper = int(n * 0.75)
        trimmed = aspect_ratios_sorted[lower:upper] if upper > lower else aspect_ratios_sorted
        # Apply aspect scale transformation for 1920x1080 native resolution
        ASPECT_SCALE = (1920.0 / 1080.0) / (640.0 / 640.0)  # = 1.778
        avg_aspect = (sum(trimmed) / len(trimmed) if trimmed else 1.0) * ASPECT_SCALE
        # For high aspect ratios (>= 1.5 * ASPECT_SCALE), average them
        high_threshold = 1.5 * ASPECT_SCALE
        high_aspects = [ar * ASPECT_SCALE for ar in aspect_ratios if ar >= 1.5]
        high_avg_aspect = sum(high_aspects) / len(high_aspects) if high_aspects else avg_aspect * 2.4
        
        # Process each bounding box with independent card selection
        box_index = 0
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                # Select a card image from provided set directories (independent sampling)
                if not card_files:
                    raise RuntimeError("No card PNG files found in the specified card directories.")
                
                # Coverage-guided pick or random/weighted fallback
                if args.coverage_guided and class_to_files:
                    target_split = args.makeup_split or split_folder
                    class_name = choose_class_for_split(coverage, target_split, class_to_files.keys(), makeup_min=args.makeup_min)
                    
                    # Use weighted selection within the chosen class if weights enabled
                    if args.use_popularity_weights:
                        set_dir, card_filename = weighted_card_choice(card_files, weights_dict, class_to_files, class_name)
                    else:
                        # If chosen class has no files (shouldn't happen), fall back to random
                        files_for_class = class_to_files.get(class_name) or [random.choice(card_files)]
                        set_dir, card_filename = random.choice(files_for_class)
                else:
                    # Use weighted selection if enabled, otherwise uniform random
                    if args.use_popularity_weights:
                        set_dir, card_filename = weighted_card_choice(card_files, weights_dict)
                    else:
                        set_dir, card_filename = random.choice(card_files)
                    raw_name = os.path.splitext(card_filename)[0]
                    class_name = canonicalize_name(raw_name)
                
                card_path = os.path.join(set_dir, card_filename)
                raw_name = os.path.splitext(card_filename)[0]
                class_name = canonicalize_name(raw_name)
                class_id = name_to_id[class_name]
                
                # Get cached rotation for this box if available
                cached_rotation = None
                if positioning_cache and box_index < len(positioning_cache.get('boxes', [])):
                    cached_rotation = positioning_cache['boxes'][box_index].get('best_rotation')
                
                with Image.open(card_path).convert('RGBA') as card_img:
                    img_copy = paste_card(img_copy, card_img, parts[1:], avg_aspect, high_avg_aspect, avg_area, avg_bbox_width, avg_bbox_height, highlighted_card_threshold, cached_rotation, skip_bg_tint=False)
                # Update class id using the persistent mapping
                label_copy.append(f"{class_id} {' '.join(parts[1:])}\n")
                # Update in-memory coverage counters
                if args.coverage_guided:
                    try:
                        coverage['counts'][split_folder][class_name] += 1
                        coverage['counts']['total'][class_name] += 1
                    except Exception:
                        pass
                
                box_index += 1
    # Final random rotation (0, 90, 180, 270 degrees) and label update
    final_angle = random.choice([0, 90, 180, 270])
    if final_angle != 0:
        img_copy = img_copy.rotate(final_angle, expand=True)
        w, h = img_copy.size
        updated_labels = []
        for lbl in label_copy:
            parts = lbl.strip().split()
            if len(parts) == 5:
                class_id, cx, cy, bw, bh = parts
                cx = float(cx)
                cy = float(cy)
                bw = float(bw)
                bh = float(bh)
                # Apply rotation transformation
                if final_angle == 90:
                    new_cx, new_cy = cy, 1 - cx
                    new_bw, new_bh = bh, bw
                elif final_angle == 180:
                    new_cx, new_cy = 1 - cx, 1 - cy
                    new_bw, new_bh = bw, bh
                elif final_angle == 270:
                    new_cx, new_cy = 1 - cy, cx
                    new_bw, new_bh = bh, bw
                else:
                    new_cx, new_cy = cx, cy
                    new_bw, new_bh = bw, bh
                updated_labels.append(f"{class_id} {new_cx:.6f} {new_cy:.6f} {new_bw:.6f} {new_bh:.6f}\n")
            else:
                updated_labels.append(lbl)
        label_copy = updated_labels
    unique_id = uuid.uuid4().hex[:8]
    out_img_name = f"synthetic_{idx}_{unique_id}_{os.path.splitext(os.path.basename(template_path))[0]}.png"
    out_label_name = f"synthetic_{idx}_{unique_id}_{os.path.splitext(os.path.basename(template_path))[0]}.txt"
    # Use the chosen split folder
    img_out_dir = os.path.join(OUTPUT_BASE_DIR, split_folder, 'images')
    label_out_dir = os.path.join(OUTPUT_BASE_DIR, split_folder, 'labels')
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(label_out_dir, exist_ok=True)
    
    # Apply screen capture emulation to ~75% of images
    if random.random() < 0.75:
        img_copy = simulate_screen_capture(img_copy)
    
    # Apply occluders based on split:
    # - train: 40% get occluders
    # - valid: keep clean (no occluders)
    # - test: keep clean (no occluders)
    if split_folder == 'train':
        img_copy, label_copy = apply_occluders(
            img_copy, label_copy, 
            coverage_drop=0.85, 
            p_apply=0.40,
            occl_per_img=(1,4),
            per_box_cov=(0.20,0.60)
        )
    
    img_copy.save(os.path.join(img_out_dir, out_img_name))
    with open(os.path.join(label_out_dir, out_label_name), 'w') as outf:
        outf.writelines(label_copy)
    idx += 1
    counts[split_folder] = counts.get(split_folder, 0) + 1
    # Lightweight progress log every 100 images
    if idx % 100 == 0 or idx == NUM_SYNTHETIC_IMAGES:
        try:
            print(f"[progress] generated {idx}/{NUM_SYNTHETIC_IMAGES} | train={counts.get('train',0)} valid={counts.get('valid',0)} test={counts.get('test',0)}")
        except Exception:
            # Avoid any logging-related crashes
            pass
    # Periodically persist all config files to survive crashes
    if idx % 50 == 0:
        try:
            if args.coverage_guided:
                save_coverage(coverage_path, coverage)
            # Also checkpoint classes.yaml and data.yaml
            save_class_index(class_index_path, class_names)
            data_yaml_checkpoint = {
                'train': os.path.join(OUTPUT_BASE_DIR, 'train', 'images').replace('\\', '/'),
                'val': os.path.join(OUTPUT_BASE_DIR, 'valid', 'images').replace('\\', '/'),
                'test': os.path.join(OUTPUT_BASE_DIR, 'test', 'images').replace('\\', '/'),
                'nc': len(class_names),
                'names': class_names,
            }
            with open(data_yaml_path, 'w') as f:
                yaml.dump(data_yaml_checkpoint, f, default_flow_style=False)
        except Exception:
            pass

# After all synthetic images/labels are generated, write classes index and data.yaml
save_class_index(class_index_path, class_names)

data_yaml = {
    'train': os.path.join(OUTPUT_BASE_DIR, 'train', 'images').replace('\\', '/'),
    'val': os.path.join(OUTPUT_BASE_DIR, 'valid', 'images').replace('\\', '/'),
    'test': os.path.join(OUTPUT_BASE_DIR, 'test', 'images').replace('\\', '/'),
    'nc': len(class_names),
    'names': class_names,
}

data_yaml_path = os.path.join(OUTPUT_BASE_DIR, 'data.yaml')
with open(data_yaml_path, 'w') as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

# Persist final coverage if enabled
if args.coverage_guided:
    save_coverage(coverage_path, coverage)
