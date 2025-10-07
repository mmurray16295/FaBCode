# ===================== PATH & RUN SETTINGS =====================
# Change these paths to select which sets and backgrounds to use

# Folder containing background images (playmat screenshots)
TEMPLATE_BASE = r'C:\VS Code\FaB Code\data\images\YouTube_Labeled'

# Folder containing card images for the set you want to use (e.g., SEA)
SEA_DIR = r'C:\VS Code\FaB Code\data\images\SEA'

# Folder to save synthetic output (will create train, test, valid subfolders)
OUTPUT_BASE_DIR = r'C:\VS Code\FaB Code\data\synthetic 2'

# Number of synthetic images to generate (set this for each run)
NUM_SYNTHETIC_IMAGES = 20  # <--- Change this value for your trial runs

# Allow specifying multiple card set directories; defaults to [SEA_DIR]
CARD_SET_DIRS = [SEA_DIR]

# ==============================================================

import os
import glob
import random
import os
import uuid
from PIL import Image, ImageFilter, ImageDraw
import yaml
import argparse
import re
import json
import io
import numpy as np
import math

import pathlib

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
    Collapse reprints by stripping a trailing _<SET><digits> suffix if present.
    Examples:
      'Enlightened_Strike_WTR159' -> 'Enlightened_Strike'
      'Snatch_SEA169' -> 'Snatch'
    If no suffix matches, returns the name unchanged.
    """
    m = re.match(r"^(.*)_([A-Z]{2,5}\d{1,5})$", name)
    return m.group(1) if m else name


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

def paste_card(template_img, card_img, bbox, avg_aspect, high_avg_aspect, avg_area, target_bbox_width, target_bbox_height, highlighted_threshold):
    """
    Place card in bbox with uniform scale across all cards:
    1. Detect if this is a "highlighted" card (much larger bbox)
    2. For normal cards: use uniform scale across scene
    3. For highlighted cards: scale independently to fit bbox
    4. Test all 360 rotation angles to maximize visible card area
    5. Crop to bbox if card extends beyond
    6. Apply background color tint and other post-processing
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
    
    # Apply uniform scale to card (same for all cards in scene)
    scaled_w = int(orig_w * card_scale)
    scaled_h = int(orig_h * card_scale)
    card_scaled = card_img.resize((scaled_w, scaled_h), Image.LANCZOS)
    
    # Test rotation angles to maximize card visibility AND maximize corner contact with bbox
    best_angle = 0
    best_visible_area = 0
    best_corners_touching = 0
    
    # Calculate total card pixels for early exit condition
    card_arr = np.array(card_scaled)
    if len(card_arr.shape) == 3 and card_arr.shape[2] == 4:
        total_card_pixels = np.sum(card_arr[:, :, 3] > 10)
    else:
        total_card_pixels = scaled_w * scaled_h
    
    # Tolerance for "corner touching": 5% of bbox dimensions
    tolerance_w = box_w * 0.05
    tolerance_h = box_h * 0.05
    
    # Sample every 8 degrees for speed (45 iterations instead of 360)
    for angle in range(0, 360, 8):
        # Rotate scaled card WITH expansion to see full rotated bounds
        test_card = card_scaled.rotate(-angle, expand=True, resample=Image.BICUBIC, fillcolor=(0, 0, 0, 0))
        
        # Calculate overlap between rotated card bounds and bbox
        rot_w, rot_h = test_card.size
        
        # Count how many corners of the rotated card touch the bbox edges (with tolerance)
        corners_touching = 0
        if rot_w >= box_w - tolerance_w:  # Left and right edges touch (within tolerance)
            corners_touching += 2
        if rot_h >= box_h - tolerance_h:  # Top and bottom edges touch (within tolerance)
            corners_touching += 2
        
        # Visible area is the intersection
        overlap_w = min(rot_w, box_w)
        overlap_h = min(rot_h, box_h)
        
        # Count actual card pixels in this overlap region (not transparent)
        overlap_region = test_card.crop((0, 0, overlap_w, overlap_h))
        test_arr = np.array(overlap_region)
        
        if len(test_arr.shape) == 3 and test_arr.shape[2] == 4:  # Has alpha channel
            visible_pixels = np.sum(test_arr[:, :, 3] > 10)  # Count non-transparent pixels
        else:
            visible_pixels = overlap_w * overlap_h  # No alpha, all visible
        
        # Prioritize: 100% visible + 4 corners > fewer corners > more visible area
        is_better = False
        if visible_pixels >= total_card_pixels * 0.98:  # Card is fully visible
            if corners_touching > best_corners_touching:
                is_better = True
            elif corners_touching == best_corners_touching and visible_pixels > best_visible_area:
                is_better = True
        elif visible_pixels > best_visible_area:
            # Not fully visible yet, just maximize visible area
            is_better = True
        
        if is_better:
            best_visible_area = visible_pixels
            best_corners_touching = corners_touching
            best_angle = angle
            
            # Early exit: 100% visible AND 4 corners touching = perfect fit
            if best_visible_area >= total_card_pixels * 0.98 and best_corners_touching >= 4:
                break
    
    # Apply best rotation WITH expansion
    if best_angle != 0:
        card_rotated = card_scaled.rotate(-best_angle, expand=True, resample=Image.BICUBIC, fillcolor=(0, 0, 0, 0))
    else:
        card_rotated = card_scaled
    
    # Crop or center card in bbox
    card_w, card_h = card_rotated.size
    if card_w > box_w or card_h > box_h:
        # Card is larger - crop to bbox (randomly choose corner alignment)
        corner = random.choice(['tl', 'tr', 'bl', 'br', 'center'])
        if corner == 'tl':
            crop_x, crop_y = 0, 0
        elif corner == 'tr':
            crop_x, crop_y = max(0, card_w - box_w), 0
        elif corner == 'bl':
            crop_x, crop_y = 0, max(0, card_h - box_h)
        elif corner == 'br':
            crop_x, crop_y = max(0, card_w - box_w), max(0, card_h - box_h)
        else:  # center
            crop_x = max(0, (card_w - box_w) // 2)
            crop_y = max(0, (card_h - box_h) // 2)
        
        card_final = card_rotated.crop((crop_x, crop_y, crop_x + box_w, crop_y + box_h))
    else:
        # Card is smaller - center it in bbox with transparent padding
        card_final = Image.new('RGBA', (box_w, box_h), (0, 0, 0, 0))
        paste_x = (box_w - card_w) // 2
        paste_y = (box_h - card_h) // 2
        card_final.paste(card_rotated, (paste_x, paste_y), card_rotated)
    
    # Post-processing: sample background properties and apply to card
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
args = parser.parse_args()

# Number of synthetic images to generate
NUM_SYNTHETIC_IMAGES = args.num_images if args.num_images is not None else NUM_SYNTHETIC_IMAGES

# Override card set directories from CLI if provided
if args.card_dirs:
    CARD_SET_DIRS = args.card_dirs

# Load card files for this run
card_files = load_card_files(CARD_SET_DIRS)

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
        img_copy = template_img.copy()
        label_copy = []
        # Find min_height for all bounding boxes in this image
        heights = []
        areas = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                _, cx, cy, bw, bh = parts
                heights.append(float(bh))
                # Calculate area for each bounding box
                bw = float(bw)
                bh = float(bh)
                areas.append(bw * bh)
        min_height = min(heights) if heights else 1.0
        
        # Calculate average area, but filter out outliers (covered/partial cards)
        # Also track the actual bbox dimensions (not just areas)
        bbox_widths = []
        bbox_heights = []
        highlighted_card_threshold = None  # For detecting blown-up cards
        
        if areas:
            # Sort areas to find the central cluster
            sorted_areas = sorted(areas)
            n_areas = len(sorted_areas)
            
            # Find median area as the cluster center
            if n_areas % 2 == 0:
                median_area = (sorted_areas[n_areas//2 - 1] + sorted_areas[n_areas//2]) / 2
            else:
                median_area = sorted_areas[n_areas//2]
            
            # Filter: keep only boxes within ±10% of median (normal cards)
            # Exclude boxes >1.5x median (highlighted/blown-up cards)
            lower_bound = median_area * 0.9
            upper_bound = median_area * 1.1
            highlighted_card_threshold = median_area * 1.5  # Cards bigger than this are "highlighted"
            
            # Collect dimensions of filtered boxes (normal cards only)
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    _, cx, cy, bw, bh = parts
                    bw = float(bw)
                    bh = float(bh)
                    area = bw * bh
                    if lower_bound <= area <= upper_bound:
                        # Convert normalized coords to pixels
                        bbox_widths.append(bw * template_img.size[0])
                        bbox_heights.append(bh * template_img.size[1])
            
            # Use median of filtered boxes (more robust than average)
            if len(bbox_widths) >= max(3, n_areas // 3):  # At least 3 or 1/3 of boxes
                bbox_widths_sorted = sorted(bbox_widths)
                bbox_heights_sorted = sorted(bbox_heights)
                n_filtered = len(bbox_widths_sorted)
                
                if n_filtered % 2 == 0:
                    avg_bbox_width = (bbox_widths_sorted[n_filtered//2 - 1] + bbox_widths_sorted[n_filtered//2]) / 2
                    avg_bbox_height = (bbox_heights_sorted[n_filtered//2 - 1] + bbox_heights_sorted[n_filtered//2]) / 2
                else:
                    avg_bbox_width = bbox_widths_sorted[n_filtered//2]
                    avg_bbox_height = bbox_heights_sorted[n_filtered//2]
                
                avg_area = avg_bbox_width * avg_bbox_height
            else:
                # Fall back to median calculation
                avg_area = median_area * template_img.size[0] * template_img.size[1]
                avg_bbox_width = math.sqrt(avg_area / 1.4)
                avg_bbox_height = avg_bbox_width * 1.4
        else:
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
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                # Select a card image from provided set directories
                if not card_files:
                    raise RuntimeError("No card PNG files found in the specified card directories.")
                # Coverage-guided pick or random fallback
                if args.coverage_guided and class_to_files:
                    target_split = args.makeup_split or split_folder
                    class_name = choose_class_for_split(coverage, target_split, class_to_files.keys(), makeup_min=args.makeup_min)
                    # If chosen class has no files (shouldn't happen), fall back to random
                    files_for_class = class_to_files.get(class_name) or [random.choice(card_files)]
                    set_dir, card_filename = random.choice(files_for_class)
                else:
                    set_dir, card_filename = random.choice(card_files)
                    raw_name = os.path.splitext(card_filename)[0]
                    class_name = canonicalize_name(raw_name)
                card_path = os.path.join(set_dir, card_filename)
                raw_name = os.path.splitext(card_filename)[0]
                class_name = canonicalize_name(raw_name)
                class_id = name_to_id[class_name]
                with Image.open(card_path).convert('RGBA') as card_img:
                    img_copy = paste_card(img_copy, card_img, parts[1:], avg_aspect, high_avg_aspect, avg_area, avg_bbox_width, avg_bbox_height, highlighted_card_threshold)
                # Update class id using the persistent mapping
                label_copy.append(f"{class_id} {' '.join(parts[1:])}\n")
                # Update in-memory coverage counters
                if args.coverage_guided:
                    try:
                        coverage['counts'][split_folder][class_name] += 1
                        coverage['counts']['total'][class_name] += 1
                    except Exception:
                        pass
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
