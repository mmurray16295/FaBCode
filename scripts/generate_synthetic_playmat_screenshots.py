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
from PIL import Image
import yaml
import argparse
import re
import json

import pathlib

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
    aspect_ratio = box_h / box_w if box_w > 0 else 1.0
    area = box_w * box_h
    crop_condition = aspect_ratio < 0.75
    if avg_area is not None and aspect_ratio > 1.5 and area < 0.65 * avg_area:
        crop_condition = True
    if crop_condition:
        rw, rh = card_img.size
        left = max(0, (rw - box_w) // 2)
        upper = max(0, (rh - box_h) // 2)
        right = left + box_w
        lower = upper + box_h
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

def paste_card(template_img, card_img, bbox, avg_aspect, high_avg_aspect, avg_area):
    w, h = template_img.size
    cx, cy, bw, bh = [float(x) for x in bbox]
    box_w, box_h = int(bw * w), int(bh * h)
    box_x, box_y = int((cx - bw/2) * w), int((cy - bh/2) * h)
    # Fixed card size
    FIXED_CARD_W, FIXED_CARD_H = int(62/1.15), int(87/1.15)
    card_resized = card_img.resize((FIXED_CARD_W, FIXED_CARD_H), Image.LANCZOS)
    # First scrunch/widen
    adj_w1 = int(FIXED_CARD_W * 1.05)
    adj_h1 = int(FIXED_CARD_H * 0.96)
    card_resized = card_resized.resize((adj_w1, adj_h1), Image.LANCZOS)
    # Second scrunch/widen
    adj_w2 = int(adj_w1 * 1.05)
    adj_h2 = int(adj_h1 * 0.96)
    card_resized = card_resized.resize((adj_w2, adj_h2), Image.LANCZOS)
    # Compress height to match 640x640 aspect ratio
    aspect_compress = 63 / 88
    adj_h3 = int(card_resized.size[1] * aspect_compress)
    card_resized = card_resized.resize((card_resized.size[0], adj_h3), Image.LANCZOS)
    # Rotation logic based on bounding box aspect ratio (height/width)
    ratio = box_h / box_w if box_w > 0 else 1.0
    # Use refined logic for high aspect ratios
    if ratio >= 1.5:
        angle = refined_rotation(ratio, high_avg_aspect)
    else:
        # Original logic for others
        default_angles = [90, 270]
        vertical_angles = [0, 180]
        if ratio <= avg_aspect:
            angle = random.choice(default_angles)
        elif ratio >= avg_aspect * 2.4:
            angle = random.choice(vertical_angles)
        else:
            interp = (ratio - avg_aspect) / (avg_aspect * 2.4 - avg_aspect)
            if random.random() < 0.5:
                angle = int(90 * (1 - interp) + 0 * interp)
            else:
                angle = int(270 * (1 - interp) + 180 * interp)
    # Heuristic: guess rotation direction
    direction = guess_rotation_direction(template_img, box_x, box_y, box_w, box_h)
    card_rotated = card_resized.rotate(angle * direction, expand=True)
    # Best fit the rotated card to the bounding box
    # Calculate average area for all bounding boxes in this image
    # (Move this logic to where paste_card is called)
    card_fitted = best_fit_to_bbox(card_rotated, box_w, box_h, avg_area)
    rw, rh = card_fitted.size
    # Center the card in the bounding box
    offset_x = box_x + (box_w - rw) // 2
    offset_y = box_y + (box_h - rh) // 2
    # Post-processing: sample background properties and apply to card
    from PIL import ImageFilter, ImageEnhance
    import numpy as np
    # Sample bounding box region from template
    bbox_crop = template_img.crop((box_x, box_y, box_x + box_w, box_y + box_h)).convert('RGB')
    # Calculate average brightness
    arr = np.array(bbox_crop)
    avg_brightness = np.mean(arr)
    # Estimate blur by variance of Laplacian (simple proxy)
    gray = np.mean(arr, axis=2)
    laplacian = np.abs(np.gradient(gray)[0]) + np.abs(np.gradient(gray)[1])
    blur_level = max(0.5, min(2.5, 2.5 - np.var(laplacian) / 50))  # scale to [0.5, 2.5]
    # Apply blur to card
    card_post = card_fitted.filter(ImageFilter.GaussianBlur(radius=blur_level))
    # Match brightness
    card_post = ImageEnhance.Brightness(card_post).enhance(avg_brightness / 128)
    # Paste processed card
    template_img.paste(card_post, (offset_x, offset_y), card_post if card_post.mode == 'RGBA' else None)
    return template_img

def refined_rotation(ratio, high_avg_aspect):
    # For high aspect ratios, interpolate between vertical and default
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
class_names = load_or_create_class_index(class_index_path)
existing = set(class_names)
for set_dir, card_filename in card_files:
    raw_name = os.path.splitext(card_filename)[0]
    name = canonicalize_name(raw_name)
    if name not in existing:
        class_names.append(name)
        existing.add(name)
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
    # Choose the class with the smallest (split_count, total_count), break ties randomly
    # Prioritize classes with 0 in the target split
    split_counts = cov['counts'][split]
    total_counts = cov['counts']['total']
    # Filter to available classes (i.e., we have image files)
    candidates = [c for c in available_class_names if c in split_counts]
    if not candidates:
        # Fallback: random from available
        return random.choice(list(available_class_names))
    # If makeup_min is specified, prioritize classes below the threshold in this split
    if makeup_min and makeup_min > 0:
        backlog = [c for c in candidates if split_counts.get(c, 0) < makeup_min]
        if backlog:
            # Sort backlog by (split_count, total_count)
            backlog.sort(key=lambda c: (split_counts.get(c,0), total_counts.get(c,0)))
            top_k = max(1, min(10, len(backlog)))
            return random.choice(backlog[:top_k])
    # Bucket by split count
    by_split = {}
    for c in candidates:
        sc = split_counts.get(c, 0)
        by_split.setdefault(sc, []).append(c)
    min_split = min(by_split.keys())
    zero_bucket = by_split[min_split]
    # If many tied, use total count to break ties (lowest first)
    zero_bucket.sort(key=lambda c: total_counts.get(c, 0))
    # Random among the lowest few to avoid overfitting a single class
    top_k = max(1, min(10, len(zero_bucket)))
    pick = random.choice(zero_bucket[:top_k])
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
        avg_area = (sum(areas) / len(areas)) * template_img.size[0] * template_img.size[1] if areas else None
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
        avg_aspect = sum(trimmed) / len(trimmed) if trimmed else 1.0
        # For high aspect ratios (>= 1.5), average them
        high_aspects = [ar for ar in aspect_ratios if ar >= 1.5]
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
                    img_copy = paste_card(img_copy, card_img, parts[1:], avg_aspect, high_avg_aspect, avg_area)
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
    # Periodically persist coverage file
    if args.coverage_guided and idx % 50 == 0:
        try:
            save_coverage(coverage_path, coverage)
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
