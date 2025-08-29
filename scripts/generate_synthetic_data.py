
import os
import glob
import random
import os
import uuid
from PIL import Image
import yaml
import argparse

# Paths
TEMPLATE_BASE = 'data/images/YouTube_Labeled/FaB Card Detection.v4i.yolov11'
WTR_DIR = 'data/images/WTR'  # Folder with WTR card images
OUTPUT_BASE_DIR = os.path.join('data', 'synthetic')

# Gather all template images and labels from train, test, valid
template_pairs = []
for split in ['train', 'test', 'valid']:
    img_dir = os.path.join(TEMPLATE_BASE, split, 'images')
    label_dir = os.path.join(TEMPLATE_BASE, split, 'labels')
    images = glob.glob(os.path.join(img_dir, '*.png')) + glob.glob(os.path.join(img_dir, '*.jpg'))
    for img_path in images:
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, base + '.txt')
        if os.path.exists(label_path):
            template_pairs.append((img_path, label_path))
wtr_cards = [f for f in os.listdir(WTR_DIR) if f.endswith('.png')]

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
    theta = math.radians(angle % 180)  # Normalize angle to [0, 180]
    # Avoid division by zero for sin/cos
    sin_theta = abs(math.sin(theta)) if abs(math.sin(theta)) > 1e-3 else 1.0
    cos_theta = abs(math.cos(theta)) if abs(math.cos(theta)) > 1e-3 else 1.0
    if sin_theta == 1.0 and cos_theta == 1.0:
        # Unrotated, just fit to box
        new_w, new_h = box_w, box_h
    else:
        new_h = int(box_h / sin_theta)
        new_w = int(box_w / cos_theta)
    return card_img.resize((new_w, new_h), Image.LANCZOS)
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

def paste_card(template_img, card_img, bbox, min_height):
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
parser.add_argument('--num-images', type=int, default=100, help='Number of synthetic images to generate')
args = parser.parse_args()

# Number of synthetic images to generate
NUM_SYNTHETIC_IMAGES = args.num_images


for idx in range(NUM_SYNTHETIC_IMAGES):
    # Randomly select a template image and its label
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
                # Randomly select a WTR card for this bounding box
                card_idx = random.randint(0, len(wtr_cards)-1)
                card_name = wtr_cards[card_idx]
                card_path = os.path.join(WTR_DIR, card_name)
                with Image.open(card_path).convert('RGBA') as card_img:
                    img_copy = paste_card(img_copy, card_img, parts[1:], (avg_aspect, high_avg_aspect))
                # Update class id to WTR card index
                label_copy.append(f"{card_idx} {' '.join(parts[1:])}\n")
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
    # Assign to train/valid/test split
    split_rand = random.random()
    if split_rand < 0.7:
        split_folder = 'train'
    elif split_rand < 0.9:
        split_folder = 'valid'
    else:
        split_folder = 'test'
    img_out_dir = os.path.join(OUTPUT_BASE_DIR, split_folder, 'images')
    label_out_dir = os.path.join(OUTPUT_BASE_DIR, split_folder, 'labels')
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(label_out_dir, exist_ok=True)
    img_copy.save(os.path.join(img_out_dir, out_img_name))
    with open(os.path.join(label_out_dir, out_label_name), 'w') as outf:
        outf.writelines(label_copy)

# After all synthetic images/labels are generated, write data.yaml
wtr_card_names = [os.path.splitext(f)[0] for f in wtr_cards]
data_yaml = {
    'train': os.path.join(OUTPUT_BASE_DIR, 'train', 'images').replace('\\', '/'),
    'val': os.path.join(OUTPUT_BASE_DIR, 'valid', 'images').replace('\\', '/'),
    'test': os.path.join(OUTPUT_BASE_DIR, 'test', 'images').replace('\\', '/'),
    'nc': len(wtr_card_names),
    'names': wtr_card_names
}
data_yaml_path = os.path.join(OUTPUT_BASE_DIR, 'data.yaml')
with open(data_yaml_path, 'w') as f:
    yaml.dump(data_yaml, f, default_flow_style=False)
