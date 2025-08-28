# Fix: import glob and define TEMPLATE_BASE
import os
import random
import glob
from PIL import Image

# Paths
TEMPLATE_BASE = 'data/images/YouTube_Labeled/FaB Card Detection.v2i.yolov11'
WTR_DIR = 'data/images/WTR'  # Folder with WTR card images

wtr_cards = [f for f in os.listdir(WTR_DIR) if f.endswith('.png')]


import os
import random
from PIL import Image

# Paths
WTR_DIR = 'data/images/WTR'  # Folder with WTR card images
TEMPLATE_DIR = 'data/images/YouTube_Labeled'  # Folder with template images
LABELS_DIR = 'data/images/YouTube_Labeled/labels'  # Folder with YOLO label files
OUTPUT_IMG_DIR = 'data/synthetic/images'
OUTPUT_LABEL_DIR = 'data/synthetic/labels'
# Gather all template images and labels from train, test, valid
template_pairs = []
WTR_DIR = 'data/images/WTR'  # Folder with WTR card images
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

def paste_card(template_img, card_img, bbox, angle):
    w, h = template_img.size
    cx, cy, bw, bh = [float(x) for x in bbox]
    box_w, box_h = int(bw * w), int(bh * h)
    box_x, box_y = int((cx - bw/2) * w), int((cy - bh/2) * h)
    # Fixed card size (e.g., 252x352 px for standard TCG card)
    FIXED_CARD_W, FIXED_CARD_H = int(62/1.15), int(87/1.15)
    # Resize to fixed scale
    card_resized = card_img.resize((FIXED_CARD_W, FIXED_CARD_H), Image.LANCZOS)
    # First scrunch/widen
    adj_w1 = int(FIXED_CARD_W * 1.05)
    adj_h1 = int(FIXED_CARD_H * 0.96)
    card_resized = card_resized.resize((adj_w1, adj_h1), Image.LANCZOS)
    # Second scrunch/widen
    adj_w2 = int(adj_w1 * 1.05)
    adj_h2 = int(adj_h1 * 0.96)
    card_resized = card_resized.resize((adj_w2, adj_h2), Image.LANCZOS)
    # Search for rotation angle that maximizes area inside bounding box
    max_inside = None
    best_angle = 0
    best_card = None
    for test_angle in range(0, 180, 5):
        test_rotated = card_resized.rotate(test_angle, expand=True)
        rw, rh = test_rotated.size
        # Card is centered in the box
        offset_x = box_x + (box_w - rw) // 2
        offset_y = box_y + (box_h - rh) // 2
        # Bounding box rectangle
        box_rect = (box_x, box_y, box_x + box_w, box_y + box_h)
        # Card rectangle
        card_rect = (offset_x, offset_y, offset_x + rw, offset_y + rh)
        # Calculate overlap area
        x_overlap = max(0, min(box_rect[2], card_rect[2]) - max(box_rect[0], card_rect[0]))
        y_overlap = max(0, min(box_rect[3], card_rect[3]) - max(box_rect[1], card_rect[1]))
        overlap_area = x_overlap * y_overlap
        if (max_inside is None) or (overlap_area > max_inside):
            max_inside = overlap_area
            best_angle = test_angle
            best_card = test_rotated
    # Paste best rotated card
    rw, rh = best_card.size
    offset_x = box_x + (box_w - rw) // 2
    offset_y = box_y + (box_h - rh) // 2
    template_img.paste(best_card, (offset_x, offset_y), best_card if best_card.mode == 'RGBA' else None)
    return template_img

# Number of synthetic images to generate
NUM_SYNTHETIC_IMAGES = 1

for idx in range(NUM_SYNTHETIC_IMAGES):
    # Randomly select a template image and its label
    template_path, label_path = random.choice(template_pairs)
    with Image.open(template_path).convert('RGB') as template_img:
        with open(label_path, 'r') as lf:
            lines = lf.readlines()
        img_copy = template_img.copy()
        label_copy = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                # Randomly select a WTR card for this bounding box
                card_idx = random.randint(0, len(wtr_cards)-1)
                card_name = wtr_cards[card_idx]
                card_path = os.path.join(WTR_DIR, card_name)
                with Image.open(card_path).convert('RGBA') as card_img:
                    img_copy = paste_card(img_copy, card_img, parts[1:], None)
                # Update class id to WTR card index
                label_copy.append(f"{card_idx} {' '.join(parts[1:])}\n")
    out_img_name = f"synthetic_{idx}_{os.path.splitext(os.path.basename(template_path))[0]}.png"
    out_label_name = f"synthetic_{idx}_{os.path.splitext(os.path.basename(template_path))[0]}.txt"
    img_copy.save(os.path.join(OUTPUT_IMG_DIR, out_img_name))
    with open(os.path.join(OUTPUT_LABEL_DIR, out_label_name), 'w') as outf:
        outf.writelines(label_copy)
