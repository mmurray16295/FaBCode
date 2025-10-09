"""
Test chaos mode by generating images with bounding boxes drawn on them.
This allows visual verification that the labels match the actual card positions.
"""
import sys
import os

# Run the main generator with 100% chaos mode and save to synth_3
os.system('python scripts/generate_synthetic_playmat_screenshots.py --num-images 10 --popularity-min 1 --popularity-max 50 --use-popularity-weights --chaos-mode-probability 1.0 --output-dir data/synth_3')
print("\n" + "="*60)
print("Now drawing bounding boxes on the generated images...")
print("="*60 + "\n")

# Now draw bounding boxes on the generated images
from PIL import Image, ImageDraw, ImageFont
import glob

# Find all generated images
splits = ['train', 'valid', 'test']
for split in splits:
    img_dir = f'data/synth_3/{split}/images'
    label_dir = f'data/synth_3/{split}/labels'
    
    if not os.path.exists(img_dir):
        continue
    
    images = glob.glob(os.path.join(img_dir, '*.png'))
    print(f"\nProcessing {len(images)} images in {split}...")
    
    for img_path in images:
        # Load image
        img = Image.open(img_path).convert('RGB')
        W, H = img.size
        draw = ImageDraw.Draw(img)
        
        # Load corresponding label
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, base + '.txt')
        
        if not os.path.exists(label_path):
            print(f"  Warning: No label found for {base}")
            continue
        
        with open(label_path, 'r') as f:
            labels = f.readlines()
        
        # Draw each bounding box
        colors = [
            (255, 0, 0),      # Red
            (0, 255, 0),      # Green
            (0, 0, 255),      # Blue
            (255, 255, 0),    # Yellow
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Cyan
            (255, 128, 0),    # Orange
            (128, 0, 255),    # Purple
        ]
        
        for idx, line in enumerate(labels):
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            
            class_id, cx, cy, bw, bh = parts
            cx, cy, bw, bh = float(cx), float(cy), float(bw), float(bh)
            
            # Convert YOLO format to pixel coordinates
            x1 = int((cx - bw/2) * W)
            y1 = int((cy - bh/2) * H)
            x2 = int((cx + bw/2) * W)
            y2 = int((cy + bh/2) * H)
            
            # Draw bounding box with thick line
            color = colors[idx % len(colors)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            
            # Draw class ID label
            draw.text((x1 + 5, y1 + 5), f"Class {class_id}", fill=color)
        
        # Save image with bounding boxes
        output_path = img_path.replace('.png', '_with_boxes.png')
        img.save(output_path)
        print(f"  Saved: {os.path.basename(output_path)}")

print("\n" + "="*60)
print("DONE! Check data/synth_3/*/images/*_with_boxes.png")
print("="*60)
