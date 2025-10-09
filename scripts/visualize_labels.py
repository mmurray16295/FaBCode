"""
Visualize YOLO labels by drawing bounding boxes on images.
This helps verify that labels match the actual card positions after all transformations.
"""
import os
import glob
from PIL import Image, ImageDraw, ImageFont

def draw_boxes_on_images(base_dir, splits=['train', 'valid', 'test'], max_images_per_split=None):
    """
    Draw bounding boxes on all images in the specified splits.
    
    Args:
        base_dir: Base directory containing split folders (e.g., 'data/synthetic_2')
        splits: List of split names to process
        max_images_per_split: Optional limit on number of images to process per split
    """
    colors = [
        (255, 0, 0),      # Red
        (0, 255, 0),      # Green
        (0, 0, 255),      # Blue
        (255, 255, 0),    # Yellow
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Cyan
        (255, 128, 0),    # Orange
        (128, 0, 255),    # Purple
        (255, 128, 128),  # Light Red
        (128, 255, 128),  # Light Green
    ]
    
    total_processed = 0
    
    for split in splits:
        img_dir = os.path.join(base_dir, split, 'images')
        label_dir = os.path.join(base_dir, split, 'labels')
        
        if not os.path.exists(img_dir):
            print(f"[skip] {split}: directory not found")
            continue
        
        images = glob.glob(os.path.join(img_dir, '*.png')) + glob.glob(os.path.join(img_dir, '*.jpg'))
        
        if max_images_per_split:
            images = images[:max_images_per_split]
        
        print(f"\n[{split}] Processing {len(images)} images...")
        
        for img_path in images:
            # Load image
            img = Image.open(img_path).convert('RGB')
            W, H = img.size
            draw = ImageDraw.Draw(img)
            
            # Load corresponding label
            base = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(label_dir, base + '.txt')
            
            if not os.path.exists(label_path):
                print(f"  [warning] No label found for {base}")
                continue
            
            with open(label_path, 'r') as f:
                labels = f.readlines()
            
            # Draw each bounding box
            for idx, line in enumerate(labels):
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                
                class_id, cx, cy, bw, bh = parts
                cx, cy, bw, bh = float(cx), float(cy), float(bw), float(bh)
                
                # Convert YOLO format (normalized center + size) to pixel coordinates
                x1 = int((cx - bw/2) * W)
                y1 = int((cy - bh/2) * H)
                x2 = int((cx + bw/2) * W)
                y2 = int((cy + bh/2) * H)
                
                # Draw bounding box with thick line
                color = colors[idx % len(colors)]
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                # Draw class ID label at top-left of box
                label_text = f"C{class_id}"
                # Draw text background for readability
                text_bbox = draw.textbbox((x1 + 2, y1 + 2), label_text)
                draw.rectangle(text_bbox, fill=(0, 0, 0, 200))
                draw.text((x1 + 2, y1 + 2), label_text, fill=color)
            
            # Save image with bounding boxes
            output_path = img_path.replace('.png', '_labeled.png').replace('.jpg', '_labeled.jpg')
            img.save(output_path)
            total_processed += 1
        
        print(f"  [done] Saved {len(images)} labeled images to {split}/images/*_labeled.*")
    
    print(f"\n[complete] Total images processed: {total_processed}")
    print(f"[complete] Look for files ending in '_labeled.png' or '_labeled.jpg'")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize YOLO labels by drawing bounding boxes on images')
    parser.add_argument('--data-dir', type=str, default='data/synthetic_2', 
                        help='Base directory containing train/valid/test splits')
    parser.add_argument('--splits', nargs='+', default=['train', 'valid', 'test'],
                        help='Splits to process (default: train valid test)')
    parser.add_argument('--max-per-split', type=int, default=None,
                        help='Maximum number of images to process per split (default: all)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("YOLO Label Visualization Tool")
    print("="*60)
    print(f"Data directory: {args.data_dir}")
    print(f"Splits: {', '.join(args.splits)}")
    if args.max_per_split:
        print(f"Max images per split: {args.max_per_split}")
    print("="*60)
    
    draw_boxes_on_images(args.data_dir, args.splits, args.max_per_split)
