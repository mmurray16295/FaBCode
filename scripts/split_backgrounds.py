import os
import random
import shutil

# Source directories
src_img_dir = r"/root/FaBCode/data/images/YouTube_Labeled/Full Size Card Detection.v2-full-size.yolov11/train/images"
src_lbl_dir = r"/root/FaBCode/data/images/YouTube_Labeled/Full Size Card Detection.v2-full-size.yolov11/train/labels"

# Destination base (YouTube_Labeled folder for background templates)
dst_base = r"/root/FaBCode/data/images/YouTube_Labeled"
splits = {"train": 0.7, "valid": 0.2, "test": 0.1}

# Gather all image files
img_files = [f for f in os.listdir(src_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
random.shuffle(img_files)

# Calculate split sizes
n = len(img_files)
split_counts = {k: int(v * n) for k, v in splits.items()}
split_counts["test"] = n - split_counts["train"] - split_counts["valid"]  # ensure all files used

# Assign files to splits
split_files = {
    "train": img_files[:split_counts["train"]],
    "valid": img_files[split_counts["train"]:split_counts["train"]+split_counts["valid"]],
    "test": img_files[split_counts["train"]+split_counts["valid"]:]
}

# Clear all existing background images and labels from destination folders first
print("Clearing existing backgrounds from destination folders...")
for split in splits.keys():
    img_dst = os.path.join(dst_base, split, "images")
    lbl_dst = os.path.join(dst_base, split, "labels")
    if os.path.exists(img_dst):
        for f in os.listdir(img_dst):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                os.remove(os.path.join(img_dst, f))
    if os.path.exists(lbl_dst):
        for f in os.listdir(lbl_dst):
            if f.lower().endswith('.txt'):
                os.remove(os.path.join(lbl_dst, f))

print(f"Splitting {len(img_files)} backgrounds into train/valid/test...")
for split, files in split_files.items():
    img_dst = os.path.join(dst_base, split, "images")
    lbl_dst = os.path.join(dst_base, split, "labels")
    os.makedirs(img_dst, exist_ok=True)
    os.makedirs(lbl_dst, exist_ok=True)
    for fname in files:
        # Copy image
        shutil.copy2(os.path.join(src_img_dir, fname), os.path.join(img_dst, fname))
        # Copy label (if exists)
        lbl_name = os.path.splitext(fname)[0] + ".txt"
        lbl_src_path = os.path.join(src_lbl_dir, lbl_name)
        if os.path.exists(lbl_src_path):
            shutil.copy2(lbl_src_path, os.path.join(lbl_dst, lbl_name))

print("Split complete!")
