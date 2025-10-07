"""
Fix data.yaml to match the actual class IDs used in the synthetic data labels.
This script scans all label files to find which class IDs were used, then creates
a corrected data.yaml using the class list from coverage.yaml.
"""
import os
import glob
import yaml
from collections import defaultdict

# Paths
SYNTHETIC_BASE = r'C:\VS Code\FaB Code\data\synthetic 2'
COVERAGE_PATH = os.path.join(SYNTHETIC_BASE, 'coverage.yaml')
CLASSES_PATH = os.path.join(SYNTHETIC_BASE, 'classes.yaml')
DATA_YAML_PATH = os.path.join(SYNTHETIC_BASE, 'data.yaml')

print("Step 1: Loading coverage.yaml to get full class list...")
with open(COVERAGE_PATH, 'r') as f:
    coverage = yaml.safe_load(f)
    all_classes = coverage['classes']

print(f"  Found {len(all_classes)} classes in coverage.yaml (indices 0-{len(all_classes)-1})")

print("\nStep 2: Scanning all label files to find which class IDs were actually used...")
used_class_ids = set()
split_stats = defaultdict(lambda: {'files': 0, 'labels': 0})

for split in ['train', 'valid', 'test']:
    label_dir = os.path.join(SYNTHETIC_BASE, split, 'labels')
    if not os.path.exists(label_dir):
        print(f"  WARNING: {label_dir} doesn't exist")
        continue
    
    label_files = glob.glob(os.path.join(label_dir, '*.txt'))
    split_stats[split]['files'] = len(label_files)
    
    for label_file in label_files:
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    used_class_ids.add(class_id)
                    split_stats[split]['labels'] += 1

print(f"\n  Statistics:")
for split, stats in split_stats.items():
    print(f"    {split}: {stats['files']} files, {stats['labels']} labels")

print(f"\n  Total unique class IDs used: {len(used_class_ids)}")
print(f"  Range: {min(used_class_ids)} to {max(used_class_ids)}")

# Check if any class IDs are out of range
out_of_range = [cid for cid in used_class_ids if cid >= len(all_classes)]
if out_of_range:
    print(f"\n  ERROR: Found class IDs out of range: {out_of_range}")
    print(f"  These IDs are >= {len(all_classes)} (max valid index)")
    exit(1)

print("\nStep 3: Creating corrected files...")

# Update classes.yaml to match coverage.yaml
print(f"  Writing classes.yaml with {len(all_classes)} classes...")
with open(CLASSES_PATH, 'w') as f:
    yaml.dump({'names': all_classes}, f, default_flow_style=False)

# Create data.yaml with correct class list
print(f"  Writing data.yaml with {len(all_classes)} classes...")
data_yaml = {
    'train': os.path.join(SYNTHETIC_BASE, 'train', 'images').replace('\\', '/'),
    'val': os.path.join(SYNTHETIC_BASE, 'valid', 'images').replace('\\', '/'),
    'test': os.path.join(SYNTHETIC_BASE, 'test', 'images').replace('\\', '/'),
    'nc': len(all_classes),
    'names': all_classes,
}

with open(DATA_YAML_PATH, 'w') as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

print("\nStep 4: Verification...")
# Verify the files match
with open(CLASSES_PATH, 'r') as f:
    classes_yaml = yaml.safe_load(f)
with open(DATA_YAML_PATH, 'r') as f:
    data_yaml_check = yaml.safe_load(f)

if classes_yaml['names'] == data_yaml_check['names'] == all_classes:
    print("  ✓ All files synchronized!")
    print(f"  ✓ classes.yaml: {len(classes_yaml['names'])} classes")
    print(f"  ✓ data.yaml: {len(data_yaml_check['names'])} classes (nc={data_yaml_check['nc']})")
    print(f"  ✓ coverage.yaml: {len(all_classes)} classes")
else:
    print("  ✗ ERROR: Files do not match!")

print("\nStep 5: Analyzing which classes were actually used...")
used_classes = sorted([all_classes[i] for i in sorted(used_class_ids)])
unused_classes = sorted([c for i, c in enumerate(all_classes) if i not in used_class_ids])

print(f"  Used: {len(used_classes)} classes")
print(f"  Unused: {len(unused_classes)} classes")

# Save detailed analysis
analysis_path = os.path.join(SYNTHETIC_BASE, 'class_usage_analysis.txt')
with open(analysis_path, 'w') as f:
    f.write(f"Synthetic Data Class Usage Analysis\n")
    f.write(f"=" * 60 + "\n\n")
    f.write(f"Total classes defined: {len(all_classes)}\n")
    f.write(f"Classes actually used: {len(used_classes)}\n")
    f.write(f"Classes not used: {len(unused_classes)}\n\n")
    
    f.write(f"Used Class IDs and Names:\n")
    f.write(f"-" * 60 + "\n")
    for class_id in sorted(used_class_ids):
        f.write(f"{class_id}: {all_classes[class_id]}\n")
    
    f.write(f"\n\nUnused Classes:\n")
    f.write(f"-" * 60 + "\n")
    for class_name in unused_classes:
        class_id = all_classes.index(class_name)
        f.write(f"{class_id}: {class_name}\n")

print(f"\n  Saved detailed analysis to: {analysis_path}")
print("\nDone! Files are now synchronized.")
