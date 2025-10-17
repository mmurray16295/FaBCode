#!/usr/bin/env python3
"""
Count occurrences of each card class in the generated dataset.
Parses YOLO label files to count how many times each class appears.
"""

import os
import json
from collections import Counter
from pathlib import Path

def load_class_mapping():
    """Load the class ID to card name mapping."""
    mapping_path = Path('/root/FaBCode/data/card_name_to_class_id.json')
    with open(mapping_path, 'r') as f:
        card_to_id = json.load(f)
    
    # Reverse mapping: class_id -> card_name
    id_to_card = {v: k for k, v in card_to_id.items()}
    return id_to_card

def count_classes_in_labels(label_dir):
    """Count class occurrences in YOLO label files."""
    class_counter = Counter()
    
    label_files = list(Path(label_dir).rglob('*.txt'))
    
    for label_file in label_files:
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    class_counter[class_id] += 1
    
    return class_counter

def main():
    print("=" * 80)
    print("CARD OCCURRENCE COUNTER")
    print("=" * 80)
    
    # Load class mapping
    print("\nLoading class mapping...")
    id_to_card = load_class_mapping()
    total_classes = len(id_to_card)
    print(f"✓ Loaded {total_classes} card classes")
    
    # Count occurrences in all splits
    synthetic_dir = Path('/root/FaBCode/data/synthetic')
    
    print("\nCounting card occurrences in labels...")
    all_class_counts = Counter()
    
    for split in ['train', 'valid', 'test']:
        labels_dir = synthetic_dir / split / 'labels'
        if labels_dir.exists():
            split_counts = count_classes_in_labels(labels_dir)
            all_class_counts.update(split_counts)
            print(f"  {split:6s}: {sum(split_counts.values()):,} card instances in {len(list(labels_dir.glob('*.txt')))} images")
    
    total_instances = sum(all_class_counts.values())
    classes_seen = len(all_class_counts)
    classes_not_seen = total_classes - classes_seen
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total card instances:        {total_instances:,}")
    print(f"Unique classes seen:         {classes_seen:,} / {total_classes:,} ({100*classes_seen/total_classes:.1f}%)")
    print(f"Classes not yet seen:        {classes_not_seen:,}")
    print(f"Average instances per class: {total_instances/classes_seen:.1f}")
    
    # Show most and least common cards
    if all_class_counts:
        print("\n" + "=" * 80)
        print("TOP 20 MOST COMMON CARDS")
        print("=" * 80)
        for class_id, count in all_class_counts.most_common(20):
            card_name = id_to_card.get(class_id, f"Unknown (ID {class_id})")
            print(f"  {count:4d}x  {card_name}")
        
        print("\n" + "=" * 80)
        print("TOP 20 LEAST COMMON CARDS (that have appeared)")
        print("=" * 80)
        for class_id, count in all_class_counts.most_common()[-20:]:
            card_name = id_to_card.get(class_id, f"Unknown (ID {class_id})")
            print(f"  {count:4d}x  {card_name}")
        
        # Show some cards not yet seen
        classes_not_seen_list = [cid for cid in id_to_card.keys() if cid not in all_class_counts]
        if classes_not_seen_list:
            print("\n" + "=" * 80)
            print(f"SAMPLE OF CARDS NOT YET SEEN ({len(classes_not_seen_list)} total)")
            print("=" * 80)
            for class_id in sorted(classes_not_seen_list)[:20]:
                card_name = id_to_card[class_id]
                print(f"  ID {class_id:4d}:  {card_name}")
    
    # Distribution statistics
    if all_class_counts:
        counts_list = list(all_class_counts.values())
        counts_list.sort()
        
        print("\n" + "=" * 80)
        print("DISTRIBUTION STATISTICS (for cards that have appeared)")
        print("=" * 80)
        print(f"Minimum occurrences:  {min(counts_list)}")
        print(f"Maximum occurrences:  {max(counts_list)}")
        print(f"Median occurrences:   {counts_list[len(counts_list)//2]}")
        print(f"Mean occurrences:     {sum(counts_list)/len(counts_list):.1f}")
        
        # Show distribution buckets
        print("\nOccurrence Distribution:")
        # First show cards not yet seen (0 occurrences)
        print(f"    0 occurrences:       {classes_not_seen:4d} classes (not yet seen)")
        
        buckets = [1, 5, 10, 20, 50, 100, 200, 500]
        for i, bucket in enumerate(buckets):
            if i == 0:
                count = sum(1 for c in counts_list if c == 1)
                print(f"    1 occurrence:        {count:4d} classes")
                count = sum(1 for c in counts_list if 1 < c < bucket)
                if bucket > 2:
                    print(f"    2-  {bucket-1:1d} occurrences:    {count:4d} classes")
            else:
                prev_bucket = buckets[i-1]
                count = sum(1 for c in counts_list if prev_bucket <= c < bucket)
                print(f"  {prev_bucket:3d}-{bucket-1:3d} occurrences:    {count:4d} classes")
        
        count = sum(1 for c in counts_list if c >= buckets[-1])
        print(f"  {buckets[-1]}+ occurrences:       {count:4d} classes")

if __name__ == '__main__':
    main()
