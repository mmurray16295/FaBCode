#!/usr/bin/env python3
"""
Analyze Training Data Coverage
Identifies cards missing or underrepresented in training dataset
"""

import json
import os
from collections import Counter
from pathlib import Path

def load_card_database(card_json_path):
    """Load the complete card database"""
    with open(card_json_path, 'r') as f:
        return json.load(f)

def analyze_dataset_coverage(data_dir, card_db):
    """Analyze which cards are in train/val/test splits"""
    
    splits = ['train', 'val', 'test']
    card_counts = {split: Counter() for split in splits}
    
    for split in splits:
        labels_dir = Path(data_dir) / split / 'labels'
        
        if not labels_dir.exists():
            print(f"Warning: {labels_dir} not found")
            continue
        
        # Count occurrences of each class in labels
        for label_file in labels_dir.glob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        card_counts[split][class_id] += 1
    
    return card_counts

def generate_report(card_db, card_counts):
    """Generate comprehensive coverage report"""
    
    # Get all class IDs from card database
    all_classes = set(range(len(card_db)))
    
    # Get classes that appear in training
    train_classes = set(card_counts['train'].keys())
    val_classes = set(card_counts['val'].keys())
    test_classes = set(card_counts['test'].keys())
    all_present_classes = train_classes | val_classes | test_classes
    
    # Identify missing cards
    missing_classes = all_classes - all_present_classes
    
    print("\n" + "="*70)
    print("FaB CARD TRAINING COVERAGE ANALYSIS")
    print("="*70)
    
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"  Total cards in database: {len(card_db)}")
    print(f"  Cards with training data: {len(all_present_classes)}")
    print(f"  Cards MISSING from training: {len(missing_classes)}")
    print(f"  Coverage: {len(all_present_classes)/len(card_db)*100:.1f}%")
    
    # Split statistics
    print(f"\n📁 SPLIT COVERAGE:")
    print(f"  Training set: {len(train_classes)} cards")
    print(f"  Validation set: {len(val_classes)} cards")
    print(f"  Test set: {len(test_classes)} cards")
    
    # Missing cards detail
    if missing_classes:
        print(f"\n⚠️  MISSING CARDS ({len(missing_classes)} total):")
        missing_list = sorted(missing_classes)
        for i, class_id in enumerate(missing_list[:50], 1):  # Show first 50
            card_name = card_db[class_id]['name'] if class_id < len(card_db) else f"Unknown (ID {class_id})"
            print(f"  {i:3d}. Class {class_id:4d}: {card_name}")
        
        if len(missing_list) > 50:
            print(f"  ... and {len(missing_list) - 50} more")
    
    # Underrepresented cards (< 10 examples in training)
    underrep_threshold = 10
    underrep_cards = [(class_id, count) for class_id, count in card_counts['train'].items() 
                      if count < underrep_threshold]
    underrep_cards.sort(key=lambda x: x[1])  # Sort by count
    
    if underrep_cards:
        print(f"\n⚡ UNDERREPRESENTED CARDS (< {underrep_threshold} training examples):")
        print(f"  Total: {len(underrep_cards)} cards")
        print(f"\n  Bottom 20 cards by training count:")
        for i, (class_id, count) in enumerate(underrep_cards[:20], 1):
            card_name = card_db[class_id]['name'] if class_id < len(card_db) else f"Unknown (ID {class_id})"
            print(f"  {i:3d}. Class {class_id:4d}: {card_name:40s} ({count:2d} examples)")
    
    # Well-represented cards (top 20)
    top_cards = sorted(card_counts['train'].items(), key=lambda x: x[1], reverse=True)[:20]
    print(f"\n✅ BEST REPRESENTED CARDS (Top 20):")
    for i, (class_id, count) in enumerate(top_cards, 1):
        card_name = card_db[class_id]['name'] if class_id < len(card_db) else f"Unknown (ID {class_id})"
        print(f"  {i:3d}. Class {class_id:4d}: {card_name:40s} ({count:4d} examples)")
    
    # Training distribution stats
    train_counts_list = list(card_counts['train'].values())
    if train_counts_list:
        print(f"\n📈 TRAINING DATA DISTRIBUTION:")
        print(f"  Average examples per card: {sum(train_counts_list)/len(train_counts_list):.1f}")
        print(f"  Median examples per card: {sorted(train_counts_list)[len(train_counts_list)//2]}")
        print(f"  Min examples: {min(train_counts_list)}")
        print(f"  Max examples: {max(train_counts_list)}")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS:")
    print("="*70)
    
    if missing_classes:
        print(f"\n1. CRITICAL: Generate training data for {len(missing_classes)} missing cards")
        print(f"   These cards have 0% detection capability")
    
    if underrep_cards:
        print(f"\n2. IMPORTANT: Augment data for {len(underrep_cards)} underrepresented cards")
        print(f"   Target: At least {underrep_threshold} examples per card")
        print(f"   Focus on bottom 100 cards first")
    
    print(f"\n3. BALANCED DATASET TARGET:")
    print(f"   Current average: {sum(train_counts_list)/len(train_counts_list):.1f} examples/card")
    print(f"   Recommended: 30-50 examples per card minimum")
    print(f"   This would require ~{len(all_classes) * 40 - sum(train_counts_list):,} additional images")
    
    print("\n" + "="*70 + "\n")
    
    # Save detailed report
    return {
        'missing_cards': sorted(missing_classes),
        'underrepresented_cards': [(cid, cnt) for cid, cnt in underrep_cards],
        'statistics': {
            'total_cards': len(card_db),
            'cards_with_data': len(all_present_classes),
            'coverage_percent': len(all_present_classes)/len(card_db)*100,
            'missing_count': len(missing_classes),
            'underrepresented_count': len(underrep_cards)
        }
    }

def save_report(report_data, output_path):
    """Save detailed report to JSON"""
    with open(output_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    print(f"✅ Detailed report saved to: {output_path}")

if __name__ == '__main__':
    # Paths
    card_json = 'data/card.json'
    data_dir = 'data/synthetic'
    output_report = 'TRAINING_COVERAGE_REPORT.json'
    
    print("Loading card database...")
    card_db = load_card_database(card_json)
    
    print(f"Analyzing training data in {data_dir}...")
    card_counts = analyze_dataset_coverage(data_dir, card_db)
    
    print("Generating report...")
    report_data = generate_report(card_db, card_counts)
    
    save_report(report_data, output_report)
