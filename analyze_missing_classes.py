#!/usr/bin/env python3
"""
Analyze Missing Training Classes - Find Programming Bugs in Card Selection

This script identifies which of the 2,641 intended classes are missing from training
and analyzes card.json to find patterns that explain WHY they were excluded.
"""

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

def load_card_database(card_json_path):
    """Load the complete card database"""
    with open(card_json_path, 'r') as f:
        return json.load(f)

def get_training_classes(data_dir):
    """Get set of class IDs that actually appear in training data"""
    classes_in_training = set()
    
    labels_dir = Path(data_dir) / 'train' / 'labels'
    if not labels_dir.exists():
        print(f"ERROR: {labels_dir} not found")
        return classes_in_training
    
    for label_file in labels_dir.glob('*.txt'):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    classes_in_training.add(class_id)
    
    return classes_in_training

def deduplicate_cards_by_name(card_db):
    """
    Deduplicate cards by name (combining all art variants)
    Returns: dict mapping class_id -> representative card info
    """
    name_to_cards = defaultdict(list)
    
    # Group cards by name
    for idx, card in enumerate(card_db):
        card_name = card.get('name', f'Unknown_{idx}')
        name_to_cards[card_name].append({
            'class_id': idx,
            'card': card
        })
    
    # Pick first occurrence of each name as the representative
    class_mapping = {}
    for card_name, cards in name_to_cards.items():
        # Sort by class_id to be deterministic
        cards.sort(key=lambda x: x['class_id'])
        representative = cards[0]
        class_mapping[representative['class_id']] = {
            'name': card_name,
            'card': representative['card'],
            'variant_count': len(cards),
            'all_class_ids': [c['class_id'] for c in cards]
        }
    
    return class_mapping

def analyze_missing_patterns(missing_classes, class_mapping):
    """Analyze patterns in missing classes to identify programming bugs"""
    
    patterns = {
        'split_class': [],  # Cards with "/" in class (e.g., "Brute / Guardian")
        'dual_talent': [],  # Cards with multiple talents
        'no_class': [],     # Cards with no class specified
        'no_talent': [],    # Cards with no talent specified
        'generic': [],      # Generic cards
        'token': [],        # Token cards
        'other': []
    }
    
    for class_id in missing_classes:
        if class_id not in class_mapping:
            continue
            
        card_info = class_mapping[class_id]
        card = card_info['card']
        card_name = card_info['name']
        
        # Check for split class (Brute / Guardian)
        card_class = card.get('class', '')
        if '/' in card_class or ' / ' in card_class:
            patterns['split_class'].append({
                'class_id': class_id,
                'name': card_name,
                'class': card_class,
                'talent': card.get('talent', 'N/A')
            })
        # Check for dual talent
        elif card.get('talent') and ('/' in card.get('talent', '') or len(card.get('talent', '').split()) > 1):
            patterns['dual_talent'].append({
                'class_id': class_id,
                'name': card_name,
                'class': card_class,
                'talent': card.get('talent', 'N/A')
            })
        # Check for no class
        elif not card_class or card_class.strip() == '':
            patterns['no_class'].append({
                'class_id': class_id,
                'name': card_name,
                'type': card.get('type', 'N/A'),
                'talent': card.get('talent', 'N/A')
            })
        # Check for generic
        elif card_class.lower() == 'generic':
            patterns['generic'].append({
                'class_id': class_id,
                'name': card_name,
                'type': card.get('type', 'N/A')
            })
        # Check for token
        elif card.get('type', '').lower() == 'token':
            patterns['token'].append({
                'class_id': class_id,
                'name': card_name,
                'class': card_class
            })
        # No talent
        elif not card.get('talent'):
            patterns['no_talent'].append({
                'class_id': class_id,
                'name': card_name,
                'class': card_class,
                'type': card.get('type', 'N/A')
            })
        else:
            patterns['other'].append({
                'class_id': class_id,
                'name': card_name,
                'class': card_class,
                'talent': card.get('talent', 'N/A'),
                'type': card.get('type', 'N/A')
            })
    
    return patterns

def main():
    # Paths
    card_json = 'data/card.json'
    data_dir = 'data/synthetic'
    
    print("="*80)
    print("MISSING CLASS ANALYSIS - Finding Programming Bugs")
    print("="*80)
    
    print("\n[1/4] Loading card database...")
    card_db = load_card_database(card_json)
    print(f"  Total cards in database: {len(card_db)}")
    
    print("\n[2/4] Deduplicating cards by name...")
    class_mapping = deduplicate_cards_by_name(card_db)
    intended_classes = len(class_mapping)
    print(f"  Deduplicated classes: {intended_classes}")
    print(f"  (Combined {len(card_db) - intended_classes} duplicate printings)")
    
    print("\n[3/4] Scanning training data...")
    classes_in_training = get_training_classes(data_dir)
    print(f"  Classes found in training: {len(classes_in_training)}")
    
    # Find missing classes
    all_intended = set(class_mapping.keys())
    missing_classes = all_intended - classes_in_training
    
    print(f"\n{'='*80}")
    print(f"COVERAGE SUMMARY")
    print(f"{'='*80}")
    print(f"  Intended classes (deduplicated): {intended_classes}")
    print(f"  Classes with training data: {len(classes_in_training)}")
    print(f"  Missing classes: {len(missing_classes)}")
    print(f"  Coverage: {len(classes_in_training)/intended_classes*100:.1f}%")
    
    if len(missing_classes) == 0:
        print("\n✅ Perfect coverage! All intended classes have training data.")
        return
    
    print(f"\n[4/4] Analyzing patterns in {len(missing_classes)} missing classes...")
    patterns = analyze_missing_patterns(missing_classes, class_mapping)
    
    print(f"\n{'='*80}")
    print(f"PATTERN ANALYSIS - Identifying Programming Bugs")
    print(f"{'='*80}")
    
    # Split class issue (Brute / Guardian, etc.)
    if patterns['split_class']:
        print(f"\n🔴 SPLIT CLASS BUG ({len(patterns['split_class'])} cards)")
        print(f"   Cards with '/' in class field (e.g., 'Brute / Guardian')")
        print(f"   These need OR logic, not AND logic for matching")
        print(f"\n   Examples:")
        for i, card in enumerate(patterns['split_class'][:10], 1):
            print(f"   {i:2d}. [{card['class_id']:4d}] {card['name']:40s} | Class: {card['class']:20s} | Talent: {card['talent']}")
        if len(patterns['split_class']) > 10:
            print(f"   ... and {len(patterns['split_class']) - 10} more")
    
    # Dual talent issue
    if patterns['dual_talent']:
        print(f"\n🟡 DUAL TALENT ({len(patterns['dual_talent'])} cards)")
        print(f"   Cards with multiple talents or '/' in talent field")
        print(f"\n   Examples:")
        for i, card in enumerate(patterns['dual_talent'][:10], 1):
            print(f"   {i:2d}. [{card['class_id']:4d}] {card['name']:40s} | Class: {card['class']:20s} | Talent: {card['talent']}")
        if len(patterns['dual_talent']) > 10:
            print(f"   ... and {len(patterns['dual_talent']) - 10} more")
    
    # No class
    if patterns['no_class']:
        print(f"\n🟠 NO CLASS SPECIFIED ({len(patterns['no_class'])} cards)")
        print(f"   Cards without a class field (might be generic/token)")
        print(f"\n   Examples:")
        for i, card in enumerate(patterns['no_class'][:10], 1):
            print(f"   {i:2d}. [{card['class_id']:4d}] {card['name']:40s} | Type: {card['type']:15s} | Talent: {card['talent']}")
        if len(patterns['no_class']) > 10:
            print(f"   ... and {len(patterns['no_class']) - 10} more")
    
    # Generic cards
    if patterns['generic']:
        print(f"\n⚪ GENERIC CARDS ({len(patterns['generic'])} cards)")
        print(f"   Cards marked as 'Generic' class")
        print(f"\n   Examples:")
        for i, card in enumerate(patterns['generic'][:10], 1):
            print(f"   {i:2d}. [{card['class_id']:4d}] {card['name']:40s} | Type: {card['type']}")
        if len(patterns['generic']) > 10:
            print(f"   ... and {len(patterns['generic']) - 10} more")
    
    # Tokens
    if patterns['token']:
        print(f"\n🔵 TOKEN CARDS ({len(patterns['token'])} cards)")
        print(f"   Cards with type='Token'")
        print(f"\n   Examples:")
        for i, card in enumerate(patterns['token'][:10], 1):
            print(f"   {i:2d}. [{card['class_id']:4d}] {card['name']:40s} | Class: {card['class']}")
        if len(patterns['token']) > 10:
            print(f"   ... and {len(patterns['token']) - 10} more")
    
    # No talent
    if patterns['no_talent']:
        print(f"\n🟤 NO TALENT SPECIFIED ({len(patterns['no_talent'])} cards)")
        print(f"   Cards without talent field")
        print(f"\n   Examples:")
        for i, card in enumerate(patterns['no_talent'][:10], 1):
            print(f"   {i:2d}. [{card['class_id']:4d}] {card['name']:40s} | Class: {card['class']:20s} | Type: {card['type']}")
        if len(patterns['no_talent']) > 10:
            print(f"   ... and {len(patterns['no_talent']) - 10} more")
    
    # Other
    if patterns['other']:
        print(f"\n⚫ OTHER/UNKNOWN ({len(patterns['other'])} cards)")
        print(f"\n   Examples:")
        for i, card in enumerate(patterns['other'][:10], 1):
            print(f"   {i:2d}. [{card['class_id']:4d}] {card['name']:40s} | Class: {card['class']:20s} | Talent: {card['talent']}")
        if len(patterns['other']) > 10:
            print(f"   ... and {len(patterns['other']) - 10} more")
    
    print(f"\n{'='*80}")
    print(f"SUMMARY OF BUGS FOUND")
    print(f"{'='*80}")
    
    total_identified = sum(len(p) for p in patterns.values())
    print(f"\n  Split class (OR logic needed): {len(patterns['split_class'])} cards")
    print(f"  Dual talent issues: {len(patterns['dual_talent'])} cards")
    print(f"  No class specified: {len(patterns['no_class'])} cards")
    print(f"  Generic cards: {len(patterns['generic'])} cards")
    print(f"  Token cards: {len(patterns['token'])} cards")
    print(f"  No talent: {len(patterns['no_talent'])} cards")
    print(f"  Other/Unknown: {len(patterns['other'])} cards")
    print(f"\n  Total categorized: {total_identified} / {len(missing_classes)}")
    
    # Save detailed report
    report = {
        'summary': {
            'intended_classes': intended_classes,
            'classes_with_data': len(classes_in_training),
            'missing_classes': len(missing_classes),
            'coverage_percent': len(classes_in_training)/intended_classes*100
        },
        'patterns': patterns,
        'missing_class_ids': sorted(list(missing_classes))
    }
    
    with open('MISSING_CLASSES_ANALYSIS.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Detailed report saved to: MISSING_CLASSES_ANALYSIS.json")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
