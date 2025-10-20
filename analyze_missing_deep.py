#!/usr/bin/env python3
"""
Deep analysis of missing cards with FIXED card selector logic.
This will show us what other bugs exist beyond the split-class issue.
"""

import json
import sys
from collections import defaultdict, Counter
from pathlib import Path

# Add scripts to path
sys.path.insert(0, 'scripts')
from card_selector import CardSelector, ALL_CLASSES, ALL_TALENTS, is_card_legal_for_format

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
    """Deduplicate cards by name (combining all art variants)"""
    name_to_cards = defaultdict(list)
    
    for idx, card in enumerate(card_db):
        card_name = card.get('name', f'Unknown_{idx}')
        name_to_cards[card_name].append({
            'class_id': idx,
            'card': card
        })
    
    class_mapping = {}
    for card_name, cards in name_to_cards.items():
        cards.sort(key=lambda x: x['class_id'])
        representative = cards[0]
        class_mapping[representative['class_id']] = {
            'name': card_name,
            'card': representative['card'],
            'variant_count': len(cards),
            'all_class_ids': [c['class_id'] for c in cards]
        }
    
    return class_mapping

def analyze_card_selection_logic(card, hero_classes, hero_talents, format):
    """
    Simulate the FIXED card selector logic and explain WHY a card was excluded.
    Returns (should_include, reason)
    """
    
    # Check format legality
    if not is_card_legal_for_format(card, format):
        return False, f"Not legal in {format}"
    
    # Check if it's a hero
    card_types = set(card.get('types', []))
    if 'Hero' in card_types:
        return False, "Is a hero card"
    
    # Check if it's generic
    if 'Generic' in card_types:
        return True, "Generic (should be included)"
    
    # Extract card classes and talents
    card_classes = card_types & ALL_CLASSES
    card_talents = card_types & ALL_TALENTS
    
    # If hero has no talents, exclude cards with ANY talent
    if not hero_talents and card_talents:
        return False, f"Card has talents {card_talents} but hero has no talents"
    
    # Check class matching (FIXED - now uses intersection)
    card_classes_match = not card_classes or bool(card_classes & hero_classes)
    if not card_classes_match:
        return False, f"Card classes {card_classes} don't match hero classes {hero_classes}"
    
    # Check talent matching (FIXED - now uses intersection)
    card_talents_match = not card_talents or bool(card_talents & hero_talents)
    if not card_talents_match:
        return False, f"Card talents {card_talents} don't match hero talents {hero_talents}"
    
    return True, "Should be included"

def main():
    print("="*80)
    print("DEEP MISSING CARD ANALYSIS (with FIXED logic)")
    print("="*80)
    
    # Load data
    print("\n[1/5] Loading card database...")
    with open('data/card.json', 'r') as f:
        card_db = json.load(f)
    
    print(f"  Total cards: {len(card_db)}")
    
    print("\n[2/5] Deduplicating by name...")
    class_mapping = deduplicate_cards_by_name(card_db)
    print(f"  Deduplicated classes: {len(class_mapping)}")
    
    print("\n[3/5] Scanning training data...")
    classes_in_training = get_training_classes('data/synthetic')
    print(f"  Classes in training: {len(classes_in_training)}")
    
    # Find missing
    all_intended = set(class_mapping.keys())
    missing_classes = all_intended - classes_in_training
    print(f"  Missing classes: {len(missing_classes)}")
    
    print("\n[4/5] Loading card selector with weights...")
    selector = CardSelector()
    
    print("\n[5/5] Analyzing missing cards with FIXED logic...")
    
    # Test against all heroes to see if card COULD be selected
    exclusion_reasons = defaultdict(list)
    never_selectable = []  # Cards that can't be selected by ANY hero in ANY format
    
    formats = ['cc', 'll', 'blitz']
    
    for class_id in sorted(missing_classes):
        if class_id not in class_mapping:
            continue
        
        card_info = class_mapping[class_id]
        card = card_info['card']
        card_name = card_info['name']
        
        # Check if this card could EVER be selected by ANY hero in ANY format
        selectable_by_any = False
        reasons_across_heroes = []
        
        for format in formats:
            # Try all heroes
            for hero_key, hero_weights in selector.weights_data.get('heroes', {}).items():
                # Get hero card
                hero_card = None
                for c in card_db:
                    if 'Hero' in c.get('types', []) and hero_key.lower().replace('-', '').replace(' ', '') in c['name'].lower().replace(' ', '').replace(',', '').replace("'", ''):
                        hero_card = c
                        break
                
                if not hero_card:
                    continue
                
                # Get hero classes/talents
                hero_types = set(hero_card.get('types', []))
                hero_classes = hero_types & ALL_CLASSES
                hero_talents = hero_types & ALL_TALENTS
                
                # Check if card would be selected
                should_include, reason = analyze_card_selection_logic(card, hero_classes, hero_talents, format)
                
                if should_include:
                    selectable_by_any = True
                    break
                else:
                    reasons_across_heroes.append(reason)
            
            if selectable_by_any:
                break
        
        if not selectable_by_any:
            # Get the most common reason
            reason_counter = Counter(reasons_across_heroes)
            most_common_reason = reason_counter.most_common(1)[0][0] if reason_counter else "Unknown"
            
            never_selectable.append({
                'class_id': class_id,
                'name': card_name,
                'reason': most_common_reason,
                'types': card.get('types', [])
            })
            
            exclusion_reasons[most_common_reason].append(card_name)
    
    # Print results
    print(f"\n{'='*80}")
    print(f"ANALYSIS RESULTS")
    print(f"{'='*80}")
    
    print(f"\n📊 Summary:")
    print(f"  Total missing classes: {len(missing_classes)}")
    print(f"  Cards that can't be selected by ANY hero: {len(never_selectable)}")
    print(f"  Cards that COULD be selected but weren't: {len(missing_classes) - len(never_selectable)}")
    
    print(f"\n{'='*80}")
    print(f"EXCLUSION REASONS")
    print(f"{'='*80}")
    
    for reason, cards in sorted(exclusion_reasons.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"\n🔴 {reason} ({len(cards)} cards)")
        for i, card_name in enumerate(sorted(cards)[:20], 1):
            print(f"   {i:3d}. {card_name}")
        if len(cards) > 20:
            print(f"   ... and {len(cards) - 20} more")
    
    # Detailed breakdown
    print(f"\n{'='*80}")
    print(f"DETAILED BREAKDOWN (First 30 never-selectable cards)")
    print(f"{'='*80}")
    
    for i, card_info in enumerate(never_selectable[:30], 1):
        print(f"\n{i:3d}. [{card_info['class_id']:4d}] {card_info['name']}")
        print(f"     Types: {', '.join(card_info['types'])}")
        print(f"     Reason: {card_info['reason']}")
    
    if len(never_selectable) > 30:
        print(f"\n... and {len(never_selectable) - 30} more cards that can't be selected by any hero")
    
    # Save report
    report = {
        'summary': {
            'total_missing': len(missing_classes),
            'never_selectable': len(never_selectable),
            'could_be_selected_but_werent': len(missing_classes) - len(never_selectable)
        },
        'exclusion_reasons': {reason: len(cards) for reason, cards in exclusion_reasons.items()},
        'never_selectable_cards': never_selectable
    }
    
    with open('DEEP_MISSING_ANALYSIS.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Detailed report saved to: DEEP_MISSING_ANALYSIS.json")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
