#!/usr/bin/env python3
"""
Create a deduplicated list of cards that SHOULD have appeared in training
but didn't, excluding the ones that legitimately can't be selected.
"""

import json
from collections import defaultdict

def main():
    # Load the deep analysis
    with open('DEEP_MISSING_ANALYSIS.json', 'r') as f:
        analysis = json.load(f)
    
    # Load card database
    with open('data/card.json', 'r') as f:
        card_db = json.load(f)
    
    # Get the 132 cards that legitimately can't be selected
    never_selectable_ids = set(card['class_id'] for card in analysis['never_selectable_cards'])
    
    # Get training data
    from pathlib import Path
    classes_in_training = set()
    labels_dir = Path('data/synthetic/train/labels')
    if labels_dir.exists():
        for label_file in labels_dir.glob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        classes_in_training.add(class_id)
    
    print(f"Classes in training: {len(classes_in_training)}")
    print(f"Never selectable: {len(never_selectable_ids)}")
    
    # Deduplicate cards by name
    name_to_cards = defaultdict(list)
    for idx, card in enumerate(card_db):
        card_name = card.get('name', f'Unknown_{idx}')
        name_to_cards[card_name].append({
            'class_id': idx,
            'card': card
        })
    
    # For each unique card name, pick the first class_id as representative
    unique_card_names = {}
    for card_name, cards in name_to_cards.items():
        cards.sort(key=lambda x: x['class_id'])
        representative = cards[0]
        unique_card_names[card_name] = {
            'class_id': representative['class_id'],
            'card': representative['card'],
            'total_printings': len(cards),
            'all_class_ids': [c['class_id'] for c in cards]
        }
    
    print(f"Unique card names: {len(unique_card_names)}")
    
    # Find cards that should have appeared but didn't
    should_have_appeared = []
    
    for card_name, info in unique_card_names.items():
        class_id = info['class_id']
        card = info['card']
        all_class_ids = info['all_class_ids']
        
        # Skip if any printing appeared in training
        if any(cid in classes_in_training for cid in all_class_ids):
            continue
        
        # Skip if this card legitimately can't be selected
        if class_id in never_selectable_ids:
            continue
        
        # This card should have appeared!
        should_have_appeared.append({
            'name': card_name,
            'class_id': class_id,
            'types': card.get('types', []),
            'color': card.get('color', ''),
            'cost': card.get('cost', ''),
            'pitch': card.get('pitch', ''),
            'power': card.get('power', ''),
            'defense': card.get('defense', ''),
            'total_printings': info['total_printings'],
            'all_class_ids': all_class_ids,
            'type_text': card.get('type_text', '')
        })
    
    # Sort by name
    should_have_appeared.sort(key=lambda x: x['name'])
    
    print(f"\n{'='*80}")
    print(f"CARDS THAT SHOULD HAVE APPEARED (Deduplicated)")
    print(f"{'='*80}")
    print(f"\nTotal unique card names: {len(should_have_appeared)}")
    print(f"(These cards are legal, match hero requirements, but didn't appear)")
    print(f"\n{'='*80}\n")
    
    # Print the list
    for i, card in enumerate(should_have_appeared, 1):
        types_str = ', '.join(card['types'])
        printings_note = f" ({card['total_printings']} printings)" if card['total_printings'] > 1 else ""
        print(f"{i:4d}. {card['name']:45s} | {types_str:40s}{printings_note}")
    
    # Save to file
    with open('CARDS_MISSING_FROM_TRAINING.json', 'w') as f:
        json.dump({
            'summary': {
                'total_unique_cards_missing': len(should_have_appeared),
                'total_printings_missing': sum(c['total_printings'] for c in should_have_appeared),
                'explanation': 'These cards are legal in their formats and match hero class/talent requirements, but did not appear in the 125K training images due to randomness or insufficient generation volume.'
            },
            'missing_cards': should_have_appeared
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✅ Saved detailed list to: CARDS_MISSING_FROM_TRAINING.json")
    print(f"{'='*80}\n")
    
    # Statistics by type
    type_counter = defaultdict(int)
    for card in should_have_appeared:
        for card_type in card['types']:
            type_counter[card_type] += 1
    
    print(f"\nMissing cards by type:")
    for card_type, count in sorted(type_counter.items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"  {card_type:20s}: {count:4d} cards")

if __name__ == '__main__':
    main()
