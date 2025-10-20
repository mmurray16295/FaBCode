#!/usr/bin/env python3
"""
Add missing CC/LL legal heroes to the weights file with minimal placeholder data.
For heroes with no deck data on fabrec.gg, we'll create minimal entries so they can still be selected.
"""

import json
from pathlib import Path

def main():
    # Load files
    card_json_path = Path('data/card.json')
    weights_path = Path('data/card_popularity_weights_by_hero.json')
    
    with open(card_json_path, 'r') as f:
        cards = json.load(f)
    
    with open(weights_path, 'r') as f:
        weights = json.load(f)
    
    # Find adult CC/LL legal heroes not in weights
    weight_hero_keys = set(weights['heroes'].keys())
    
    missing_heroes = []
    for card in cards:
        if 'Hero' in card.get('types', []) and 'Young' not in card.get('types', []):
            if card.get('cc_legal') or card.get('ll_legal'):
                # Normalize hero name to key format
                hero_key = card['name'].lower().replace(' ', '-').replace(',', '').replace("'", '').replace('!', '').replace('/', '-')
                
                # Check if already in weights
                found = False
                for weight_key in weight_hero_keys:
                    weight_key_norm = weight_key.replace('-', '')
                    hero_key_norm = hero_key.replace('-', '')
                    if weight_key_norm in hero_key_norm or hero_key_norm in weight_key_norm:
                        found = True
                        break
                
                if not found:
                    missing_heroes.append((hero_key, card))
    
    print(f"Found {len(missing_heroes)} missing CC/LL legal heroes:")
    for hero_key, card in missing_heroes:
        print(f"  {hero_key:50s} {card['name']}")
    
    if not missing_heroes:
        print("\nNo missing heroes - weights file is complete!")
        return
    
    # Create minimal entries for missing heroes
    print(f"\nAdding {len(missing_heroes)} heroes to weights file...")
    
    for hero_key, hero_card in missing_heroes:
        # Create a minimal entry with placeholder data
        # Use very low deck percentage (0.0001%) so they're rarely selected but still available
        weights['heroes'][hero_key] = {
            'hero_deck_percentage': 0.0001,  # Minimal percentage
            'total_unique_cards': 0,
            'sections': {
                'equipment': [],
                'weapon': [],
                'maindeck': []
            }
        }
        print(f"  Added: {hero_key}")
    
    # Update metadata
    weights['metadata']['total_heroes'] = len(weights['heroes'])
    weights['metadata']['description'] += ' Missing heroes added with minimal placeholder data.'
    
    # Save updated weights
    with open(weights_path, 'w') as f:
        json.dump(weights, f, indent=2)
    
    print(f"\n✓ Updated weights file saved to: {weights_path}")
    print(f"✓ Total heroes now: {len(weights['heroes'])}")

if __name__ == '__main__':
    main()
