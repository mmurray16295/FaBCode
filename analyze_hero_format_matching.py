"""
Analyze how hero selection works across formats and identify the issue
with Young vs Adult hero matching.
"""

import json
from pathlib import Path

# Load card data
with open('data/card.json', 'r', encoding='utf-8') as f:
    all_cards = json.load(f)

# Load weights
with open('data/card_popularity_weights_by_hero.json', 'r', encoding='utf-8') as f:
    weights_data = json.load(f)

# Get all hero cards
hero_cards = [c for c in all_cards if 'Hero' in c.get('types', [])]

print(f"Total hero cards in card.json: {len(hero_cards)}")
print(f"Total heroes in weights file: {len(weights_data['heroes'])}")

# Group heroes by base name
from collections import defaultdict
heroes_by_base_name = defaultdict(list)

for card in hero_cards:
    # Extract base name (remove commas and everything after)
    base_name = card['name'].split(',')[0].strip()
    heroes_by_base_name[base_name].append(card)

# Find heroes with both Young and Adult versions
print("\n" + "="*80)
print("HEROES WITH MULTIPLE VERSIONS:")
print("="*80)

for base_name, cards in sorted(heroes_by_base_name.items()):
    if len(cards) > 1:
        print(f"\n{base_name}: {len(cards)} versions")
        for card in cards:
            is_young = 'Young' in card.get('types', [])
            cc = '✓' if card.get('cc_legal') else '✗'
            ll = '✓' if card.get('ll_legal') else '✗'
            blitz = '✓' if card.get('blitz_legal') else '✗'
            print(f"  {'YOUNG' if is_young else 'ADULT'}: {card['name']}")
            print(f"    CC:{cc} LL:{ll} Blitz:{blitz}")
            print(f"    Types: {card.get('types', [])}")

# Check weights file matching
print("\n" + "="*80)
print("WEIGHTS FILE ANALYSIS:")
print("="*80)

# Check a few key heroes
test_heroes = [
    'ser-boltyn-breaker-of-dawn',
    'rhinar-reckless-rampage', 
    'vynnset-iron-maiden'
]

for hero_key in test_heroes:
    if hero_key in weights_data['heroes']:
        print(f"\n{hero_key}:")
        
        # Simulate matching logic from card_selector.py
        hero_key_normalized = hero_key.lower().replace('-', '').replace(' ', '').replace(',', '').replace("'", '').replace('!', '').replace('/', '')
        
        matches = []
        for card in hero_cards:
            name_normalized = card['name'].lower().replace(' ', '').replace(',', '').replace("'", '').replace('!', '').replace('/', '')
            if hero_key_normalized in name_normalized:
                matches.append(card)
        
        print(f"  Matches found: {len(matches)}")
        for card in matches:
            is_young = 'Young' in card.get('types', [])
            cc = '✓' if card.get('cc_legal') else '✗'
            ll = '✓' if card.get('ll_legal') else '✗'
            blitz = '✓' if card.get('blitz_legal') else '✗'
            print(f"    {'YOUNG' if is_young else 'ADULT'}: {card['name']} - CC:{cc} LL:{ll} Blitz:{blitz}")

# Check which heroes in weights file would fail for Blitz format
print("\n" + "="*80)
print("HEROES IN WEIGHTS FILE THAT ARE NOT BLITZ-LEGAL:")
print("="*80)

not_blitz_legal = []
for hero_key in weights_data['heroes'].keys():
    hero_key_normalized = hero_key.lower().replace('-', '').replace(' ', '').replace(',', '').replace("'", '').replace('!', '').replace('/', '')
    
    # Find first matching card (simulates current logic)
    matched_card = None
    for card in all_cards:
        if 'Hero' in card.get('types', []):
            name_normalized = card['name'].lower().replace(' ', '').replace(',', '').replace("'", '').replace('!', '').replace('/', '')
            if hero_key_normalized in name_normalized:
                matched_card = card
                break
    
    if matched_card and not matched_card.get('blitz_legal'):
        not_blitz_legal.append((hero_key, matched_card['name']))

print(f"\nFound {len(not_blitz_legal)} heroes in weights that match to non-Blitz-legal cards:")
for hero_key, card_name in not_blitz_legal[:20]:  # Show first 20
    print(f"  {hero_key} → {card_name}")

print("\n" + "="*80)
print("SOLUTION:")
print("="*80)
print("""
The issue is that select_random_hero() uses 'hero_key in name_normalized' matching,
which finds the FIRST hero card containing that substring. For heroes with both
Young and Adult versions, it may find the wrong version for the format.

PROPOSED FIX:
1. When format='blitz', prefer Young heroes in matching
2. When format='cc' or 'll', prefer Adult heroes in matching
3. Fall back to any legal version if preferred version not found

This requires modifying the matching logic in select_random_hero() to:
a) Find ALL matching hero cards
b) Filter by format legality
c) Prefer Young/Adult based on format
d) Return best match
""")
