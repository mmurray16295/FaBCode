"""
Investigate why 57 talent cards show as "talent_mismatch" when they should be selectable.
Check if heroes with Earth/Ice/Lightning/Shadow talents are in the weights file.
"""

import json
from pathlib import Path

# Load card data
with open('data/card.json', 'r', encoding='utf-8') as f:
    all_cards = json.load(f)

# Load weights
with open('data/card_popularity_weights_by_hero.json', 'r', encoding='utf-8') as f:
    weights_data = json.load(f)

# Load the "unfixable" cards
with open('CARDS_CANNOT_BE_SELECTED.json', 'r', encoding='utf-8') as f:
    unfixable = json.load(f)

print("="*80)
print("INVESTIGATING TALENT MISMATCH ISSUE")
print("="*80)

# Get all talent mismatch cards
talent_mismatch_cards = unfixable['detailed']['talent_mismatch']

# Group by talent
from collections import defaultdict
cards_by_talent = defaultdict(list)

for card in talent_mismatch_cards:
    types = card['types']
    talents = []
    for t in types:
        if t in ['Earth', 'Ice', 'Lightning', 'Shadow']:
            talents.append(t)
    
    for talent in talents:
        cards_by_talent[talent].append(card['name'])

print("\nCards grouped by talent:")
for talent, cards in sorted(cards_by_talent.items()):
    print(f"\n{talent}: {len(cards)} cards")
    for card in cards[:5]:
        print(f"  - {card}")
    if len(cards) > 5:
        print(f"  ... and {len(cards) - 5} more")

# Now check which heroes have these talents
print("\n" + "="*80)
print("HEROES WITH ELEMENTAL TALENTS IN CARD.JSON:")
print("="*80)

hero_cards = [c for c in all_cards if 'Hero' in c.get('types', [])]

talents_to_check = ['Earth', 'Ice', 'Lightning', 'Shadow']
heroes_by_talent = defaultdict(list)

for hero in hero_cards:
    hero_types = set(hero.get('types', []))
    hero_talents = hero_types & set(talents_to_check)
    
    if hero_talents:
        for talent in hero_talents:
            heroes_by_talent[talent].append({
                'name': hero['name'],
                'is_young': 'Young' in hero.get('types', []),
                'cc_legal': hero.get('cc_legal'),
                'blitz_legal': hero.get('blitz_legal')
            })

for talent in sorted(talents_to_check):
    print(f"\n{talent} heroes:")
    if talent in heroes_by_talent:
        for hero in heroes_by_talent[talent]:
            version = 'YOUNG' if hero['is_young'] else 'ADULT'
            cc = '✓' if hero['cc_legal'] else '✗'
            blitz = '✓' if hero['blitz_legal'] else '✗'
            print(f"  {version}: {hero['name']} (CC:{cc} Blitz:{blitz})")
    else:
        print(f"  NO HEROES FOUND!")

# Check if these heroes are in the weights file
print("\n" + "="*80)
print("ARE THESE HEROES IN THE WEIGHTS FILE?")
print("="*80)

weights_hero_keys = set(weights_data['heroes'].keys())

for talent in sorted(talents_to_check):
    print(f"\n{talent} heroes:")
    if talent in heroes_by_talent:
        for hero in heroes_by_talent[talent]:
            # Normalize hero name to match weights format
            hero_normalized = hero['name'].lower().replace(' ', '-').replace(',', '').replace("'", '')
            
            # Check if any weights key matches
            found = False
            for key in weights_hero_keys:
                if hero_normalized in key or key in hero_normalized:
                    found = True
                    print(f"  ✓ FOUND: {hero['name']} → {key}")
                    break
            
            if not found:
                print(f"  ✗ MISSING: {hero['name']}")
    else:
        print(f"  N/A - No heroes with this talent exist")

# Check heroes with "Elemental" type (they should have talents via Essence)
print("\n" + "="*80)
print("ELEMENTAL HEROES (should have talents via Essence):")
print("="*80)

elemental_heroes = [h for h in hero_cards if 'Elemental' in h.get('types', [])]
print(f"\nFound {len(elemental_heroes)} Elemental heroes in card.json:")

for hero in elemental_heroes[:10]:
    is_young = 'Young' in hero.get('types', [])
    version = 'YOUNG' if is_young else 'ADULT'
    cc = '✓' if hero.get('cc_legal') else '✗'
    talents = [t for t in hero.get('types', []) if t in talents_to_check]
    keywords = hero.get('card_keywords', [])
    essence = [k for k in keywords if 'Essence' in k]
    
    print(f"\n{version}: {hero['name']} (CC:{cc})")
    print(f"  Types: {hero.get('types', [])}")
    print(f"  Talents in types: {talents if talents else 'NONE'}")
    print(f"  Essence keywords: {essence if essence else 'NONE'}")
    
    # Check if in weights
    hero_normalized = hero['name'].lower().replace(' ', '-').replace(',', '').replace("'", '')
    in_weights = any(hero_normalized in key or key in hero_normalized for key in weights_hero_keys)
    print(f"  In weights: {'✓' if in_weights else '✗'}")

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)
print("""
The issue is likely one of the following:
1. Heroes with these talents exist but are NOT in the weights file
2. Heroes with Elemental type should have talents via "Essence of X" keywords
3. The card selector's get_hero_classes_and_talents() isn't properly extracting talents
""")
