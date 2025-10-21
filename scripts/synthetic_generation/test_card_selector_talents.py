"""
Test the get_hero_classes_and_talents() function to see why it's not working.
"""

import json
import sys
from pathlib import Path

from card_selector import CardSelector

# Initialize selector
selector = CardSelector()

# Load card data
with open('data/card.json', 'r', encoding='utf-8') as f:
    all_cards = json.load(f)

# Get some Elemental heroes
elemental_heroes = [
    'Bravo, Star of the Show',
    'Aurora, Shooting Star',
    'Briar, Warden of Thorns',
    'Iyslander, Stormbind',
    'Florian, Rotwood Harbinger',
    'Jarl Vetreiði'
]

print("="*80)
print("TESTING get_hero_classes_and_talents()")
print("="*80)

for hero_name in elemental_heroes:
    hero_card = next((c for c in all_cards if c['name'] == hero_name), None)
    
    if not hero_card:
        print(f"\n✗ {hero_name}: NOT FOUND")
        continue
    
    print(f"\n{hero_name}:")
    print(f"  Types: {hero_card.get('types', [])}")
    print(f"  Keywords: {hero_card.get('card_keywords', [])}")
    
    # Test the function
    classes, talents = selector.get_hero_classes_and_talents(hero_card)
    
    print(f"  Extracted classes: {classes}")
    print(f"  Extracted talents: {talents}")
    
    if not talents:
        print(f"  ⚠️  NO TALENTS EXTRACTED!")
    else:
        print(f"  ✓ Talents found: {talents}")

# Now test with a specific Earth card to see if it would match
print("\n" + "="*80)
print("TESTING IF EARTH CARDS MATCH BRAVO")
print("="*80)

bravo = next((c for c in all_cards if c['name'] == 'Bravo, Star of the Show'), None)
if bravo:
    bravo_classes, bravo_talents = selector.get_hero_classes_and_talents(bravo)
    print(f"\nBravo talents: {bravo_talents}")
    
    # Test an Earth card
    earth_card = next((c for c in all_cards if c['name'] == 'Redwood Hammer'), None)
    if earth_card:
        print(f"\nRedwood Hammer:")
        print(f"  Types: {earth_card.get('types', [])}")
        
        earth_card_types = set(earth_card.get('types', []))
        from card_selector import ALL_TALENTS
        card_talents = earth_card_types & ALL_TALENTS
        
        print(f"  Card talents: {card_talents}")
        print(f"  Hero talents: {bravo_talents}")
        
        if card_talents and not bravo_talents:
            print(f"  ✗ MISMATCH: Card has talents but hero has none")
        elif card_talents and bravo_talents:
            if card_talents & bravo_talents:
                print(f"  ✓ MATCH: Card talents overlap with hero talents")
            else:
                print(f"  ✗ MISMATCH: Card talents don't overlap with hero talents")
