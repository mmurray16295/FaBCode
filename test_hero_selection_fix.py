"""
Test the fixed hero selection logic to verify Young/Adult matching works correctly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

from card_selector import CardSelector

# Initialize selector
selector = CardSelector()

print("=" * 80)
print("TESTING HERO SELECTION FIX")
print("=" * 80)

# Test each format
formats = ['cc', 'll', 'blitz']

for format in formats:
    print(f"\n{'='*80}")
    print(f"FORMAT: {format.upper()}")
    print(f"{'='*80}")
    
    # Try to select 10 heroes for this format
    success_count = 0
    young_count = 0
    adult_count = 0
    
    for i in range(10):
        try:
            hero_key, hero_card, hero_weights = selector.select_random_hero(format)
            success_count += 1
            
            is_young = 'Young' in hero_card.get('types', [])
            if is_young:
                young_count += 1
            else:
                adult_count += 1
            
            print(f"{i+1}. {hero_key} → {hero_card['name']} ({'YOUNG' if is_young else 'ADULT'})")
            
        except ValueError as e:
            print(f"{i+1}. FAILED: {e}")
    
    print(f"\nSummary for {format.upper()}:")
    print(f"  Success: {success_count}/10")
    print(f"  Young: {young_count}")
    print(f"  Adult: {adult_count}")

print("\n" + "=" * 80)
print("EXPECTED RESULTS:")
print("=" * 80)
print("CC Format: Should select mostly/all ADULT heroes")
print("LL Format: Should select mostly/all ADULT heroes")
print("Blitz Format: Should select mostly/all YOUNG heroes")
print("\nAll formats should succeed 10/10 times (no ValueError)")
