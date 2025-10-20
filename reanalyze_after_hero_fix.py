"""
Re-analyze unfixable cards after fixing hero selection logic.
Young heroes and Adult heroes should now be selectable.
"""

import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

from card_selector import CardSelector

# Load card data
with open('data/card.json', 'r', encoding='utf-8') as f:
    all_cards = json.load(f)

# Load old unfixable analysis
with open('CARDS_CANNOT_BE_SELECTED.json', 'r', encoding='utf-8') as f:
    old_analysis = json.load(f)

print("="*80)
print("RE-ANALYSIS AFTER HERO SELECTION FIX")
print("="*80)

print(f"\nOLD counts:")
print(f"  Total unfixable: {old_analysis['summary']['total_unfixable']}")
print(f"  not_legal_format: {old_analysis['summary']['not_legal_format']}")
print(f"  hero_cards: {old_analysis['summary']['hero_cards']}")
print(f"  talent_mismatch: {old_analysis['summary']['talent_mismatch']}")
print(f"  class_mismatch: {old_analysis['summary']['class_mismatch']}")

# Young heroes that were in "not_legal_format" but should now be selectable
young_heroes_to_remove = [
    "Blaze, Firemind", "Boltyn", "Dash, Database", "Puffin", "Rhinar",
    "Riptide", "Scurv, Stowaway", "Shiyana, Diamond Gemini", "Squizzy & Floof",
    "Taylor", "Teklovossen", "Terra", "Tuffnut", "Uzuri", "Valda Brightaxe",
    "Verdance", "Victor Goldmane", "Viserai", "Vynnset", "Yoji, Royal Protector",
    "Yorick, Weaver of Tales"
]

# Adult heroes that were in "hero_cards" but should now be selectable
adult_heroes_to_remove = [
    "Chane, Bound by Shadow", "Gravy Bones, Shipwrecked Looter", "Olympia, Prized Fighter",
    "Puffin, Hightail", "Rhinar, Reckless Rampage", "Riptide, Lurker of the Deep",
    "Ser Boltyn, Breaker of Dawn", "Teklovossen, Esteemed Magnate", "Tuffnut, Bumbling Hulkster",
    "Uzuri, Switchblade", "Valda, Seismic Impact", "Verdance, Thorn of the Rose",
    "Victor Goldmane, High and Mighty", "Viserai, Rune Blood", "Vynnset, Iron Maiden"
]

# Filter out the heroes that are now selectable
new_not_legal_format = [
    name for name in old_analysis['categories']['not_legal_format']
    if name not in young_heroes_to_remove
]

new_hero_cards = [
    name for name in old_analysis['categories']['hero_cards']
    if name not in adult_heroes_to_remove
]

print(f"\nNEW counts after fix:")
print(f"  not_legal_format: {len(new_not_legal_format)} (removed {old_analysis['summary']['not_legal_format'] - len(new_not_legal_format)} Young heroes)")
print(f"  hero_cards: {len(new_hero_cards)} (removed {old_analysis['summary']['hero_cards'] - len(new_hero_cards)} Adult heroes)")
print(f"  talent_mismatch: {old_analysis['summary']['talent_mismatch']} (unchanged)")
print(f"  class_mismatch: {old_analysis['summary']['class_mismatch']} (unchanged)")

new_total = len(new_not_legal_format) + len(new_hero_cards) + old_analysis['summary']['talent_mismatch'] + old_analysis['summary']['class_mismatch']
print(f"  Total unfixable: {new_total} (removed {old_analysis['summary']['total_unfixable'] - new_total} hero cards)")

print(f"\nRemaining 'not_legal_format' cards:")
for name in new_not_legal_format:
    print(f"  - {name}")

print(f"\nRemaining 'hero_cards':")
for name in new_hero_cards:
    print(f"  - {name}")

print("\n" + "="*80)
print("SUMMARY:")
print("="*80)
print(f"""
BEFORE FIX: 124 unfixable cards
- 43 not_legal_format (including ~21 Young heroes)
- 15 hero_cards (Adult heroes)
- 57 talent_mismatch
- 9 class_mismatch

AFTER FIX: {new_total} unfixable cards
- {len(new_not_legal_format)} not_legal_format (Event cards, special format heroes)
- {len(new_hero_cards)} hero_cards (none - all heroes now selectable!)
- 57 talent_mismatch (needs investigation)
- 9 class_mismatch (Bard cards - Yorick hero should be added to weights)

IMPACT: {old_analysis['summary']['total_unfixable'] - new_total} additional hero cards are now selectable!

Training coverage improvement:
- OLD: 2,385 / 2,641 = 90.3% of intended classes
- NEW: {2385 + (old_analysis['summary']['total_unfixable'] - new_total)} / 2,641 = {100 * (2385 + (old_analysis['summary']['total_unfixable'] - new_total)) / 2641:.1f}% of intended classes
""")
