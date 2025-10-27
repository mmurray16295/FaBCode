"""
Analyze card specificity across heroes to understand training data bias.
"""
import json
from pathlib import Path
from collections import defaultdict

# Load card weights
weights_file = Path(__file__).parent.parent / 'data' / 'card_weights_all_printings.json'
with open(weights_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Count how many heroes use each card
card_hero_counts = defaultdict(set)
card_usage_data = {}

for format_name in ['cc', 'blitz']:
    for hero_name, hero_data in data['formats'][format_name].items():
        for section_name, cards in hero_data['sections'].items():
            for card in cards:
                card_name = card['card_name']
                card_hero_counts[card_name].add(hero_name)
                
                # Store max usage for this card
                if card_name not in card_usage_data or card['usage_percentage'] > card_usage_data[card_name]['usage']:
                    card_usage_data[card_name] = {
                        'usage': card['usage_percentage'],
                        'hero_count': 0  # Will update after
                    }

# Update hero counts
for card_name in card_usage_data:
    card_usage_data[card_name]['hero_count'] = len(card_hero_counts[card_name])

# Categorize cards
generic = []
semi_generic = []
class_specific = []
hero_specific = []

for card_name, heroes in card_hero_counts.items():
    hero_count = len(heroes)
    usage = card_usage_data[card_name]['usage']
    
    if hero_count >= 10:
        generic.append((card_name, hero_count, usage))
    elif hero_count >= 5:
        semi_generic.append((card_name, hero_count, usage))
    elif hero_count >= 2:
        class_specific.append((card_name, hero_count, usage))
    else:
        hero_specific.append((card_name, hero_count, usage))

print("=" * 70)
print("CARD SPECIFICITY ANALYSIS")
print("=" * 70)
print()
print(f"Total unique cards: {len(card_hero_counts)}")
print()
print(f"Generic (10+ heroes):      {len(generic):4d} cards ({len(generic)/len(card_hero_counts)*100:.1f}%)")
print(f"Semi-generic (5-9 heroes): {len(semi_generic):4d} cards ({len(semi_generic)/len(card_hero_counts)*100:.1f}%)")
print(f"Class-specific (2-4):      {len(class_specific):4d} cards ({len(class_specific)/len(card_hero_counts)*100:.1f}%)")
print(f"Hero-specific (1 hero):    {len(hero_specific):4d} cards ({len(hero_specific)/len(card_hero_counts)*100:.1f}%)")
print()
print("=" * 70)
print("TOP GENERIC CARDS (seen across most heroes)")
print("=" * 70)
for card, hero_count, usage in sorted(generic, key=lambda x: -x[1])[:10]:
    print(f"  {card:40s} - {hero_count:2d} heroes, {usage:5.1f}% usage")

print()
print("=" * 70)
print("SAMPLE CLASS-SPECIFIC CARDS (fewer heroes, high usage)")
print("=" * 70)
class_high_usage = [(c, h, u) for c, h, u in class_specific if u >= 50]
for card, hero_count, usage in sorted(class_high_usage, key=lambda x: -x[2])[:10]:
    print(f"  {card:40s} - {hero_count:2d} heroes, {usage:5.1f}% usage")

print()
print("=" * 70)
print("PROPOSED ADJUSTMENT STRATEGY")
print("=" * 70)
print()
print("Instead of penalizing by usage %, adjust by TRAINING DATA BIAS:")
print()
print("  Hero-specific (1 hero):    +15% boost   (underrepresented)")
print("  Class-specific (2-4):      +10% boost   (somewhat underrepresented)")
print("  Semi-generic (5-9):        +5% boost    (slight underrepresentation)")
print("  Generic (10+):             No change    (properly represented)")
print()
print("This compensates for the training data imbalance!")
print()
