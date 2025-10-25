import random
import json
from pathlib import Path
from collections import defaultdict
from card_selector import CardSelector
from card_selector_smooth import SmoothCardSelector

# Load card database and hero list
CARD_JSON = Path('c:/VS Code/FaB Code/data/card.json')
WEIGHTS_JSON = Path('c:/VS Code/FaB Code/data/card_weights.json')

with CARD_JSON.open('r', encoding='utf-8') as f:
    all_cards = json.load(f)
with WEIGHTS_JSON.open('r', encoding='utf-8') as f:
    weights = json.load(f)

# Get all hero cards
hero_cards = [card for card in all_cards if 'Hero' in card.get('types', [])]
formats = ['cc', 'blitz']

# Zones to test (simplified)
zones = ['Weapon', 'Weapon or Off-Hand', 'Head', 'Chest', 'Arms', 'Legs', 'Graveyard', 'Pitch', 'Banish']

# Main test function
def run_selector_test(selector_class, selector_name, iterations=25000):
    print(f'Running {selector_name} selector test for {iterations} iterations...')
    selector = selector_class(
        all_cards=all_cards,
        weights=weights,
        card_json_path=str(CARD_JSON),
        weights_json_path=str(WEIGHTS_JSON)
    )
    card_counts = defaultdict(int)
    class_counts = defaultdict(int)
    type_counts = defaultdict(int)
    for i in range(iterations):
        hero_card = random.choice(hero_cards)
        format = random.choice(formats)
        pools = selector.build_card_pools(hero_card, weights.get(hero_card['name'], {}), format)
        all_hero_cards = pools['generic'] + pools['class_only'] + pools['talent_only'] + pools['both']
        for zone in zones:
            card = selector.select_card_for_zone(pools, zone, all_hero_cards)
            if card:
                card_counts[card['name']] += 1
                for c in card.get('types', []):
                    type_counts[c] += 1
                if 'class' in card:
                    class_counts[card['class']] += 1
    # Output summary
    print(f'--- {selector_name} selector results ---')
    print('Top 20 most selected cards:')
    for name, count in sorted(card_counts.items(), key=lambda x: -x[1])[:20]:
        print(f'{name}: {count}')
    print('\nClass counts:')
    for cls, count in class_counts.items():
        print(f'{cls}: {count}')
    print('\nType counts:')
    for typ, count in type_counts.items():
        print(f'{typ}: {count}')
    print('--------------------------------------\n')

if __name__ == '__main__':
    run_selector_test(CardSelector, 'Weighted', iterations=25000)
    run_selector_test(SmoothCardSelector, 'Smooth', iterations=25000)
