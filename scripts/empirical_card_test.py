"""
Empirically test which cards can actually be selected by the CardSelector.
This runs the actual selection logic rather than trying to analyze it statically.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter

sys.path.append('scripts')
from card_selector import CardSelector

def test_card_selectability():
    """Test which cards can actually be selected by running the real CardSelector logic."""
    
    print("=" * 80)
    print("EMPIRICAL CARD SELECTABILITY TEST")
    print("=" * 80)
    print("Running actual CardSelector logic to determine which cards can be selected...")
    print()
    
    # Initialize selector
    selector = CardSelector()
    
    # Load all cards
    with open('packages/FaBCardDetector_Windows_20251010_124326/data/card.json', 'r') as f:
        all_cards = json.load(f)
    
    # Load hero weights
    with open('data/card_popularity_weights_by_hero.json', 'r') as f:
        hero_weights_data = json.load(f)
    
    # Create lookup by name
    cards_by_name = {card['name']: card for card in all_cards}
    
    # Track which cards appear in ANY hero's card pools
    cards_in_pools = defaultdict(set)  # card_name -> set of hero_keys that can select it
    cards_never_in_pools = set()  # card names that never appear in any pool
    
    print("Testing card pool generation for all heroes...")
    print("-" * 80)
    
    # Test all heroes in all formats
    formats = ['cc', 'll', 'blitz']
    format_counts = {'cc': 0, 'll': 0, 'blitz': 0}
    
    for format in formats:
        print(f"\nFormat: {format.upper()}")
        
        # Get all hero keys
        hero_keys = list(selector.weights_data['heroes'].keys())
        
        for hero_key in hero_keys:
            try:
                # Manually look up the hero card and weights
                hero_weights = selector.weights_data['heroes'][hero_key]
                
                # Find the hero card
                hero_key_normalized = hero_key.lower().replace('-', '').replace(' ', '').replace(',', '').replace("'", '').replace('!', '').replace('/', '')
                matching_heroes = []
                
                for card in selector.all_cards:
                    if 'Hero' in card.get('types', []):
                        name_normalized = card['name'].lower().replace(' ', '').replace(',', '').replace("'", '').replace('!', '').replace('/', '')
                        if hero_key_normalized in name_normalized or name_normalized in hero_key_normalized:
                            matching_heroes.append(card)
                
                if not matching_heroes:
                    continue
                
                # Filter by format legality
                format_key = f'{format}_legal'
                legal_heroes = [h for h in matching_heroes if h.get(format_key)]
                
                if not legal_heroes:
                    continue
                
                # For Blitz: prefer Young heroes, fall back to any legal
                if format == 'blitz':
                    young_heroes = [h for h in legal_heroes if 'Young' in h.get('types', [])]
                    hero_card = young_heroes[0] if young_heroes else legal_heroes[0]
                # For CC/LL: prefer Adult heroes (non-Young), fall back to any legal
                else:
                    adult_heroes = [h for h in legal_heroes if 'Young' not in h.get('types', [])]
                    hero_card = adult_heroes[0] if adult_heroes else legal_heroes[0]
                
                format_counts[format] += 1
                
                # Build card pools
                card_pools = selector.build_card_pools(hero_card, hero_weights, format)
                
                # Track all cards in all pools
                for pool_name, cards in card_pools.items():
                    for card in cards:
                        cards_in_pools[card['name']].add(hero_key)
                
                pool_size = sum(len(cards) for cards in card_pools.values())
                print(f"  {hero_key}: {pool_size} cards")
                
            except ValueError as e:
                # Hero not legal in this format
                continue
            except Exception as e:
                print(f"  ERROR with {hero_key}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print("\n" + "=" * 80)
    print("ANALYSIS RESULTS")
    print("=" * 80)
    
    # Find all unique card names (deduplicated)
    unique_card_names = set()
    for card in all_cards:
        # Deduplicate using the split approach
        base_name = card['name'].split(' // ')[0]
        if base_name == card['name']:
            unique_card_names.add(card['name'])
    
    # Mark heroes in weights as selectable (they appear on playmats)
    selectable_cards = set(cards_in_pools.keys())
    
    # With the new 75/25 weighted/unweighted hero selection:
    # - 75% of the time: Select from heroes in weights file (weighted)
    # - 25% of the time: Select from ALL heroes in card.json (unweighted)
    # This means ALL legal heroes can now be selected, not just those in weights!
    for card in all_cards:
        if 'Hero' in card.get('types', []):
            # Check if hero is legal in ANY format
            if card.get('cc_legal') or card.get('ll_legal') or card.get('blitz_legal'):
                selectable_cards.add(card['name'])
    
    print(f"\nTotal unique cards: {len(unique_card_names)}")
    print(f"Cards that CAN be selected: {len(selectable_cards)}")
    print(f"Cards that CANNOT be selected: {len(unique_card_names) - len(selectable_cards)}")
    
    # Find cards that cannot be selected
    for card_name in unique_card_names:
        if card_name not in selectable_cards:
            cards_never_in_pools.add(card_name)
    
    # Categorize unselectable cards
    print("\n" + "-" * 80)
    print("UNSELECTABLE CARDS BY REASON")
    print("-" * 80)
    
    categorized = {
        'heroes_not_in_weights': [],
        'tokens': [],
        'not_legal_any_format': [],
        'no_compatible_hero': [],
        'other': []
    }
    
    for card_name in sorted(cards_never_in_pools):
        card = cards_by_name[card_name]
        card_types = set(card.get('types', []))
        
        # Categorize
        if 'Hero' in card_types:
            categorized['heroes_not_in_weights'].append(card)
        elif card.get('card_keywords') and len(card['card_keywords']) > 0 and card['card_keywords'][0] == 'Token':
            categorized['tokens'].append(card)
        elif not (card.get('cc_legal') or card.get('ll_legal') or card.get('blitz_legal')):
            categorized['not_legal_any_format'].append(card)
        else:
            categorized['other'].append(card)
    
    print(f"\nHeroes not in weights file: {len(categorized['heroes_not_in_weights'])}")
    print(f"Tokens: {len(categorized['tokens'])}")
    print(f"Not legal in any format: {len(categorized['not_legal_any_format'])}")
    print(f"Other (needs investigation): {len(categorized['other'])}")
    
    # Show details for "other" category
    if categorized['other']:
        print("\n" + "-" * 80)
        print("CARDS THAT NEED INVESTIGATION (legal but not in pools):")
        print("-" * 80)
        for card in categorized['other'][:20]:  # Show first 20
            print(f"  {card['name']}")
            print(f"    Types: {card.get('types', [])}")
            print(f"    CC: {card.get('cc_legal')}, LL: {card.get('ll_legal')}, Blitz: {card.get('blitz_legal')}")
            print()
    
    # Save results
    output = {
        'summary': {
            'total_unique_cards': len(unique_card_names),
            'cards_can_be_selected': len(selectable_cards),
            'cards_cannot_be_selected': len(cards_never_in_pools),
            'notes': 'Empirically tested by running actual CardSelector logic. Heroes in weights file count as selectable (appear on playmats).'
        },
        'categories': {
            'heroes_not_in_weights': [c['name'] for c in categorized['heroes_not_in_weights']],
            'tokens': [c['name'] for c in categorized['tokens']],
            'not_legal_any_format': [c['name'] for c in categorized['not_legal_any_format']],
            'other': [c['name'] for c in categorized['other']]
        },
        'detailed': {
            'heroes_not_in_weights': categorized['heroes_not_in_weights'],
            'tokens': categorized['tokens'],
            'not_legal_any_format': categorized['not_legal_any_format'],
            'other': categorized['other']
        },
        'card_to_heroes_map': {
            card_name: sorted(list(heroes))
            for card_name, heroes in cards_in_pools.items()
        }
    }
    
    output_path = 'EMPIRICAL_CARD_SELECTABILITY.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n" + "=" * 80)
    print(f"Results saved to: {output_path}")
    print("=" * 80)
    
    return output

if __name__ == '__main__':
    results = test_card_selectability()
