#!/usr/bin/env python3
"""
Extract all heroes from card.json into a dedicated heroes_card.json file.
Provides clean visibility of all heroes with their format legality and metadata.
"""

import json
from pathlib import Path
from datetime import datetime

def extract_heroes():
    """Extract all heroes from card.json and create heroes_card.json"""
    
    # Paths
    card_json_path = Path(__file__).parent.parent.parent.parent / 'data' / 'card.json'
    heroes_json_path = Path(__file__).parent.parent.parent.parent / 'data' / 'heroes_card.json'
    
    print("=" * 80)
    print("HERO EXTRACTION TOOL")
    print("=" * 80)
    print(f"Reading from: {card_json_path}")
    print(f"Writing to: {heroes_json_path}")
    print()
    
    # Load card.json
    with open(card_json_path, 'r', encoding='utf-8') as f:
        cards = json.load(f)
    
    print(f"Loaded {len(cards)} cards")
    
    # Extract heroes
    heroes = []
    young_heroes = []
    
    for card in cards:
        types = card.get('types', [])
        
        if 'Hero' in types:
            is_young = 'Young' in types
            
            hero_data = {
                'name': card.get('name', ''),
                'pitch': card.get('pitch', ''),
                'cost': card.get('cost', ''),
                'power': card.get('power', ''),
                'defense': card.get('defense', ''),
                'health': card.get('health', ''),
                'intellect': card.get('intellect', ''),
                'types': types,
                'card_keywords': card.get('card_keywords', []),
                'abilities_and_effects': card.get('abilities_and_effects', []),
                'ability_and_effect_keywords': card.get('ability_and_effect_keywords', []),
                'granted_keywords': card.get('granted_keywords', []),
                'printings': card.get('printings', []),
                'legality': {
                    'cc_legal': card.get('cc_legal', False),
                    'cc_living_legend': card.get('cc_living_legend', False),
                    'll_legal': card.get('ll_legal', False),
                    'll_living_legend': card.get('ll_living_legend', False),
                    'blitz_legal': card.get('blitz_legal', False),
                    'blitz_living_legend': card.get('blitz_living_legend', False),
                    'commoner_legal': card.get('commoner_legal', False),
                    'blitz_living_legend_start': card.get('blitz_living_legend_start', None),
                    'cc_living_legend_start': card.get('cc_living_legend_start', None),
                    'll_living_legend_start': card.get('ll_living_legend_start', None)
                },
                'is_young': is_young
            }
            
            if is_young:
                young_heroes.append(hero_data)
            else:
                heroes.append(hero_data)
    
    # Sort alphabetically
    heroes.sort(key=lambda x: x['name'])
    young_heroes.sort(key=lambda x: x['name'])
    
    # Create output structure
    output = {
        'metadata': {
            'version': '1.0',
            'generated': datetime.now().isoformat(),
            'source': 'card.json',
            'total_adult_heroes': len(heroes),
            'total_young_heroes': len(young_heroes),
            'total_heroes': len(heroes) + len(young_heroes)
        },
        'adult_heroes': heroes,
        'young_heroes': young_heroes
    }
    
    # Calculate format statistics
    cc_count = sum(1 for h in heroes if h['legality']['cc_legal'])
    cc_ll_count = sum(1 for h in heroes if h['legality']['cc_living_legend'])
    ll_count = sum(1 for h in heroes if h['legality']['ll_legal'])
    ll_ll_count = sum(1 for h in heroes if h['legality']['ll_living_legend'])
    blitz_count = sum(1 for h in heroes if h['legality']['blitz_legal'])
    blitz_ll_count = sum(1 for h in heroes if h['legality']['blitz_living_legend'])
    
    output['metadata']['format_stats'] = {
        'cc_legal': cc_count,
        'cc_living_legend': cc_ll_count,
        'll_legal': ll_count,
        'll_living_legend': ll_ll_count,
        'blitz_legal': blitz_count,
        'blitz_living_legend': blitz_ll_count
    }
    
    # Save to file
    with open(heroes_json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print()
    print("=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"Adult Heroes: {len(heroes)}")
    print(f"Young Heroes: {len(young_heroes)}")
    print(f"Total Heroes: {len(heroes) + len(young_heroes)}")
    print()
    print("Format Legality (Adult Heroes):")
    print(f"  CC Legal:        {cc_count}")
    print(f"  CC Living Legend: {cc_ll_count}")
    print(f"  LL Legal:        {ll_count}")
    print(f"  LL Living Legend: {ll_ll_count}")
    print(f"  Blitz Legal:     {blitz_count}")
    print(f"  Blitz LL:        {blitz_ll_count}")
    print()
    print(f"✓ Saved to: {heroes_json_path}")
    print(f"✓ File: data/heroes_card.json")
    print("=" * 80)

if __name__ == '__main__':
    extract_heroes()
