#!/usr/bin/env python3
"""
Scrape card popularity data from fabrec.gg
Uses image IDs to cross-reference with card.json
Calculates weighted scores based on hero popularity × card usage
"""

import json
import re
import requests
from bs4 import BeautifulSoup
import time
from collections import defaultdict
from typing import Dict, List, Tuple
import sys

def load_card_database(card_json_path: str) -> Dict[str, dict]:
    """Load card.json and create lookup by card ID"""
    print(f"Loading card database from {card_json_path}...")
    with open(card_json_path, 'r') as f:
        cards = json.load(f)
    
    # Create lookup by card ID (e.g., "EVR018")
    card_lookup = {}
    for card in cards:
        for printing in card.get('printings', []):
            card_id = printing.get('id', '')
            if card_id:
                card_lookup[card_id] = {
                    'name': card['name'],
                    'types': card.get('types', []),
                    'card_id': card_id,
                    'full_card': card
                }
    
    print(f"Loaded {len(card_lookup)} card printings")
    return card_lookup

def get_hero_list_with_percentages() -> Dict[str, float]:
    """Get all heroes with their deck percentages from the main page"""
    url = "https://fabrec.gg/?format=constructed"
    print(f"Fetching hero list from {url}...")
    
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    script_tag = soup.find('script', {'id': '__NEXT_DATA__'})
    if not script_tag:
        print("ERROR: Could not find hero data")
        return {}
    
    data = json.loads(script_tag.string)
    page_props = data['props']['pageProps']
    
    # Get total decks
    total_decks = page_props.get('deckTotal', {}).get('total_decks', 1)
    
    # Get hero list
    heroes = page_props.get('heroes', {}).get('constructed', [])
    
    hero_percentages = {}
    for hero in heroes:
        hero_name = hero.get('name')
        count = hero.get('count', 0)
        percentage = (count / total_decks) * 100
        hero_percentages[hero_name] = percentage
    
    print(f"Found {len(hero_percentages)} heroes, total decks: {total_decks}")
    return hero_percentages

def scrape_hero_page(hero_identifier: str, hero_percentage: float, card_lookup: Dict[str, dict]) -> List[Tuple[str, str, float]]:
    """
    Scrape a hero page for card usage data
    Returns: [(card_name, card_id, usage_percentage)]
    """
    url = f"https://fabrec.gg/hero/{hero_identifier}"
    print(f"\nScraping {url}... (hero at {hero_percentage:.4f}% of decks)")
    
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract Next.js JSON data
    script_tag = soup.find('script', {'id': '__NEXT_DATA__'})
    if not script_tag:
        print(f"ERROR: Could not find data for {hero_identifier}")
        return []
    
    data = json.loads(script_tag.string)
    
    try:
        page_props = data['props']['pageProps']
        
        # Find card usage data in cardlists
        cards_found = []
        
        if 'cardData' not in page_props:
            print("  No cardData found")
            return []
        
        card_data = page_props['cardData']
        
        # cardlists are in container.cardlists
        if 'container' not in card_data or 'cardlists' not in card_data['container']:
            print("  No cardlists found")
            return []
        
        cardlists = card_data['container']['cardlists'].get('jsonData', {})
        
        # Process different card sections (equipment, weapon, maindeck, sideboard, etc.)
        for section, cards in cardlists.items():
            if not isinstance(cards, list) or len(cards) == 0:
                continue
            print(f"  Processing section: {section} ({len(cards)} cards)")
            
            for card_info in cards:
                # Extract card ID (e.g., "EVR018-CF" or "EVR018")
                card_id_full = card_info.get('card', '')
                if not card_id_full:
                    continue
                
                # Remove suffix like "-CF" to get base card ID
                card_id = re.sub(r'-[A-Z]+$', '', card_id_full)
                
                # Get usage percentage (maxPopularity is already a decimal, e.g., 0.8126 = 81.26%)
                usage_decimal = card_info.get('maxPopularity', 0.0)
                usage_pct = usage_decimal * 100  # Convert to percentage
                
                # Look up card name using base card ID
                if card_id in card_lookup:
                    card_name = card_lookup[card_id]['name']
                    cards_found.append((card_name, card_id, usage_pct))
                    print(f"    Found: {card_name} ({card_id}) - {usage_pct:.2f}% of decks")
                else:
                    print(f"    WARNING: Card ID {card_id} not found in card database")
        
        return cards_found
        
    except Exception as e:
        print(f"ERROR parsing hero page: {e}")
        import traceback
        traceback.print_exc()
        # Print first 2000 chars of data structure for debugging
        print("\nData structure preview:")
        print(json.dumps(data, indent=2)[:2000])
        return 0.0, []

def normalize_card_name(name: str, card_id: str) -> str:
    """Convert card name to filename format: Remove punctuation, spaces to underscores"""
    # Remove punctuation except underscores and hyphens
    normalized = re.sub(r'[^\w\s-]', '', name)
    # Replace spaces with underscores
    normalized = normalized.replace(' ', '_')
    # Add card ID
    return f"{normalized}_{card_id}"

def main():
    # Configuration
    CARD_JSON_PATH = '/workspace/fab/src/data/card.json'
    OUTPUT_PATH = '/workspace/fab/src/data/card_popularity_weights.json'
    MAX_HEROES = int(sys.argv[1]) if len(sys.argv) > 1 else None  # None = all heroes
    
    # Load card database
    card_lookup = load_card_database(CARD_JSON_PATH)
    
    # Get hero list with percentages
    hero_percentages = get_hero_list_with_percentages()
    
    # Get list of heroes to scrape
    heroes_to_scrape = list(hero_percentages.keys())
    if MAX_HEROES:
        heroes_to_scrape = heroes_to_scrape[:MAX_HEROES]
    
    print(f"\nWill scrape {len(heroes_to_scrape)} heroes")
    
    # Dictionary to accumulate weights per card
    # Format: {normalized_card_name: [weight1, weight2, ...]}
    card_weights = defaultdict(list)
    card_metadata = {}  # Store full card info for each normalized name
    
    # Scrape each hero
    for hero_id in heroes_to_scrape:
        hero_pct = hero_percentages.get(hero_id, 0.0)
        
        if hero_pct == 0:
            print(f"WARNING: No deck percentage found for {hero_id}, skipping...")
            continue
        
        cards = scrape_hero_page(hero_id, hero_pct, card_lookup)
        
        # Calculate weights for each card
        for card_name, card_id, usage_pct in cards:
            weight = (hero_pct / 100.0) * (usage_pct / 100.0)
            normalized_name = normalize_card_name(card_name, card_id)
            
            card_weights[normalized_name].append(weight)
            
            if normalized_name not in card_metadata:
                card_metadata[normalized_name] = {
                    'original_name': card_name,
                    'card_id': card_id,
                    'hero_contributions': []
                }
            
            card_metadata[normalized_name]['hero_contributions'].append({
                'hero': hero_id,
                'hero_pct': hero_pct,
                'usage_pct': usage_pct,
                'weight': weight
            })
        
        # Be nice to the server
        time.sleep(1)
    
    # Calculate total weights
    final_weights = {}
    for card_name, weights in card_weights.items():
        total_weight = sum(weights)
        final_weights[card_name] = {
            'total_weight': total_weight,
            'individual_weights': weights,
            'metadata': card_metadata[card_name]
        }
    
    # Sort by total weight (descending)
    sorted_weights = dict(sorted(final_weights.items(), key=lambda x: x[1]['total_weight'], reverse=True))
    
    # Save results
    output_data = {
        'metadata': {
            'total_cards': len(sorted_weights),
            'total_heroes_scraped': len(heroes_to_scrape),
            'generation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'weights': sorted_weights
    }
    
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Scraping complete!")
    print(f"Total cards found: {len(sorted_weights)}")
    print(f"Output saved to: {OUTPUT_PATH}")
    print(f"\nTop 10 most popular cards:")
    for i, (card_name, data) in enumerate(list(sorted_weights.items())[:10], 1):
        print(f"  {i}. {data['metadata']['original_name']} ({data['metadata']['card_id']})")
        print(f"     Total weight: {data['total_weight']:.6f}")
        print(f"     Used by {len(data['individual_weights'])} heroes")

if __name__ == '__main__':
    main()
