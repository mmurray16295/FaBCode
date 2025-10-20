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
    with open(card_json_path, 'r', encoding='utf-8') as f:
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

def scrape_hero_page(hero_identifier: str, hero_percentage: float, card_lookup: Dict[str, dict]) -> Dict[str, List[Tuple[str, str, float]]]:
    """
    Scrape a hero page for card usage data
    Returns: {section_name: [(card_name, card_id, usage_percentage)]}
    Preserves website's section ordering (equipment, weapon, maindeck, etc.)
    """
    url = f"https://fabrec.gg/hero/{hero_identifier}"
    print(f"\nScraping {url}... (hero at {hero_percentage:.4f}% of decks)")
    
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract Next.js JSON data
    script_tag = soup.find('script', {'id': '__NEXT_DATA__'})
    if not script_tag:
        print(f"ERROR: Could not find data for {hero_identifier}")
        return {}
    
    data = json.loads(script_tag.string)
    
    try:
        page_props = data['props']['pageProps']
        
        # Find card usage data in cardlists
        cards_by_section = {}
        
        if 'cardData' not in page_props:
            print("  No cardData found")
            return {}
        
        card_data = page_props['cardData']
        
        # cardlists are in container.cardlists
        if 'container' not in card_data or 'cardlists' not in card_data['container']:
            print("  No cardlists found")
            return {}
        
        cardlists = card_data['container']['cardlists'].get('jsonData', {})
        
        # Process different card sections (equipment, weapon, maindeck, sideboard, etc.)
        # Preserve the order from the website
        for section, cards in cardlists.items():
            if not isinstance(cards, list) or len(cards) == 0:
                continue
            print(f"  Processing section: {section} ({len(cards)} cards)")
            
            section_cards = []
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
                    section_cards.append((card_name, card_id, usage_pct))
                    print(f"    Found: {card_name} ({card_id}) - {usage_pct:.2f}% of decks")
                else:
                    print(f"    WARNING: Card ID {card_id} not found in card database")
            
            if section_cards:
                cards_by_section[section] = section_cards
        
        return cards_by_section
        
    except Exception as e:
        print(f"ERROR parsing hero page: {e}")
        import traceback
        traceback.print_exc()
        # Print first 2000 chars of data structure for debugging
        print("\nData structure preview:")
        print(json.dumps(data, indent=2)[:2000])
        return {}

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
    CARD_JSON_PATH = 'data/card.json'
    OUTPUT_PATH = 'data/card_popularity_weights_by_hero.json'
    MAX_HEROES = int(sys.argv[1]) if len(sys.argv) > 1 else None  # None = all heroes
    
    # Load card database
    card_lookup = load_card_database(CARD_JSON_PATH)
    
    # Get hero list with percentages
    hero_percentages = get_hero_list_with_percentages()
    
    # Get list of heroes to scrape (sorted alphabetically)
    heroes_to_scrape = sorted(list(hero_percentages.keys()))
    if MAX_HEROES:
        heroes_to_scrape = heroes_to_scrape[:MAX_HEROES]
    
    print(f"\nWill scrape {len(heroes_to_scrape)} heroes (alphabetically sorted)")
    
    # Dictionary to store hero-centric data
    # Format: {hero_name: {section_name: [{card_name, card_id, usage_pct, normalized_name}]}}
    hero_data = {}
    
    # Scrape each hero
    for hero_id in heroes_to_scrape:
        hero_pct = hero_percentages.get(hero_id, 0.0)
        
        if hero_pct == 0:
            print(f"WARNING: No deck percentage found for {hero_id}, skipping...")
            continue
        
        cards_by_section = scrape_hero_page(hero_id, hero_pct, card_lookup)
        
        if not cards_by_section:
            print(f"WARNING: No cards found for {hero_id}")
            continue
        
        # Build hero's card data with sections preserved
        hero_sections = {}
        total_cards = 0
        
        for section, cards in cards_by_section.items():
            section_data = []
            for card_name, card_id, usage_pct in cards:
                normalized_name = normalize_card_name(card_name, card_id)
                section_data.append({
                    'card_name': card_name,
                    'card_id': card_id,
                    'normalized_name': normalized_name,
                    'usage_percentage': usage_pct
                })
                total_cards += 1
            
            # Sort cards within section by usage percentage (descending)
            section_data.sort(key=lambda x: x['usage_percentage'], reverse=True)
            hero_sections[section] = section_data
        
        hero_data[hero_id] = {
            'hero_deck_percentage': hero_pct,
            'total_unique_cards': total_cards,
            'sections': hero_sections
        }
        
        # Be nice to the server
        time.sleep(1)
    
    # Save results
    output_data = {
        'metadata': {
            'total_heroes': len(hero_data),
            'generation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'description': 'Card popularity data organized by hero. Each hero contains cards sorted by section (equipment, weapon, etc.) and usage percentage.'
        },
        'heroes': hero_data
    }
    
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Scraping complete!")
    print(f"Total heroes scraped: {len(hero_data)}")
    print(f"Output saved to: {OUTPUT_PATH}")
    print(f"\nHero summary (first 5):")
    for i, (hero_name, data) in enumerate(list(hero_data.items())[:5], 1):
        print(f"  {i}. {hero_name}")
        print(f"     Deck percentage: {data['hero_deck_percentage']:.2f}%")
        print(f"     Total cards: {data['total_unique_cards']}")
        print(f"     Sections: {', '.join(data['sections'].keys())}")

if __name__ == '__main__':
    main()
