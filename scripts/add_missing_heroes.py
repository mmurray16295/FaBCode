#!/usr/bin/env python3
"""
Add missing heroes to card_popularity_weights_by_hero.json by scraping their pages directly
"""

import json
import re
import requests
from bs4 import BeautifulSoup
import time

def load_card_database(card_json_path: str):
    """Load card.json and create lookup by card ID"""
    print(f"Loading card database from {card_json_path}...")
    with open(card_json_path, 'r', encoding='utf-8') as f:
        cards = json.load(f)
    
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

def scrape_hero_page(url: str, card_lookup: dict):
    """Scrape a hero page for card usage data"""
    print(f"\nScraping {url}...")
    
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract Next.js JSON data
    script_tag = soup.find('script', {'id': '__NEXT_DATA__'})
    if not script_tag:
        print(f"ERROR: Could not find data")
        return None
    
    data = json.loads(script_tag.string)
    
    try:
        page_props = data['props']['pageProps']
        
        if 'cardData' not in page_props:
            print("  No cardData found")
            return None
        
        card_data = page_props['cardData']
        
        # Get hero name and deck count
        hero_name = card_data.get('name', '')
        deck_count = card_data.get('total_decks', 0)
        
        print(f"  Hero: {hero_name}")
        print(f"  Decks: {deck_count}")
        
        if 'container' not in card_data or 'cardlists' not in card_data['container']:
            print("  No cardlists found")
            return None
        
        cardlists = card_data['container']['cardlists'].get('jsonData', {})
        
        # Process card sections
        cards_by_section = {}
        
        for section, cards in cardlists.items():
            if not isinstance(cards, list) or len(cards) == 0:
                continue
            print(f"  Processing section: {section} ({len(cards)} cards)")
            
            section_cards = []
            for card_info in cards:
                card_id_full = card_info.get('card', '')
                if not card_id_full:
                    continue
                
                # Remove suffix like "-CF"
                card_id = re.sub(r'-[A-Z]+$', '', card_id_full)
                
                # Get usage percentage
                usage_decimal = card_info.get('maxPopularity', 0.0)
                usage_pct = usage_decimal * 100
                
                # Look up card name
                if card_id in card_lookup:
                    card_name = card_lookup[card_id]['name']
                    section_cards.append({
                        'card_name': card_name,
                        'card_id': card_id,
                        'usage_percentage': usage_pct
                    })
                else:
                    print(f"    WARNING: Card ID {card_id} not found")
            
            if section_cards:
                # Sort by usage percentage
                section_cards.sort(key=lambda x: x['usage_percentage'], reverse=True)
                cards_by_section[section] = section_cards
        
        return {
            'hero_name': hero_name,
            'deck_count': deck_count,
            'sections': cards_by_section
        }
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # Heroes to add
    heroes_to_add = [
        ('puffin-hightail', 'https://fabrec.gg/hero/puffin-hightail'),
        ('gravy-bones-shipwrecked-looter', 'https://fabrec.gg/hero/gravy-bones-shipwrecked-looter'),
        ('marlynn-treasure-hunter', 'https://fabrec.gg/hero/marlynn-treasure-hunter'),
        ('jarl-vetreidi', 'https://fabrec.gg/hero/jarl-vetreidi'),
        ('pleiades-superstar', 'https://fabrec.gg/hero/pleiades-superstar'),
        ('valda-brightaxe', 'https://fabrec.gg/hero/valda-brightaxe'),
    ]
    
    # Load card database
    card_lookup = load_card_database('data/card.json')
    
    # Load existing weights file
    print("\nLoading existing weights file...")
    with open('data/card_popularity_weights_by_hero.json', 'r') as f:
        weights_data = json.load(f)
    
    print(f"Current heroes in file: {len(weights_data['heroes'])}")
    
    # Scrape each missing hero
    added_count = 0
    skipped_count = 0
    
    for hero_key, url in heroes_to_add:
        # Check if already exists
        if hero_key in weights_data['heroes']:
            print(f"\n✓ {hero_key} already exists, skipping...")
            skipped_count += 1
            continue
        
        # Scrape hero data
        hero_data = scrape_hero_page(url, card_lookup)
        
        if hero_data is None:
            print(f"✗ Failed to scrape {hero_key}")
            continue
        
        # Calculate hero percentage (we don't have global total, so use a placeholder)
        # The actual percentage doesn't matter much for card selection
        hero_percentage = 0.01  # Placeholder value
        
        # Add to weights data
        weights_data['heroes'][hero_key] = {
            'hero_deck_percentage': hero_percentage,
            'total_unique_cards': sum(len(cards) for cards in hero_data['sections'].values()),
            'sections': hero_data['sections']
        }
        
        print(f"✓ Added {hero_key} with {sum(len(cards) for cards in hero_data['sections'].values())} cards")
        added_count += 1
        
        # Be nice to the server
        time.sleep(1)
    
    # Update metadata
    weights_data['metadata']['total_heroes'] = len(weights_data['heroes'])
    weights_data['metadata']['generation_timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    
    # Save updated file
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Added: {added_count} heroes")
    print(f"  Skipped (already existed): {skipped_count} heroes")
    print(f"  Total heroes now: {len(weights_data['heroes'])}")
    
    with open('data/card_popularity_weights_by_hero.json', 'w') as f:
        json.dump(weights_data, f, indent=2)
    
    print(f"\n✓ Saved to data/card_popularity_weights_by_hero.json")

if __name__ == '__main__':
    main()
