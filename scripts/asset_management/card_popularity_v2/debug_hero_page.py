#!/usr/bin/env python3
"""Debug script to see what data is available on a hero page"""

import requests
from bs4 import BeautifulSoup
import json
import sys

def debug_hero_page(hero_url):
    url = f"https://fabrec.gg/hero/{hero_url}"
    print(f"Fetching: {url}")
    print("=" * 80)
    
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find the Next.js data
    script_tag = soup.find('script', {'id': '__NEXT_DATA__'})
    if not script_tag:
        print("ERROR: No __NEXT_DATA__ found!")
        return
    
    data = json.loads(script_tag.string)
    
    # Save full data to file for inspection
    with open('debug_full_data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print("✓ Full data saved to: debug_full_data.json")
    print()
    
    # Navigate to the relevant sections
    page_props = data.get('props', {}).get('pageProps', {})
    
    print("Available keys in pageProps:")
    for key in page_props.keys():
        print(f"  - {key}")
    print()
    
    if 'cardData' in page_props:
        card_data = page_props['cardData']
        print("cardData keys:")
        for key in card_data.keys():
            print(f"  - {key}")
        print()
        
        print(f"total_decks: {card_data.get('total_decks', 'N/A')}")
        print(f"name: {card_data.get('name', 'N/A')}")
        print()
        
        if 'container' in card_data:
            container = card_data['container']
            print("container keys:")
            for key in container.keys():
                print(f"  - {key}")
            print()
            
            if 'cardlists' in container:
                cardlists = container['cardlists']
                print("cardlists keys:")
                for key in cardlists.keys():
                    print(f"  - {key}")
                print()
                
                if 'jsonData' in cardlists:
                    json_data = cardlists['jsonData']
                    print("jsonData sections:")
                    for section_name, section_cards in json_data.items():
                        if isinstance(section_cards, list):
                            print(f"  - {section_name}: {len(section_cards)} cards")
                        else:
                            print(f"  - {section_name}: {type(section_cards)}")
                    print()
                    
                    # Show first card from first section
                    if json_data:
                        first_section = list(json_data.keys())[0]
                        first_cards = json_data[first_section]
                        if isinstance(first_cards, list) and len(first_cards) > 0:
                            print(f"Example card from {first_section}:")
                            print(json.dumps(first_cards[0], indent=2))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python debug_hero_page.py <hero-url>")
        print("Example: python debug_hero_page.py dorinthea-ironsong")
        sys.exit(1)
    
    debug_hero_page(sys.argv[1])
