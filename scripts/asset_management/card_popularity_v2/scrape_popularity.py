#!/usr/bin/env python3
"""
Card Popularity Scraper V2
Scrapes card usage data from fabrec.gg by hero across multiple formats (CC, LL, Blitz).
Uses heroes_card.json to ensure complete hero coverage with no gaps.
"""

import json
import re
import requests
from bs4 import BeautifulSoup
import time
from pathlib import Path
from typing import Dict, List, Optional
import argparse
from datetime import datetime

class CardPopularityScraper:
    """Scraper for card popularity data organized by hero and format"""
    
    def __init__(self, card_json_path: str, heroes_json_path: str):
        self.card_json_path = card_json_path
        self.heroes_json_path = heroes_json_path
        self.card_lookup = {}
        self.heroes_data = {}
        
    def load_data(self):
        """Load card.json and heroes_card.json"""
        print("=" * 80)
        print("LOADING DATA")
        print("=" * 80)
        
        # Load card database
        print(f"Loading card database: {self.card_json_path}")
        with open(self.card_json_path, 'r', encoding='utf-8') as f:
            cards = json.load(f)
        
        # Create lookup by card ID
        for card in cards:
            for printing in card.get('printings', []):
                card_id = printing.get('id', '')
                if card_id:
                    self.card_lookup[card_id] = {
                        'name': card['name'],
                        'types': card.get('types', []),
                        'card_id': card_id
                    }
        
        print(f"  Loaded {len(self.card_lookup)} card printings")
        
        # Load heroes
        print(f"Loading heroes: {self.heroes_json_path}")
        with open(self.heroes_json_path, 'r', encoding='utf-8') as f:
            self.heroes_data = json.load(f)
        
        print(f"  Adult heroes: {self.heroes_data['metadata']['total_adult_heroes']}")
        print(f"  Young heroes: {self.heroes_data['metadata']['total_young_heroes']}")
        print()
    
    def normalize_hero_name_to_url(self, hero_name: str) -> str:
        """Convert hero name to fabrec.gg URL format"""
        # Example: "Ser Boltyn, Breaker of Dawn" -> "ser-boltyn-breaker-of-dawn"
        # Special character mappings (fabrec.gg uses these)
        char_map = {
            'ð': 'd',  # eth -> d (Jarl Vetreiði -> jarl-vetreidi)
            'í': 'i',  # i with acute
            'á': 'a',  # a with acute
            'é': 'e',  # e with acute
            'ó': 'o',  # o with acute
            'ú': 'u',  # u with acute
        }
        
        normalized = hero_name.lower()
        # Replace special characters
        for old_char, new_char in char_map.items():
            normalized = normalized.replace(old_char, new_char)
        # Remove punctuation
        normalized = normalized.replace(',', '')
        normalized = normalized.replace("'", '')
        normalized = normalized.replace('!', '')
        # Remove any remaining non-alphanumeric except spaces and hyphens
        normalized = re.sub(r'[^a-z0-9\s-]', '', normalized)
        # Convert spaces to hyphens
        normalized = re.sub(r'\s+', '-', normalized.strip())
        return normalized
    
    def scrape_hero_page(self, hero_name: str, format_code: str) -> Optional[Dict]:
        """
        Scrape a hero's card usage data from fabrec.gg
        
        Args:
            hero_name: Full hero name from heroes_card.json
            format_code: 'cc', 'll', or 'blitz'
        
        Returns:
            Dict with sections and card data, or None if failed
        """
        hero_url = self.normalize_hero_name_to_url(hero_name)
        
        # The hero page URL is simple: https://fabrec.gg/hero/{hero-name}
        # The page itself shows different format tabs, we scrape the default view
        # which typically shows the most popular format for that hero
        url = f"https://fabrec.gg/hero/{hero_url}"
        
        try:
            print(f"  Scraping: {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            script_tag = soup.find('script', {'id': '__NEXT_DATA__'})
            
            if not script_tag:
                print(f"    ⚠ No data found")
                return None
            
            data = json.loads(script_tag.string)
            page_props = data.get('props', {}).get('pageProps', {})
            card_data = page_props.get('cardData', {})
            
            if not card_data:
                print(f"    ⚠ No card data")
                return None
            
            # Get container
            container = card_data.get('container', {})
            
            # Get deck count (it's in container.totalDecks, not card_data.total_decks)
            deck_count = container.get('totalDecks', 0)
            
            if deck_count == 0:
                print(f"    ℹ No decks (0 decks in format)")
                return None
            
            # Get cardlists
            cardlists = container.get('cardlists', {}).get('jsonData', {})
            
            if not cardlists:
                print(f"    ⚠ No cardlists")
                return None
            
            # Process sections
            sections = {}
            total_cards = 0
            
            for section_name, cards in cardlists.items():
                if not isinstance(cards, list) or len(cards) == 0:
                    continue
                
                section_cards = []
                for card_info in cards:
                    card_id_full = card_info.get('card', '')
                    if not card_id_full:
                        continue
                    
                    # Remove suffix like "-CF"
                    card_id = re.sub(r'-[A-Z]+$', '', card_id_full)
                    
                    # Get usage data
                    usage_decimal = card_info.get('maxPopularity', 0.0)
                    usage_pct = usage_decimal * 100
                    
                    # Look up card name
                    if card_id in self.card_lookup:
                        card_name = self.card_lookup[card_id]['name']
                        section_cards.append({
                            'card_name': card_name,
                            'card_id': card_id,
                            'usage_percentage': usage_pct
                        })
                        total_cards += 1
                    else:
                        print(f"    ⚠ Unknown card ID: {card_id}")
                
                if section_cards:
                    # Sort by usage percentage
                    section_cards.sort(key=lambda x: x['usage_percentage'], reverse=True)
                    sections[section_name] = section_cards
            
            print(f"    ✓ Found {total_cards} cards across {len(sections)} sections, {deck_count} decks")
            
            return {
                'deck_count': deck_count,
                'total_unique_cards': total_cards,
                'sections': sections
            }
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
            return None
    
    def scrape_format(self, format_code: str, hero_list_key: str) -> tuple[Dict, List[str]]:
        """
        Scrape all heroes for a specific format
        
        Args:
            format_code: 'cc' or 'blitz'
            hero_list_key: 'adult_heroes' or 'young_heroes'
        
        Returns:
            Tuple of (hero_data_dict, failed_heroes_list)
        """
        format_names = {
            'cc': 'Classic Constructed',
            'blitz': 'Blitz'
        }
        
        print("\n" + "=" * 80)
        print(f"SCRAPING {format_names[format_code].upper()}")
        print("=" * 80)
        
        heroes = self.heroes_data[hero_list_key]
        legality_key = f'{format_code}_legal'
        
        # Filter to only legal heroes for this format
        legal_heroes = [h for h in heroes if h['legality'].get(legality_key)]
        
        print(f"Found {len(legal_heroes)} legal heroes in {format_names[format_code]}")
        print()
        
        format_data = {}
        failed_heroes = []
        total_decks_scraped = 0
        heroes_with_data = 0
        heroes_no_data = 0
        
        for i, hero in enumerate(legal_heroes, 1):
            hero_name = hero['name']
            print(f"[{i}/{len(legal_heroes)}] {hero_name}")
            
            result = self.scrape_hero_page(hero_name, format_code)
            
            if result:
                format_data[hero_name] = {
                    'deck_count': result['deck_count'],
                    'total_unique_cards': result['total_unique_cards'],
                    'sections': result['sections'],
                    'data_source': 'scraped'
                }
                total_decks_scraped += result['deck_count']
                heroes_with_data += 1
            else:
                heroes_no_data += 1
                failed_heroes.append(hero_name)
            
            # Be polite to the server
            time.sleep(1)
        
        print()
        print("=" * 80)
        print(f"{format_names[format_code]} Summary:")
        print(f"  Heroes with data: {heroes_with_data}")
        print(f"  Heroes without data: {heroes_no_data}")
        print(f"  Total decks scraped: {total_decks_scraped}")
        if failed_heroes:
            print(f"\n  Failed heroes:")
            for hero_name in failed_heroes:
                print(f"    - {hero_name}")
        print("=" * 80)
        
        return format_data, failed_heroes
    
    def scrape_all_formats(self, formats: List[str]) -> Dict:
        """Scrape specified formats and return unified data structure"""
        results = {
            'metadata': {
                'version': '2.0',
                'generated': datetime.now().isoformat(),
                'formats_scraped': formats,
                'source': 'fabrec.gg',
                'failed_heroes': {}
            },
            'formats': {}
        }
        
        for format_code in formats:
            if format_code == 'cc':
                # Adult heroes for CC
                hero_data, failed = self.scrape_format(format_code, 'adult_heroes')
                results['formats'][format_code] = hero_data
                if failed:
                    results['metadata']['failed_heroes'][format_code] = failed
            elif format_code == 'blitz':
                # Young heroes for Blitz
                hero_data, failed = self.scrape_format(format_code, 'young_heroes')
                results['formats'][format_code] = hero_data
                if failed:
                    results['metadata']['failed_heroes'][format_code] = failed
        
        return results
    
    def save_results(self, data: Dict, output_path: str):
        """Save scraped data to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Results saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Scrape card popularity by hero across formats')
    parser.add_argument('--formats', nargs='+', choices=['cc', 'blitz'], 
                       default=['cc', 'blitz'],
                       help='Formats to scrape (default: all)')
    parser.add_argument('--output', default='card_weights_all_printings.json',
                       help='Output filename (saved to project data/ folder)')
    
    args = parser.parse_args()
    
    # Paths
    card_json = Path(__file__).parent.parent.parent.parent / 'data' / 'card.json'
    heroes_json = Path(__file__).parent.parent.parent.parent / 'data' / 'heroes_card.json'
    output_path = Path(__file__).parent.parent.parent.parent / 'data' / args.output
    
    # Create scraper and run
    scraper = CardPopularityScraper(str(card_json), str(heroes_json))
    scraper.load_data()
    
    results = scraper.scrape_all_formats(args.formats)
    scraper.save_results(results, str(output_path))

if __name__ == '__main__':
    main()
