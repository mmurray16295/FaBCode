#!/usr/bin/env python3
"""
Test wrapper for scraping individual heroes
Allows testing the scraper with specific heroes before running full scrape
"""

import sys
import json
from pathlib import Path
from scrape_popularity import CardPopularityScraper

def test_hero(hero_name: str, format_code: str = 'cc'):
    """Test scraping a single hero"""
    
    print("=" * 80)
    print(f"TESTING HERO SCRAPER")
    print("=" * 80)
    print(f"Hero: {hero_name}")
    print(f"Format: {format_code}")
    print()
    
    # Paths
    card_json = Path(__file__).parent.parent.parent.parent / 'data' / 'card.json'
    heroes_json = Path(__file__).parent.parent.parent.parent / 'data' / 'heroes_card.json'
    
    # Create scraper
    scraper = CardPopularityScraper(str(card_json), str(heroes_json))
    scraper.load_data()
    
    # Verify hero exists
    adult_heroes = [h['name'] for h in scraper.heroes_data['adult_heroes']]
    young_heroes = [h['name'] for h in scraper.heroes_data['young_heroes']]
    
    if hero_name not in adult_heroes and hero_name not in young_heroes:
        print(f"❌ Hero '{hero_name}' not found!")
        print()
        print("Did you mean one of these?")
        # Find similar names
        all_heroes = adult_heroes + young_heroes
        similar = [h for h in all_heroes if hero_name.lower() in h.lower()][:10]
        for h in similar:
            hero_type = "Adult" if h in adult_heroes else "Young"
            print(f"  - {h} ({hero_type})")
        return
    
    # Check if hero is in adult or young list
    is_young = hero_name in young_heroes
    hero_type = "Young" if is_young else "Adult"
    
    print(f"Hero Type: {hero_type}")
    
    # Get legality
    hero_data = None
    if is_young:
        hero_data = next(h for h in scraper.heroes_data['young_heroes'] if h['name'] == hero_name)
    else:
        hero_data = next(h for h in scraper.heroes_data['adult_heroes'] if h['name'] == hero_name)
    
    legality = hero_data['legality']
    print(f"Legality:")
    print(f"  CC Legal: {legality.get('cc_legal', False)}")
    print(f"  LL Legal: {legality.get('ll_legal', False)}")
    print(f"  Blitz Legal: {legality.get('blitz_legal', False)}")
    print()
    
    # Check if format is valid for this hero
    format_legality_map = {
        'cc': 'cc_legal',
        'll': 'll_legal',
        'blitz': 'blitz_legal'
    }
    
    if not legality.get(format_legality_map[format_code], False):
        print(f"⚠️ Warning: {hero_name} is not legal in {format_code.upper()} format!")
        print()
    
    # Scrape
    print("=" * 80)
    print("SCRAPING")
    print("=" * 80)
    result = scraper.scrape_hero_page(hero_name, format_code)
    
    if result:
        print()
        print("=" * 80)
        print("RESULTS")
        print("=" * 80)
        print(f"Deck Count: {result['deck_count']}")
        print(f"Total Unique Cards: {result['total_unique_cards']}")
        print(f"Sections: {len(result['sections'])}")
        print()
        
        for section_name, cards in result['sections'].items():
            print(f"{section_name.upper()} ({len(cards)} cards):")
            for i, card in enumerate(cards[:5], 1):  # Show top 5
                print(f"  {i}. {card['card_name']} ({card['card_id']}) - {card['usage_percentage']:.1f}%")
            if len(cards) > 5:
                print(f"  ... and {len(cards) - 5} more")
            print()
        
        # Save test output
        output_file = Path(__file__).parent / 'test_output.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'hero_name': hero_name,
                'format': format_code,
                'result': result
            }, f, indent=2)
        print(f"✓ Full results saved to: {output_file}")
    else:
        print()
        print("❌ No data found for this hero/format combination")

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_scraper.py <hero_name> [format]")
        print()
        print("Examples:")
        print("  python test_scraper.py \"Boltyn\" cc")
        print("  python test_scraper.py \"Ser Boltyn, Breaker of Dawn\" cc")
        print("  python test_scraper.py \"Dorinthea\" ll")
        print()
        print("Formats: cc (default), ll, blitz")
        return
    
    hero_name = sys.argv[1]
    format_code = sys.argv[2] if len(sys.argv) > 2 else 'cc'
    
    if format_code not in ['cc', 'll', 'blitz']:
        print(f"Invalid format: {format_code}")
        print("Valid formats: cc, ll, blitz")
        return
    
    test_hero(hero_name, format_code)

if __name__ == '__main__':
    main()
