"""
Add 'is_young' flag to hero entries in card_weights_all_printings.json
by checking the 'types' field in card.json for 'Young' type.
"""

import json
from pathlib import Path

# Paths
WEIGHTS_FILE = Path(r"c:\VS Code\FaB Code\data\card_weights_all_printings.json")
CARD_DATA_FILE = Path(r"c:\VS Code\FaB Code\data\card.json")
OUTPUT_FILE = Path(r"c:\VS Code\FaB Code\data\card_weights_all_printings.json")

def main():
    # Load card.json to get hero types
    print("Loading card.json...")
    with open(CARD_DATA_FILE, 'r', encoding='utf-8') as f:
        card_data = json.load(f)
    
    # Build a mapping of hero name -> is_young
    # Use EXACT names only (no fuzzy matching needed)
    hero_young_map = {}
    for card in card_data:
        if 'Hero' in card.get('types', []):
            hero_name = card['name']
            is_young = 'Young' in card.get('types', [])
            hero_young_map[hero_name] = is_young
            if is_young:
                print(f"  Found young hero: {hero_name}")
    
    print(f"\nTotal heroes in card.json: {len(hero_young_map)}")
    print(f"Young heroes: {sum(hero_young_map.values())}")
    print(f"Adult heroes: {sum(not v for v in hero_young_map.values())}")
    
    # Load weights file
    print("\nLoading card_weights_all_printings.json...")
    with open(WEIGHTS_FILE, 'r', encoding='utf-8') as f:
        weights_data = json.load(f)
    
    # Add is_young flag to each hero in both formats
    young_added = 0
    adult_added = 0
    not_found = []
    
    for format_name in ['cc', 'blitz']:
        if format_name not in weights_data['formats']:
            continue
        
        print(f"\nProcessing {format_name.upper()} format...")
        for hero_name in weights_data['formats'][format_name]:
            # Use exact match only
            if hero_name in hero_young_map:
                is_young = hero_young_map[hero_name]
                weights_data['formats'][format_name][hero_name]['is_young'] = is_young
                if is_young:
                    young_added += 1
                    print(f"  ✓ {hero_name} -> Young")
                else:
                    adult_added += 1
            else:
                # Hero not found in card.json, mark as adult (safer default for CC)
                weights_data['formats'][format_name][hero_name]['is_young'] = False
                not_found.append((format_name, hero_name))
                adult_added += 1
                print(f"  ⚠ {hero_name} -> Not found in card.json, defaulting to Adult")
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Young heroes marked: {young_added}")
    print(f"  Adult heroes marked: {adult_added}")
    print(f"  Heroes not found: {len(not_found)}")
    
    if not_found:
        print(f"\nHeroes not found in card.json:")
        for fmt, hero in not_found:
            print(f"  [{fmt}] {hero}")
    
    # Save updated weights file
    print(f"\nSaving updated weights to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(weights_data, f, indent=2, ensure_ascii=False)
    
    print("✓ Done!")
    
    # Verify the changes
    print("\nVerifying changes...")
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        verify_data = json.load(f)
    
    for format_name in ['cc', 'blitz']:
        if format_name not in verify_data['formats']:
            continue
        young_count = sum(
            1 for hero_data in verify_data['formats'][format_name].values()
            if hero_data.get('is_young', False)
        )
        total_count = len(verify_data['formats'][format_name])
        print(f"  {format_name.upper()}: {young_count} young / {total_count} total heroes")

if __name__ == "__main__":
    main()
