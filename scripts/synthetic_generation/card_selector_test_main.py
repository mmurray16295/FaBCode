"""
Test script to simulate card selection and analyze distribution without generating images.

This script runs the card selection logic from Core_Playmat_Generator.py but skips
all image generation and file I/O. It records what cards WOULD have been selected
and provides statistics on the distribution.

Usage:
    python test_card_distribution.py --iterations 25000 --selector weighted
    python test_card_distribution.py --iterations 25000 --selector smooth
    python test_card_distribution.py --iterations 1000 --selector both
"""

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

from card_selector import CardSelector, select_format
from card_selector_smooth import SmoothCardSelector


# Zone definitions (from Core_Playmat_Generator.py)
ZONE_NAMES = {
    0: 'Arms',
    1: 'Arms 2',
    2: 'Banish',
    3: 'Banish 2',
    4: 'Card',
    5: 'Chest',
    6: 'Chest 2',
    7: 'Combat Chain 1',
    8: 'Combat Chain 2',
    9: 'Graveyard',
    10: 'Graveyard 2',
    11: 'Head',
    12: 'Head 2',
    13: 'Hero',
    14: 'Hero 2',
    15: 'Legs',
    16: 'Legs 2',
    17: 'Pitch',
    18: 'Pitch 2',
    19: 'Ref',
    20: 'Weapon',
    21: 'Weapon 2',
    22: 'Weapon or Off-Hand',
    23: 'Weapon or Off-Hand 2',
    24: 'Window'
}


def create_mock_zones() -> List[Dict]:
    """Create mock zone data representing a typical playmat layout."""
    zones = []
    for class_id, zone_name in ZONE_NAMES.items():
        # Skip Window zone - not used for cards
        if zone_name == 'Window':
            continue
        
        zones.append({
            'class_id': class_id,
            'zone_name': zone_name,
            'center_x': 0.5,  # Dummy coordinates
            'center_y': 0.5,
            'width': 0.1,
            'height': 0.1
        })
    
    return zones


def partition_zones(zones: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    """
    Split zones into hero1/hero2 categories (from Core_Playmat_Generator.py).
    
    Returns:
        Tuple of (hero1_zones, hero2_zones, hero1_available_zones, hero2_available_zones)
    """
    hero1_zones = []
    hero2_zones = []
    hero1_available = []
    hero2_available = []
    
    for z in zones:
        class_id = z['class_id']
        name = z['zone_name']
        ends_with_2 = name.endswith(' 2')
        
        # Hero 1 zones (for counting)
        if class_id not in [14, 24] and name not in ['Hero 2', 'Window'] and not ends_with_2:
            hero1_zones.append(z)
        
        # Hero 2 zones (for counting)
        if ends_with_2 or class_id == 14:
            hero2_zones.append(z)
        
        # Hero 1 available zones (for card selection)
        if class_id not in [13, 14, 24] and not ends_with_2:
            hero1_available.append(z)
        
        # Hero 2 available zones (for card selection)
        if ends_with_2 and class_id not in [13, 14, 24]:
            hero2_available.append(z)
    
    return hero1_zones, hero2_zones, hero1_available, hero2_available


def get_zone_sort_key(zone_name: str) -> int:
    """Get sort key for zone ordering (from Core_Playmat_Generator.py)."""
    sort_keys = {
        'Weapon': 0,
        'Weapon 2': 0,
        'Weapon or Off-Hand': 1,
        'Weapon or Off-Hand 2': 1
    }
    return sort_keys.get(zone_name, 2)


def simulate_single_playmat(selector, zones: List[Dict]) -> Dict:
    """
    Simulate a single playmat generation and return card selection data.
    
    Returns:
        Dict with:
            - format: str
            - hero1: str (hero name)
            - hero2: str (hero name)
            - hero1_cards: List[str] (card names)
            - hero2_cards: List[str] (card names)
            - combat_chain_cards: List[str] (card names)
            - all_cards: List[str] (all card names)
    """
    # Select format
    format = select_format()
    
    # Select two heroes
    hero1_key, hero1_card, hero1_weights = selector.select_random_hero(format)
    hero2_key, hero2_card, hero2_weights = selector.select_random_hero(format)
    
    # Build card pools for both heroes
    hero1_pools = selector.build_card_pools(hero1_card, hero1_weights, format)
    hero2_pools = selector.build_card_pools(hero2_card, hero2_weights, format)
    
    # Pre-build complete card pool lists
    all_hero1_cards = (hero1_pools['weighted'] + hero1_pools['generic'] + 
                       hero1_pools['class_only'] + hero1_pools['talent_only'] + 
                       hero1_pools['both'])
    all_hero2_cards = (hero2_pools['weighted'] + hero2_pools['generic'] + 
                       hero2_pools['class_only'] + hero2_pools['talent_only'] + 
                       hero2_pools['both'])
    
    # Partition zones
    hero1_zones, hero2_zones, hero1_available_zones, hero2_available_zones = partition_zones(zones)
    
    # Sort zones
    hero1_available_zones.sort(key=lambda z: get_zone_sort_key(z['zone_name']))
    hero2_available_zones.sort(key=lambda z: get_zone_sort_key(z['zone_name']))
    
    # Select cards for Hero 1
    hero1_cards = []
    hero1_weapon_state = {'weapon_is_2h': False}
    
    for zone in hero1_available_zones:
        if zone['zone_name'] in ['Combat Chain 1', 'Combat Chain 2']:
            continue
        
        card = selector.select_card_for_zone(
            hero1_pools, 
            zone['zone_name'], 
            all_hero1_cards, 
            hero1_weapon_state,
            pitch_weighting=True
        )
        
        if card:
            hero1_cards.append(card['name'])
            
            # Track 2H weapon
            if zone['zone_name'] == 'Weapon' and '2H' in card.get('types', []):
                hero1_weapon_state['weapon_is_2h'] = True
    
    # Select cards for Hero 2
    hero2_cards = []
    hero2_weapon_state = {'weapon_is_2h': False}
    
    for zone in hero2_available_zones:
        if zone['zone_name'] in ['Combat Chain 1', 'Combat Chain 2']:
            continue
        
        card = selector.select_card_for_zone(
            hero2_pools, 
            zone['zone_name'], 
            all_hero2_cards, 
            hero2_weapon_state,
            pitch_weighting=True
        )
        
        if card:
            hero2_cards.append(card['name'])
            
            # Track 2H weapon
            if zone['zone_name'] == 'Weapon 2' and '2H' in card.get('types', []):
                hero2_weapon_state['weapon_is_2h'] = True
    
    # Select combat chain cards (4-15 total)
    combat_chain_total = random.randint(4, 15)
    combat_chain_cards = []
    
    # Split between both heroes randomly
    for _ in range(combat_chain_total):
        if random.random() < 0.5:
            card = selector.select_card(hero1_pools)
        else:
            card = selector.select_card(hero2_pools)
        
        if card:
            combat_chain_cards.append(card['name'])
    
    # Combine all cards
    all_cards = hero1_cards + hero2_cards + combat_chain_cards
    
    return {
        'format': format,
        'hero1': hero1_card['name'],
        'hero2': hero2_card['name'],
        'hero1_cards': hero1_cards,
        'hero2_cards': hero2_cards,
        'combat_chain_cards': combat_chain_cards,
        'all_cards': all_cards
    }


def analyze_results(results: List[Dict]) -> Dict:
    """
    Analyze simulation results and return statistics.
    
    Returns:
        Dict with various statistics about card distribution
    """
    # Count formats
    format_counts = Counter(r['format'] for r in results)
    
    # Count heroes
    hero_counts = Counter()
    for r in results:
        hero_counts[r['hero1']] += 1
        hero_counts[r['hero2']] += 1
    
    # Count all card occurrences (including heroes!)
    card_counts = Counter()
    for r in results:
        # Count regular cards
        for card in r['all_cards']:
            card_counts[card] += 1
        # Count heroes as well (they are placed in hero zones)
        card_counts[r['hero1']] += 1
        card_counts[r['hero2']] += 1
    
    # Count cards per zone type
    zone_card_counts = {
        'hero1': Counter(),
        'hero2': Counter(),
        'combat_chain': Counter()
    }
    
    for r in results:
        for card in r['hero1_cards']:
            zone_card_counts['hero1'][card] += 1
        for card in r['hero2_cards']:
            zone_card_counts['hero2'][card] += 1
        for card in r['combat_chain_cards']:
            zone_card_counts['combat_chain'][card] += 1
    
    # Calculate statistics
    total_cards = sum(len(r['all_cards']) for r in results)
    avg_cards_per_playmat = total_cards / len(results)
    
    # Load all card names from card.json to ensure 0-count cards are included
    try:
        with open('data/card.json', 'r', encoding='utf-8') as f:
            all_cards_data = json.load(f)
        all_card_names = set(card['name'] for card in all_cards_data)
        
        # Add 0 counts for cards that were never selected
        for card_name in all_card_names:
            if card_name not in card_counts:
                card_counts[card_name] = 0
    except Exception as e:
        print(f"Warning: Could not load card.json to add 0-count cards: {e}")
    
    return {
        'total_simulations': len(results),
        'total_cards_selected': total_cards,
        'avg_cards_per_playmat': avg_cards_per_playmat,
        'unique_cards': len(card_counts),
        'unique_heroes': len(hero_counts),
        'format_distribution': dict(format_counts),
        'top_20_cards': card_counts.most_common(20),
        'top_20_heroes': hero_counts.most_common(20),
        'card_counts': dict(card_counts),
        'hero_counts': dict(hero_counts),
        'zone_card_counts': {
            'hero1': dict(zone_card_counts['hero1']),
            'hero2': dict(zone_card_counts['hero2']),
            'combat_chain': dict(zone_card_counts['combat_chain'])
        }
    }


def print_statistics(stats: Dict):
    """Print formatted statistics to console."""
    print("\n" + "="*80)
    print("SIMULATION STATISTICS")
    print("="*80)
    
    print(f"\nTotal Simulations: {stats['total_simulations']:,}")
    print(f"Total Cards Selected: {stats['total_cards_selected']:,}")
    print(f"Average Cards per Playmat: {stats['avg_cards_per_playmat']:.2f}")
    print(f"Unique Cards Seen: {stats['unique_cards']:,}")
    print(f"Unique Heroes Seen: {stats['unique_heroes']:,}")
    
    print("\n" + "-"*80)
    print("FORMAT DISTRIBUTION")
    print("-"*80)
    total = stats['total_simulations']
    for fmt, count in stats['format_distribution'].items():
        pct = (count / total) * 100
        print(f"  {fmt.upper():8} {count:6,} ({pct:5.2f}%)")
    
    print("\n" + "-"*80)
    print("TOP 20 MOST SELECTED CARDS")
    print("-"*80)
    print(f"{'Rank':<6} {'Count':<10} {'Card Name':<50}")
    print("-"*80)
    for i, (card, count) in enumerate(stats['top_20_cards'], 1):
        pct = (count / stats['total_cards_selected']) * 100
        print(f"{i:<6} {count:<10,} {card:<50} ({pct:.3f}%)")
    
    print("\n" + "-"*80)
    print("TOP 20 MOST SELECTED HEROES")
    print("-"*80)
    print(f"{'Rank':<6} {'Count':<10} {'Hero Name':<50}")
    print("-"*80)
    total_hero_selections = stats['total_simulations'] * 2  # 2 heroes per playmat
    for i, (hero, count) in enumerate(stats['top_20_heroes'], 1):
        pct = (count / total_hero_selections) * 100
        print(f"{i:<6} {count:<10,} {hero:<50} ({pct:.3f}%)")


def save_results(stats: Dict, output_path: str):
    """Save detailed results to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to: {output_path}")


def save_card_list(stats: Dict, output_path: str):
    """
    Save a formatted card list similar to data.yaml format.
    Cards are ordered by count (highest to lowest), then alphabetically on ties.
    Includes ALL cards from card.json, even those with 0 placements.
    """
    # Sort cards by count (descending) then by name (ascending)
    sorted_cards = sorted(
        stats['card_counts'].items(),
        key=lambda x: (-x[1], x[0].lower())  # Negative count for descending, name for alpha
    )
    
    # Count cards with 0 placements
    zero_count_cards = sum(1 for _, count in sorted_cards if count == 0)
    non_zero_cards = len(sorted_cards) - zero_count_cards
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # Write header
        f.write("# Card Distribution Analysis\n")
        f.write(f"# Total Simulations: {stats['total_simulations']:,}\n")
        f.write(f"# Total Cards Selected: {stats['total_cards_selected']:,}\n")
        f.write(f"# Total Cards in Database: {len(sorted_cards):,}\n")
        f.write(f"# Cards Selected (count > 0): {non_zero_cards:,}\n")
        f.write(f"# Cards Never Selected (count = 0): {zero_count_cards:,}\n")
        f.write(f"# Cards listed from most to least placed\n")
        f.write("#\n")
        f.write(f"# Format: Card Name (Count - Percentage)\n")
        f.write("#" + "="*78 + "\n\n")
        
        # Write all cards
        for card_name, count in sorted_cards:
            pct = (count / stats['total_cards_selected']) * 100
            f.write(f"{card_name:<60} # {count:>6,} ({pct:>6.3f}%)\n")
    
    print(f"Card list saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Test card distribution by simulating playmat generation without creating images'
    )
    parser.add_argument(
        '--iterations', '-n',
        type=int,
        default=25000,
        help='Number of playmats to simulate (default: 25000)'
    )
    parser.add_argument(
        '--selector', '-s',
        type=str,
        choices=['weighted', 'smooth', 'both'],
        default='weighted',
        help='Selector type to test (default: weighted)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output JSON file path (default: auto-generated based on selector and timestamp)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress messages'
    )
    
    args = parser.parse_args()
    
    # Create mock zones
    zones = create_mock_zones()
    
    # Determine which selectors to test
    selectors_to_test = []
    if args.selector in ['weighted', 'both']:
        selectors_to_test.append(('weighted', CardSelector()))
    if args.selector in ['smooth', 'both']:
        # Disable state persistence for testing (massive performance improvement)
        selectors_to_test.append(('smooth', SmoothCardSelector(enable_state_persistence=False)))
    
    for selector_name, selector in selectors_to_test:
        print("\n" + "="*80)
        print(f"TESTING {selector_name.upper()} SELECTOR")
        print("="*80)
        print(f"Simulating {args.iterations:,} playmats...")
        
        start_time = time.time()
        results = []
        
        # Run simulations
        for i in range(args.iterations):
            if not args.quiet and (i + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (args.iterations - (i + 1)) / rate
                print(f"  Progress: {i+1:,}/{args.iterations:,} ({(i+1)/args.iterations*100:.1f}%) "
                      f"- Rate: {rate:.1f} playmats/sec - ETA: {remaining:.1f}s")
            
            try:
                result = simulate_single_playmat(selector, zones)
                results.append(result)
            except Exception as e:
                if not args.quiet:
                    print(f"  Warning: Simulation {i+1} failed: {e}")
                continue
        
        elapsed_time = time.time() - start_time
        
        print(f"\nCompleted {len(results):,} simulations in {elapsed_time:.2f} seconds")
        print(f"Average rate: {len(results)/elapsed_time:.1f} playmats/second")
        
        # Analyze results
        stats = analyze_results(results)
        
        # Print statistics
        print_statistics(stats)
        
        # Save to file
        if args.output:
            output_path = args.output
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = f"card_distribution_{selector_name}_{args.iterations}_{timestamp}.json"
        
        save_results(stats, output_path)
        
        # Also save formatted card list
        card_list_path = output_path.replace('.json', '_cards.txt')
        save_card_list(stats, card_list_path)
        
        print("\n" + "="*80)
        print(f"TESTING {selector_name.upper()} SELECTOR COMPLETE")
        print("="*80)


if __name__ == '__main__':
    main()
