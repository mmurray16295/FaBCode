"""
Test script to verify popularity weights loading and filtering functionality.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the functions from the main script
import json
import re

def canonicalize_name(name: str) -> str:
    """Collapse reprints by stripping trailing set codes."""
    m = re.match(r"^(.*)_([A-Z]{2,5}\d{1,5})$", name)
    return m.group(1) if m else name

def load_popularity_weights(weights_path):
    """Load popularity weights from JSON file."""
    if not os.path.exists(weights_path):
        print(f"[warning] Popularity weights file not found: {weights_path}")
        return {}, {}
    
    try:
        with open(weights_path, 'r') as f:
            data = json.load(f)
        
        raw_weights = data.get('weights', {})
        
        # Build dict of canonicalized names -> total weights
        weights_dict = {}
        for card_key, weight_data in raw_weights.items():
            canon_name = canonicalize_name(card_key)
            total_weight = weight_data.get('total_weight', 0.0)
            
            # Accumulate weights for cards with multiple printings
            if canon_name in weights_dict:
                weights_dict[canon_name] += total_weight
            else:
                weights_dict[canon_name] = total_weight
        
        # Sort by weight descending to create rank mapping
        sorted_cards = sorted(weights_dict.items(), key=lambda x: x[1], reverse=True)
        rank_dict = {card: rank + 1 for rank, (card, _) in enumerate(sorted_cards)}
        
        print(f"[init] Loaded popularity weights for {len(weights_dict)} cards (ranked 1-{len(rank_dict)})")
        return weights_dict, rank_dict
    
    except Exception as e:
        print(f"[error] Failed to load popularity weights: {e}")
        return {}, {}

# Test the functions
if __name__ == "__main__":
    weights_path = "data/card_popularity_weights.json"
    
    print("Testing popularity weights loading...")
    print("-" * 60)
    
    weights_dict, rank_dict = load_popularity_weights(weights_path)
    
    if weights_dict:
        print(f"\nTotal unique cards: {len(weights_dict)}")
        print(f"Rank range: 1 to {len(rank_dict)}")
        
        # Show top 10
        print("\nTop 10 most popular cards:")
        sorted_cards = sorted(weights_dict.items(), key=lambda x: x[1], reverse=True)
        for i, (card, weight) in enumerate(sorted_cards[:10], 1):
            rank = rank_dict[card]
            print(f"  {rank:3d}. {card:40s} (weight: {weight:.6f})")
        
        # Show some examples of rank ranges
        print("\n" + "=" * 60)
        print("Testing rank range filtering:")
        print("=" * 60)
        
        test_ranges = [
            (1, 10, "Top 10"),
            (1, 300, "Top 300"),
            (301, 600, "Ranks 301-600"),
            (601, 900, "Ranks 601-900"),
        ]
        
        for min_rank, max_rank, description in test_ranges:
            cards_in_range = [card for card, rank in rank_dict.items() 
                            if min_rank <= rank <= max_rank]
            print(f"\n{description} ({min_rank}-{max_rank}): {len(cards_in_range)} cards")
            if cards_in_range and len(cards_in_range) <= 10:
                for card in cards_in_range[:10]:
                    print(f"  - {card} (rank {rank_dict[card]})")
        
        # Test canonicalization
        print("\n" + "=" * 60)
        print("Testing name canonicalization:")
        print("=" * 60)
        
        test_names = [
            "Fyendals_Spring_Tunic_WTR150",
            "Enlightened_Strike_WTR159",
            "Snatch_SEA169",
            "Command_and_Conquer",  # No set code
        ]
        
        for test_name in test_names:
            canon = canonicalize_name(test_name)
            print(f"  {test_name:40s} -> {canon}")
            if canon in rank_dict:
                print(f"    Rank: {rank_dict[canon]}, Weight: {weights_dict[canon]:.6f}")
    else:
        print("Failed to load weights - check file path and format")
