import json
import os
from collections import Counter

CARD_JSON_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'card.json')

def main():
    with open(CARD_JSON_PATH, 'r', encoding='utf-8') as f:
        cards = json.load(f)
    print(f"Total cards: {len(cards)}\n")
    print("Sample card entry:")
    print(json.dumps(cards[0], indent=2))
    print("\nAvailable fields:")
    print(list(cards[0].keys()))
    # Count cards per set using set_id from printings
    set_counter = Counter()
    for card in cards:
        printings = card.get('printings', [])
        for printing in printings:
            set_id = printing.get('set_id', 'UNKNOWN')
            set_counter[set_id] += 1
    print("\nNumber of cards per set (by set_id in printings):")
    for set_id, count in set_counter.items():
        print(f"{set_id}: {count}")

if __name__ == "__main__":
    main()
