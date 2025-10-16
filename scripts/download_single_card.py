import os
import json
import requests
import re
import argparse

CARD_JSON_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'card.json')

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    parser = argparse.ArgumentParser(description="Download all printings of a specific card")
    parser.add_argument("card_name", help="Name of the card to download (case-insensitive)")
    parser.add_argument("--force", action="store_true", help="Re-download even if file exists")
    args = parser.parse_args()

    card_name_lower = args.card_name.lower()

    with open(CARD_JSON_PATH, 'r', encoding='utf-8') as f:
        cards = json.load(f)
    
    # Find the card
    matching_card = None
    for card in cards:
        if card.get('name', '').lower() == card_name_lower:
            matching_card = card
            break
    
    if not matching_card:
        print(f"Card '{args.card_name}' not found in database.")
        return
    
    print(f"Found card: {matching_card['name']}")
    printings = matching_card.get('printings', [])
    print(f"Total printings: {len(printings)}")
    
    downloaded_count = 0
    skipped_count = 0
    failed_count = 0
    
    for printing in printings:
        set_id = printing.get('set_id', 'UNKNOWN')
        image_url = printing.get('image_url')
        printing_id = printing.get('id', 'unknown')
        
        if not image_url:
            print(f"  [{set_id}] No image URL available")
            failed_count += 1
            continue
        
        # Create directory for this set if needed
        image_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'images', set_id)
        ensure_dir(image_dir)
        
        # Generate filename
        raw_name = matching_card.get('name', 'unknown').replace(' ', '_')
        card_name_clean = re.sub(r'[^A-Za-z0-9_-]', '', raw_name)
        file_name = f"{card_name_clean}_{printing_id}.png"
        file_path = os.path.join(image_dir, file_name)
        
        # Check if file exists
        if os.path.exists(file_path) and not args.force:
            print(f"  [{set_id}] Already exists: {file_name}")
            skipped_count += 1
            continue
        
        # Download the image
        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            with open(file_path, 'wb') as img_file:
                img_file.write(response.content)
            print(f"  [{set_id}] Downloaded: {file_name}")
            downloaded_count += 1
        except Exception as e:
            print(f"  [{set_id}] Failed to download {file_name}: {e}")
            failed_count += 1
    
    print("\n" + "=" * 60)
    print(f"Summary for '{matching_card['name']}':")
    print(f"  Downloaded: {downloaded_count}")
    print(f"  Skipped (already exists): {skipped_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Total: {len(printings)}")

if __name__ == "__main__":
    main()
