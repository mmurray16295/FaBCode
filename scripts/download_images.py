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
    parser = argparse.ArgumentParser(description="Download card images for a given set id")
    parser.add_argument("--set-id", default="SEA", help="Set code to download (e.g., SEA, WTR)")
    parser.add_argument("--output-dir", default=None, help="Optional output directory; defaults to data/images/<SET_ID>")
    args = parser.parse_args()

    set_id = args.set_id.upper()
    image_dir = args.output_dir or os.path.join(os.path.dirname(__file__), '..', 'data', 'images', set_id)

    ensure_dir(image_dir)
    with open(CARD_JSON_PATH, 'r', encoding='utf-8') as f:
        cards = json.load(f)
    count = 0
    for card in cards:
        printed = False
        for printing in card.get('printings', []):
            if printing.get('set_id') == set_id and not printed:
                image_url = printing.get('image_url')
                if image_url:
                    raw_name = card.get('name', 'unknown').replace(' ', '_')
                    card_name = re.sub(r'[^A-Za-z0-9_-]', '', raw_name)
                    file_name = f"{card_name}_{printing.get('id', 'unknown')}.png"
                    file_path = os.path.join(image_dir, file_name)
                    if os.path.exists(file_path):
                        # Skip existing file
                        count += 1
                        printed = True
                        continue
                    try:
                        response = requests.get(image_url, timeout=30)
                        response.raise_for_status()
                        with open(file_path, 'wb') as img_file:
                            img_file.write(response.content)
                        print(f"Downloaded: {file_name}")
                        count += 1
                        printed = True
                    except Exception as e:
                        print(f"Failed to download {file_name}: {e}")
    print(f"Total images present/downloaded for set {set_id}: {count}")

if __name__ == "__main__":
    main()
