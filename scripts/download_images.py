import os
import json
import requests
import re

CARD_JSON_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'card.json')
IMAGE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'images', 'WTR')
SET_ID = 'WTR'

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    ensure_dir(IMAGE_DIR)
    with open(CARD_JSON_PATH, 'r', encoding='utf-8') as f:
        cards = json.load(f)
    count = 0
    for card in cards:
        printed = False
        for printing in card.get('printings', []):
            if printing.get('set_id') == SET_ID and not printed:
                image_url = printing.get('image_url')
                if image_url:
                    raw_name = card.get('name', 'unknown').replace(' ', '_')
                    card_name = re.sub(r'[^A-Za-z0-9_-]', '', raw_name)
                    file_name = f"{card_name}_{printing.get('id', 'unknown')}.png"
                    file_path = os.path.join(IMAGE_DIR, file_name)
                    try:
                        response = requests.get(image_url)
                        response.raise_for_status()
                        with open(file_path, 'wb') as img_file:
                            img_file.write(response.content)
                        print(f"Downloaded: {file_name}")
                        count += 1
                        printed = True
                    except Exception as e:
                        print(f"Failed to download {file_name}: {e}")
    print(f"Total images downloaded for set {SET_ID}: {count}")

if __name__ == "__main__":
    main()
