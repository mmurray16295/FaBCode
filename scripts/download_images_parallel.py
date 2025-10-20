import os
import json
import requests
import re
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

CARD_JSON_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'card.json')

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def download_single_card(card_info):
    """Download a single card image. Returns (success, filename, error_msg)"""
    set_id, image_url, file_name, file_path = card_info
    
    if os.path.exists(file_path):
        return (True, file_name, "already exists")
    
    try:
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        with open(file_path, 'wb') as img_file:
            img_file.write(response.content)
        return (True, file_name, None)
    except Exception as e:
        return (False, file_name, str(e))

def main():
    parser = argparse.ArgumentParser(description="Download card images for a given set id (parallel)")
    parser.add_argument("--set-id", default="SEA", help="Set code to download (e.g., SEA, WTR)")
    parser.add_argument("--output-dir", default=None, help="Optional output directory; defaults to data/images/<SET_ID>")
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel download workers (default: 10)")
    args = parser.parse_args()

    set_id = args.set_id.upper()
    image_dir = args.output_dir or os.path.join(os.path.dirname(__file__), '..', 'data', 'images', set_id)

    ensure_dir(image_dir)
    
    # Load card data
    with open(CARD_JSON_PATH, 'r', encoding='utf-8') as f:
        cards = json.load(f)
    
    # Build list of downloads
    download_tasks = []
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
                    download_tasks.append((set_id, image_url, file_name, file_path))
                    printed = True
    
    total = len(download_tasks)
    print(f"Found {total} cards to download for set {set_id}")
    
    # Download in parallel
    downloaded = 0
    skipped = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_single_card, task): task for task in download_tasks}
        
        for future in as_completed(futures):
            success, file_name, error = future.result()
            if success:
                if error == "already exists":
                    skipped += 1
                else:
                    downloaded += 1
                    if downloaded % 10 == 0:
                        print(f"Downloaded: {downloaded}/{total} ({skipped} skipped, {failed} failed)")
            else:
                failed += 1
                print(f"Failed to download {file_name}: {error}")
    
    print(f"\nTotal images present/downloaded for set {set_id}: {downloaded + skipped}")
    print(f"  Downloaded: {downloaded}")
    print(f"  Skipped (already exists): {skipped}")
    print(f"  Failed: {failed}")

if __name__ == "__main__":
    main()
