"""
Download all FaB token images to use as occluders in synthetic data generation.
"""

import json
import os
from pathlib import Path
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def load_card_database(json_path):
    """Load the card database JSON."""
    print(f"Loading card database from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        cards = json.load(f)
    print(f"Loaded {len(cards)} cards")
    return cards

def is_token(card):
    """Check if a card is a token."""
    return 'Token' in card.get('types', [])

def download_image(url, save_path):
    """Download an image from URL to save_path."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return True, save_path
    except Exception as e:
        return False, f"Error downloading {url}: {e}"

def download_tokens(json_path, output_dir, max_workers=10):
    """Download all token images from the card database."""
    # Load database
    cards = load_card_database(json_path)
    
    # Find all tokens
    tokens = [card for card in cards if is_token(card)]
    print(f"Found {len(tokens)} token cards")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Collect all unique token images to download
    download_tasks = []
    seen_images = set()
    
    for token in tokens:
        token_name = token['name']
        
        # Get all printings
        for printing in token.get('printings', []):
            # Get all image URLs (front and back if applicable)
            for key in ['image_url', 'image_url_back']:
                if key in printing and printing[key]:
                    url = printing[key]
                    
                    # Skip duplicates
                    if url in seen_images:
                        continue
                    seen_images.add(url)
                    
                    # Generate filename: TokenName_ID.ext
                    card_id = printing.get('id', 'unknown')
                    ext = url.split('.')[-1].split('?')[0]  # Get extension, remove query params
                    safe_name = "".join(c for c in token_name if c.isalnum() or c in (' ', '-', '_')).strip()
                    safe_name = safe_name.replace(' ', '_')
                    filename = f"{safe_name}_{card_id}.{ext}"
                    save_path = output_path / filename
                    
                    # Skip if already exists
                    if save_path.exists():
                        continue
                    
                    download_tasks.append((url, save_path, token_name))
    
    print(f"\nDownloading {len(download_tasks)} unique token images...")
    
    # Download with progress bar
    successful = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_image, url, save_path): (url, token_name) 
                   for url, save_path, token_name in download_tasks}
        
        with tqdm(total=len(futures), desc="Downloading tokens") as pbar:
            for future in as_completed(futures):
                url, token_name = futures[future]
                success, result = future.result()
                
                if success:
                    successful += 1
                else:
                    failed += 1
                    print(f"\n{result}")
                
                pbar.update(1)
    
    print(f"\n✓ Download complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output directory: {output_path}")
    
    return successful, failed

if __name__ == "__main__":
    # Paths
    json_path = Path(__file__).parent.parent / "data" / "card.json"
    output_dir = Path(__file__).parent.parent / "data" / "tokens"
    
    # Download tokens
    download_tokens(json_path, output_dir, max_workers=10)
