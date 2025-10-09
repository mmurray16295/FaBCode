"""
Download card images for all sets found in card.json
"""
import json
import subprocess
import sys

# Load card.json and find all unique sets
with open('../data/card.json', 'r', encoding='utf-8') as f:
    cards = json.load(f)

sets = set()
for card in cards:
    for printing in card.get('printings', []):
        set_id = printing.get('set_id', '')
        if set_id:
            sets.add(set_id)

sorted_sets = sorted(sets)
print(f"Found {len(sorted_sets)} card sets to download")
print("=" * 60)

# Download images for each set
for i, set_id in enumerate(sorted_sets, 1):
    print(f"\n[{i}/{len(sorted_sets)}] Downloading set: {set_id}")
    print("-" * 60)
    
    try:
        result = subprocess.run(
            [sys.executable, 'download_images_parallel.py', '--set-id', set_id, '--workers', '10'],
            capture_output=False,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            print(f"✓ Completed: {set_id}")
        else:
            print(f"✗ Failed: {set_id} (exit code {result.returncode})")
    
    except Exception as e:
        print(f"✗ Error downloading {set_id}: {e}")

print("\n" + "=" * 60)
print(f"Download complete! Processed {len(sorted_sets)} sets.")
