#!/usr/bin/env python3
"""
Download all card images from all printings in parallel with deduplication.

Features:
- Downloads ALL printings (not just one per card)
- URL-based deduplication (skips identical images)
- Parallel downloads with respectful rate limiting
- Downscales to 250px max dimension for efficiency
- Progress tracking and statistics

Rate Limiting Guidelines:
- Default: 5 req/sec (polite, based on common web scraping best practices)
- Conservative sites: 1-3 req/sec
- Well-resourced CDNs: 10-20 req/sec may be acceptable
- Always respect robots.txt and terms of service

Usage:
    python download_all_printings_parallel.py
    python download_all_printings_parallel.py --max-workers 10 --rate-limit 10
    python download_all_printings_parallel.py --max-size 300
"""

import os
import json
import requests
import re
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from PIL import Image
from io import BytesIO
import time
from collections import defaultdict
import threading

# ===================== CONFIGURATION =====================
CARD_JSON_PATH = '../data/card.json'
OUTPUT_DIR = '../data/images'  # Organize by set within this directory
MAX_DIMENSION = 250  # Max width or height for downscaled images
DEFAULT_MAX_WORKERS = 15  # Match to rate limit for efficiency
REQUEST_TIMEOUT = 30

# Rate limiting: Be polite to the server!
# Guidelines based on common web scraping best practices:
# - 1-3 req/sec: Very conservative (small sites, unknown capacity)
# - 5-10 req/sec: Moderate (most professional sites, CDNs)
# - 10-20 req/sec: Aggressive (large CDNs like Google Cloud Storage)
# Since these are on Google Cloud Storage, 15 req/sec is totally fine
MAX_REQUESTS_PER_SECOND = 15  # Google CDN can handle this easily
MIN_DELAY_BETWEEN_REQUESTS = 1.0 / MAX_REQUESTS_PER_SECOND

# User agent to identify ourselves
USER_AGENT = 'FaB-Card-Trainer/1.0 (Educational ML Project; Contact: mmurray1629@gmail.com)'
# =========================================================

# Thread-safe rate limiter
class RateLimiter:
    """Simple rate limiter to ensure we don't overwhelm the server."""
    def __init__(self, min_delay):
        self.min_delay = min_delay
        self.last_request_time = 0
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Wait if necessary to maintain rate limit."""
        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_delay:
                sleep_time = self.min_delay - time_since_last
                time.sleep(sleep_time)
            self.last_request_time = time.time()

class DownloadStats:
    """Track download statistics."""
    def __init__(self):
        self.total_printings = 0
        self.unique_urls = 0
        self.downloaded = 0
        self.skipped_existing = 0
        self.skipped_duplicate_url = 0
        self.failed = 0
        self.start_time = time.time()
    
    def print_summary(self):
        elapsed = time.time() - self.start_time
        print("\n" + "="*60)
        print("DOWNLOAD SUMMARY")
        print("="*60)
        print(f"Total printings in database: {self.total_printings}")
        print(f"Unique image URLs: {self.unique_urls}")
        print(f"Downloaded: {self.downloaded}")
        print(f"Skipped (already exist): {self.skipped_existing}")
        print(f"Skipped (duplicate URL): {self.skipped_duplicate_url}")
        print(f"Failed: {self.failed}")
        print(f"Time elapsed: {elapsed:.1f}s")
        if self.downloaded > 0:
            print(f"Average: {elapsed/self.downloaded:.2f}s per download")
        print("="*60)

def sanitize_filename(name):
    """Clean card name for use in filename."""
    name = name.replace(' ', '_')
    name = re.sub(r'[^A-Za-z0-9_-]', '', name)
    return name

def downscale_image(image_bytes, max_dimension):
    """
    Downscale image to max_dimension while maintaining aspect ratio.
    Preserves RGBA mode (including transparency/alpha channel).
    
    Args:
        image_bytes: Raw image bytes
        max_dimension: Maximum width or height
    
    Returns:
        PIL Image object (downscaled, RGBA mode)
    """
    img = Image.open(BytesIO(image_bytes))
    
    # Ensure RGBA mode (preserve alpha channel if present, add if not)
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Calculate new dimensions maintaining aspect ratio
    width, height = img.size
    if width > height:
        if width > max_dimension:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))
        else:
            return img  # Already small enough
    else:
        if height > max_dimension:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))
        else:
            return img  # Already small enough
    
    # Downscale with high-quality Lanczos filter
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return img_resized

def download_and_process_image(url, output_path, max_dimension, rate_limiter):
    """
    Download image from URL, downscale, and save.
    Respects rate limiting to be polite to the server.
    
    Args:
        url: Image URL
        output_path: Where to save the processed image
        max_dimension: Max dimension for downscaling
        rate_limiter: RateLimiter instance to control request rate
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Wait if needed to respect rate limit
        rate_limiter.wait_if_needed()
        
        # Make request with custom user agent
        headers = {'User-Agent': USER_AGENT}
        response = requests.get(url, timeout=REQUEST_TIMEOUT, headers=headers)
        response.raise_for_status()
        
        # Downscale the image
        img = downscale_image(response.content, max_dimension)
        
        # Save as PNG
        img.save(output_path, 'PNG', optimize=True)
        return True
    
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def process_printing(card_name, printing, base_output_dir, url_to_file, stats, max_dimension, rate_limiter, card_pitch):
    """
    Process a single printing: download or skip if duplicate.
    
    Args:
        card_name: Name of the card
        printing: Printing dictionary from card.json
        base_output_dir: Base output directory path
        url_to_file: Dict mapping URLs to first downloaded file
        stats: DownloadStats object
        max_dimension: Max dimension for downscaling
        rate_limiter: RateLimiter instance
        card_pitch: Pitch value from card (for R/Y/B designation)
    
    Returns:
        (success: bool, message: str)
    """
    image_url = printing.get('image_url')
    if not image_url:
        return False, "No image URL"
    
    # Get set ID and create set-specific directory
    set_id = printing.get('set_id', 'UNKNOWN')
    set_dir = os.path.join(base_output_dir, set_id)
    os.makedirs(set_dir, exist_ok=True)
    
    # Extract filename from URL (e.g., "HVY001_MARVEL_BACK.png" -> "HVY001_MARVEL_BACK")
    url_filename = image_url.split('/')[-1]
    url_filename_base = os.path.splitext(url_filename)[0]  # Remove .png
    
    # Generate clean card name (remove special chars, replace spaces with _)
    card_filename = sanitize_filename(card_name)
    
    # Add pitch color if present (1=R, 2=Y, 3=B)
    pitch_suffix = ""
    if card_pitch:
        pitch_map = {"1": "R", "2": "Y", "3": "B"}
        pitch_suffix = f"_{pitch_map.get(card_pitch, '')}"
    
    # Filename format: CardName_[R/Y/B]_URLFilename.png
    # Examples: 
    #   Macho_Grande_R_EVR029.png
    #   Kayo_Armed_and_Dangerous_HVY001_MARVEL_BACK.png
    filename = f"{card_filename}{pitch_suffix}_{url_filename_base}.png"
    output_path = os.path.join(set_dir, filename)
    
    # Check if already exists
    if os.path.exists(output_path):
        stats.skipped_existing += 1
        return True, f"Already exists: {filename}"
    
    # Check if we've already downloaded this URL
    if image_url in url_to_file:
        # This is a duplicate URL - we could create a symlink here
        # For now, just skip it to save disk space
        stats.skipped_duplicate_url += 1
        source_file = url_to_file[image_url]
        source_set = os.path.basename(os.path.dirname(source_file))
        return True, f"Duplicate URL (same as {source_set}/{os.path.basename(source_file)}): {set_id}/{filename}"
    
    # Download and process
    success = download_and_process_image(image_url, output_path, max_dimension, rate_limiter)
    
    if success:
        stats.downloaded += 1
        url_to_file[image_url] = output_path
        return True, f"Downloaded: {set_id}/{filename}"
    else:
        stats.failed += 1
        return False, f"Failed: {set_id}/{filename}"

def collect_all_printings(cards):
    """
    Collect all printings from all cards.
    
    Returns:
        List of (card_name, printing, card_pitch) tuples and unique URL count
    """
    printings = []
    url_set = set()
    
    for card in cards:
        card_name = card.get('name', 'unknown')
        card_pitch = card.get('pitch', '')  # May be empty string for non-pitch cards
        for printing in card.get('printings', []):
            printings.append((card_name, printing, card_pitch))
            url = printing.get('image_url')
            if url:
                url_set.add(url)
    
    return printings, len(url_set)

def main():
    parser = argparse.ArgumentParser(
        description='Download all card printings in parallel with deduplication'
    )
    parser.add_argument('--max-workers', type=int, default=DEFAULT_MAX_WORKERS,
                       help=f'Number of concurrent download workers (default: {DEFAULT_MAX_WORKERS})')
    parser.add_argument('--max-size', type=int, default=MAX_DIMENSION,
                       help=f'Maximum image dimension in pixels (default: {MAX_DIMENSION})')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                       help=f'Output directory (default: {OUTPUT_DIR})')
    parser.add_argument('--rate-limit', type=float, default=MAX_REQUESTS_PER_SECOND,
                       help=f'Max requests per second (default: {MAX_REQUESTS_PER_SECOND})')
    args = parser.parse_args()
    
    print("="*60)
    print("PARALLEL CARD IMAGE DOWNLOADER")
    print("="*60)
    print(f"Configuration:")
    print(f"  - Max workers: {args.max_workers}")
    print(f"  - Rate limit: {args.rate_limit} requests/second")
    print(f"  - Max dimension: {args.max_size}px")
    print(f"  - Output directory: {args.output_dir} (organized by set)")
    print(f"  - Source: {CARD_JSON_PATH}")
    print(f"\n⚠️  Being polite to server with rate limiting!")
    print(f"📁 Images will be organized: {args.output_dir}/<SET_ID>/<card_files>")
    print()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load card database
    print("Loading card database...")
    with open(CARD_JSON_PATH, 'r', encoding='utf-8') as f:
        cards = json.load(f)
    
    print(f"Loaded {len(cards)} cards")
    
    # Collect all printings
    print("Collecting printings...")
    all_printings, unique_urls = collect_all_printings(cards)
    
    stats = DownloadStats()
    stats.total_printings = len(all_printings)
    stats.unique_urls = unique_urls
    
    print(f"Found {stats.total_printings} total printings")
    print(f"Found {stats.unique_urls} unique image URLs")
    print(f"Expected deduplication: {stats.total_printings - stats.unique_urls} duplicate URLs")
    print()
    
    # Track URL to file mapping (thread-safe dict)
    url_to_file = {}
    
    # Create rate limiter
    min_delay = 1.0 / args.rate_limit
    rate_limiter = RateLimiter(min_delay)
    
    # Download in parallel
    print(f"Starting parallel download with {args.max_workers} workers...")
    print("-"*60)
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks
        futures = {}
        for card_name, printing, card_pitch in all_printings:
            future = executor.submit(
                process_printing,
                card_name,
                printing,
                args.output_dir,
                url_to_file,
                stats,
                args.max_size,
                rate_limiter,
                card_pitch
            )
            futures[future] = (card_name, printing.get('id', 'unknown'))
        
        # Process results as they complete
        completed = 0
        for future in as_completed(futures):
            completed += 1
            card_name, printing_id = futures[future]
            
            try:
                success, message = future.result()
                
                # Print progress every 50 items or if download/fail
                if completed % 50 == 0 or (success and stats.downloaded > 0 and message.startswith("Downloaded")):
                    progress = (completed / stats.total_printings) * 100
                    print(f"[{completed}/{stats.total_printings}] ({progress:.1f}%) - {message}")
                elif not success:
                    print(f"[{completed}/{stats.total_printings}] {message}")
            
            except Exception as e:
                stats.failed += 1
                print(f"Error processing {card_name} ({printing_id}): {e}")
    
    # Print summary
    stats.print_summary()
    
    # Save manifest of URL mappings for reference
    manifest_path = os.path.join(args.output_dir, 'url_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(url_to_file, f, indent=2)
    
    print(f"\nURL manifest saved to: {manifest_path}")
    print("This maps image URLs to downloaded files for reference.")

if __name__ == '__main__':
    main()
