# Asset Management Scripts

Scripts for downloading, processing, and managing FaB card images and related assets.

## Directory Structure

```
asset_management/
├── README.md                            # This file
├── download_card_json.py               # Download card database JSON
├── explore_card_json.py                # Explore card.json structure
├── download_all_sets.py                # Download images for all sets
├── download_all_printings_parallel.py  # Download ALL printings with deduplication
├── download_single_card.py             # Download all printings of a specific card
├── extract_occluders_from_assets.py    # Extract dice/counters from asset sheets
├── resize_occluders.py                 # Resize occluder images to standard size
├── check_image_corners.py              # Inspect corner artifacts in images
├── fix_card_corners.py                 # Remove corner artifacts from card images
└── card_popularity_v2/                 # Multi-format popularity system (V2)
    ├── README.md                        # V2 system overview
    ├── TECHNICAL_SPEC.md               # Detailed architecture and implementation plan
    ├── extract_heroes.py               # Extract heroes from card.json to heroes_card.json
    ├── scrape_popularity.py            # Main scraper with CC/Blitz support
    ├── test_scraper.py                 # Test wrapper for individual hero testing
    ├── check_blitz.py                  # Utility to verify blitz legality
    ├── test_url_gen.py                 # URL generation validation
    └── debug_hero_page.py              # Debug script to inspect page data structure
```

## Scripts Overview

### Data Download

**download_card_json.py**
- Downloads the latest card database from the-fab-cube GitHub repository
- Saves to `data/card.json`
- This is the master database used by all other scripts

**explore_card_json.py**
- Displays structure and statistics of card.json
- Shows available fields, sample entries, cards per set
- Useful for understanding the data structure

### Image Download

**download_all_sets.py**
- Downloads ALL card printings for all sets found in card.json
- Usage: `python download_all_sets.py`
- Calls download_all_printings_parallel.py automatically

**download_all_printings_parallel.py**
- Downloads ALL printings of every card (not just one per set)
- Features: URL-based deduplication, image downscaling (250px), rate limiting
- Usage: `python download_all_printings_parallel.py --max-workers 15 --rate-limit 15`
- This is the primary download script - use this for all card downloads

**download_single_card.py**
- Downloads all printings of a specific card by name
- Usage: `python download_single_card.py "Enlightened Strike" --force`
- Useful for testing or getting specific cards

### Card Popularity Data

**card_popularity_v2/ (Active System)**
- Multi-format scraper supporting Classic Constructed and Blitz
- Auto-discovery of heroes from heroes_card.json (no hardcoded lists)
- Robust error handling with failed hero tracking
- Output: `data/card_weights_all_printings.json`
- See `card_popularity_v2/README.md` for full documentation

**Main Scripts:**
- `extract_heroes.py` - Separates heroes from card.json into heroes_card.json
- `scrape_popularity.py` - Main scraper (CC/Blitz formats)
- `test_scraper.py` - Test individual heroes before full scrape

**Usage:**
```bash
# Full scrape (both formats)
python scrape_popularity.py --formats cc blitz

# Single format
python scrape_popularity.py --formats cc

# Test individual hero
python test_scraper.py "Dorinthea Ironsong" cc
```

### Occluder Processing

Occluders are small objects (dice, counters) placed on cards during synthetic generation to add realism.

**extract_occluders_from_assets.py**
- Extracts individual dice/counters from asset sheet images
- Uses background removal and contour detection
- Segments multi-object images into individual occluder files

**resize_occluders.py**
- Normalizes occluder image sizes to 10-40px range
- Maintains aspect ratio and transparency
- Must be run before using occluders in synthetic generation
- Usage: Automatically processes `data/occluders/` directory

### Image Quality Tools

**check_image_corners.py**
- Diagnostic tool to inspect corner artifacts in card images
- Shows alpha channel statistics and corner pixel values
- Usage: `python check_image_corners.py <image_path>`
- Helps identify cards that need corner fixing

**fix_card_corners.py**
- Removes white/black corner artifacts from card images
- Creates rounded corner masks based on card dimensions
- Can process individual files or entire directories
- Usage: `python fix_card_corners.py --input <path> --recursive`

## Typical Workflow

### Initial Setup
```bash
# 1. Download card database
python asset_management/download_card_json.py

# 2. Explore the data (optional)
python asset_management/explore_card_json.py

# 3. Download ALL card printings
python asset_management/download_all_printings_parallel.py

# OR download for all sets using the orchestrator
python asset_management/download_all_sets.py
```

### Card Popularity Weights Pipeline
```bash
# 1. Scrape main hero popularity data
python asset_management/scrape_card_popularity.py

# 2. Add missing heroes by scraping individual pages
python asset_management/add_missing_heroes.py

# 3. Add placeholder data for remaining heroes
python asset_management/add_missing_heroes_to_weights.py
```

### Image Quality Fixes
```bash
# Check for corner artifacts
python asset_management/check_image_corners.py data/images/WTR/some_card.png

# Fix corner artifacts in entire directory
python asset_management/fix_card_corners.py --input data/images/WTR --recursive
```

### Occluder Extraction (if using custom dice/counter images)
```bash
# Extract occluders from asset sheet
python asset_management/extract_occluders_from_assets.py

# Resize extracted occluders
python asset_management/resize_occluders.py
```

## Dependencies

These scripts require:
- `requests` - HTTP requests for downloading
- `Pillow` (PIL) - Image processing
- `opencv-python` (cv2) - Advanced image operations
- `numpy` - Numerical operations
- `tqdm` - Progress bars (for parallel downloads)

## Notes

- All scripts assume they're run from the repository root or handle paths relative to their location
- Downloaded images go to `data/images/<SET_ID>/`
- Occluders go to `data/occluders/`
- Card database is stored at `data/card.json`
- Image fixes are applied in-place (overwrites original files)
