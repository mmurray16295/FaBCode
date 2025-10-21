# Occluders Directory

This directory contains small images (10-40px) of dice, counters, and other objects used as occluders during synthetic data generation.

## Purpose

Occluders are randomly placed on cards during synthetic playmat generation to simulate real gameplay conditions where cards may be partially covered by:
- Dice (attack/defense counters)
- Resource counters
- Other game tokens/markers

This helps train the model to recognize cards even when partially obscured.

## Setup

### Option 1: Extract from Asset Sheets
If you have asset sheet images with multiple dice/counters:

```bash
# Place asset sheets in data/images/ZZAssets/
python scripts/asset_management/extract_occluders_from_assets.py

# Resize extracted occluders
python scripts/asset_management/resize_occluders.py
```

### Option 2: Manual Addition
1. Add dice/counter PNG images to this directory
2. Run resize script to normalize sizes:
   ```bash
   python scripts/asset_management/resize_occluders.py
   ```

## File Requirements

- Format: PNG with transparency (RGBA)
- Size: Will be automatically resized to 10-40px range
- Content: Dice, counters, tokens, or other small game objects
- **NOT card images** - Token cards are downloaded separately to `data/images/`

## Usage

The synthetic generation script automatically loads all PNG files from this directory and randomly places them on cards with:
- 10% probability per card
- 1% probability of second occluder on same card
- Random rotation, blur, and color shifts applied
- Centered placement with slight random offset

## Note

This directory is for **occluders only** (dice/counters), not FaB Token cards. Token cards (like "Runechant Token") are actual cards and should be in `data/images/<SET_ID>/` like all other cards.
