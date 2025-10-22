# Card Distribution Testing Tool

## Overview

This tool simulates the card selection logic from the Core Playmat Generator **without creating any images or files**. It's designed to quickly test and validate card selection distributions at scale (recommended: 25,000+ iterations).

## Purpose

- **Validate card selection logic** - Ensure the weighted/smooth selectors are working correctly
- **Analyze distributions** - See which cards appear most/least frequently
- **Test at scale** - Run 25,000+ simulations in seconds instead of hours
- **Compare selectors** - Test weighted vs smooth selectors side-by-side
- **No I/O overhead** - Pure computation, no image generation or file writing

## Features

- Simulates full playmat generation logic (zones, heroes, card selection)
- Supports both weighted and smooth card selectors
- Tracks card counts, hero frequencies, and format distributions
- Generates detailed statistics and JSON output
- Very fast: ~100-200 playmats/second on typical hardware

## Usage

### Basic Usage

```powershell
# Test weighted selector with 25,000 iterations
python scripts\synthetic_generation\test_card_distribution.py --iterations 25000 --selector weighted

# Test smooth selector with 25,000 iterations
python scripts\synthetic_generation\test_card_distribution.py --iterations 25000 --selector smooth

# Test both selectors with 10,000 iterations each
python scripts\synthetic_generation\test_card_distribution.py --iterations 10000 --selector both
```

### Command Line Arguments

- `--iterations`, `-n`: Number of playmats to simulate (default: 25000)
- `--selector`, `-s`: Which selector to test - `weighted`, `smooth`, or `both` (default: weighted)
- `--output`, `-o`: Custom output JSON filename (default: auto-generated with timestamp)
- `--quiet`, `-q`: Suppress progress messages (only show final statistics)

### Examples

```powershell
# Quick test with 1,000 iterations
python scripts\synthetic_generation\test_card_distribution.py -n 1000 -s weighted

# Silent run with 50,000 iterations, save to specific file
python scripts\synthetic_generation\test_card_distribution.py -n 50000 -s smooth -o my_results.json -q

# Compare both selectors with 25,000 each
python scripts\synthetic_generation\test_card_distribution.py -n 25000 -s both
```

## Output

### Console Output

The script prints detailed statistics including:

1. **Summary Statistics**
   - Total simulations run
   - Total cards selected
   - Average cards per playmat
   - Unique cards/heroes seen

2. **Format Distribution**
   - Classic Constructed (CC): ~80%
   - Living Legend (LL): ~15%
   - Blitz: ~5%

3. **Top 20 Most Selected Cards**
   - Card name, count, and percentage
   - Useful for validating popularity weights

4. **Top 20 Most Selected Heroes**
   - Hero name, count, and percentage

### JSON Output

A detailed JSON file is saved with:

- Complete card counts for all cards
- Complete hero counts for all heroes
- Zone-specific card counts (hero1, hero2, combat_chain)
- Format distribution
- Summary statistics

**Example filename:** `card_distribution_weighted_25000_20251022_094915.json`

### Formatted Card List (NEW!)

A formatted text file is also saved listing **all cards** in a clean, readable format:

- **Ordered by placement count** (highest to lowest)
- **Alphabetically sorted on ties**
- **Single column format** similar to data.yaml
- Includes count and percentage for each card

**Example filename:** `card_distribution_weighted_25000_20251022_094915_cards.txt`

**Format:**
```
Fyendal's Spring Tunic                                       #    407 ( 1.539%)
Nullrune Gloves                                              #    278 ( 1.051%)
Crown of Providence                                          #    254 ( 0.960%)
...
```

This makes it easy to:
- Quickly scan which cards appear most/least
- Verify all cards are being selected
- Compare distributions between runs
- Import into spreadsheets for analysis

## What Gets Simulated

The script simulates the **exact same logic** as Core_Playmat_Generator.py:

1. ✅ Format selection (CC/LL/Blitz with 80/15/5 weighting)
2. ✅ Hero selection (random hero based on format legality)
3. ✅ Card pool building (weighted, generic, class, talent pools)
4. ✅ Zone-based card selection (weapon, equipment, pitch, banish, etc.)
5. ✅ 2H weapon logic (blocks off-hand slot when 2H weapon equipped)
6. ✅ Combat chain cards (4-15 random cards split between heroes)
7. ✅ Pitch zone weighting (80% blue, 15% yellow, 5% red)

## What Gets Skipped

To maximize speed, the script **skips all I/O and rendering**:

- ❌ No image loading (backgrounds, card images)
- ❌ No image compositing (no PIL/OpenCV operations)
- ❌ No augmentations (blur, glare, rotation, etc.)
- ❌ No file writing (no images or label files created)
- ❌ No positioning cache lookups

## Performance

On typical hardware:
- **~100-200 playmats/second**
- **25,000 iterations in ~2-4 minutes**
- Uses minimal memory (only card database + results tracking)

Much faster than full generation:
- Core_Playmat_Generator.py: ~1-2 images/second (with I/O)
- test_card_distribution.py: ~100-200 simulations/second (no I/O)

**Speed improvement: ~50-100x faster**

## Validation Use Cases

### 1. Verify Weighted Selection Works

```powershell
python scripts\synthetic_generation\test_card_distribution.py -n 25000 -s weighted
```

Check that:
- Popular cards (Fyendal's Spring Tunic, Nullrune Gloves) appear frequently
- Rare cards still appear but less often
- Distribution matches expected popularity weights

### 2. Compare Weighted vs Smooth

```powershell
python scripts\synthetic_generation\test_card_distribution.py -n 25000 -s both
```

Compare the two JSON outputs to see:
- How distributions differ between selectors
- Which selector provides better variety
- Impact on card frequency

### 3. Test Format Distribution

```powershell
python scripts\synthetic_generation\test_card_distribution.py -n 10000 -s weighted
```

Verify format distribution is close to:
- CC: 80%
- LL: 15%
- Blitz: 5%

### 4. Find Edge Cases

Run large iterations (50k+) to find:
- Cards that never appear (possible bugs in filtering)
- Heroes that appear too often/rarely
- Unexpected distribution patterns

## Notes

- **Run from workspace root**: The script must be run from `c:\Users\Michael Murrary\FaBCode` (not from the scripts directory) because it needs access to `data/card.json` and `data/card_popularity_weights_by_hero.json`

- **Deterministic randomness**: Each run will produce different results due to randomness. Run multiple times or use large iteration counts for stable statistics.

- **Memory usage**: With 25,000 iterations, the script uses ~100-200MB of RAM (mostly for storing card names in results).

## Example Workflow

```powershell
# 1. Navigate to workspace root
cd "c:\Users\Michael Murrary\FaBCode"

# 2. Run quick test (1000 iterations)
python scripts\synthetic_generation\test_card_distribution.py -n 1000 -s weighted

# 3. Review console output for obvious issues

# 4. Run full test (25000 iterations)
python scripts\synthetic_generation\test_card_distribution.py -n 25000 -s weighted

# 5. Review JSON output for detailed analysis
# File: card_distribution_weighted_25000_YYYYMMDD_HHMMSS.json

# 6. Compare with smooth selector
python scripts\synthetic_generation\test_card_distribution.py -n 25000 -s smooth

# 7. Compare the two JSON files to analyze differences
```

## Troubleshooting

**Error: `FileNotFoundError: [Errno 2] No such file or directory: 'data/card.json'`**
- Solution: Run the script from the workspace root, not from the scripts directory

**Error: Simulations failing**
- Check that `data/card.json` and `data/card_popularity_weights_by_hero.json` exist and are valid
- Ensure the card selector scripts are present and working

**Slow performance**
- Normal: ~100-200 playmats/second
- If slower, check system resources (CPU, memory)
- Use `--quiet` flag to reduce console I/O overhead

## Future Enhancements

Potential additions:
- Export to CSV for easier analysis in Excel/Google Sheets
- Visualization of card distribution (histograms, charts)
- Comparison mode (diff two JSON outputs)
- Statistical tests (chi-square, KS test) for distribution validation
- Support for filtering by specific heroes or formats
