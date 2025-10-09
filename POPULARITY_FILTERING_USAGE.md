# Popularity-Based Card Filtering and Weighted Sampling

## Overview

The synthetic data generation script now supports filtering cards by popularity rank and using popularity weights for card selection. This allows you to:

1. **Filter by popularity rank ranges** - Generate data with only top N cards, or specific ranges (e.g., 301-600)
2. **Use weighted sampling** - Bias card selection towards more popular cards based on their weights

## Prerequisites

The script uses `data/card_popularity_weights.json` which should contain:
- Card popularity weights (normalized frequencies across hero decks)
- Metadata about when the weights were generated

## Command-Line Arguments

### Popularity Rank Filtering

- `--popularity-min RANK` - Minimum popularity rank (inclusive)
  - Example: `--popularity-min 1` starts from the most popular card
  - Example: `--popularity-min 301` starts from the 301st most popular card

- `--popularity-max RANK` - Maximum popularity rank (inclusive)
  - Example: `--popularity-max 300` includes up to the 300th most popular card
  - Example: `--popularity-max 600` includes up to the 600th most popular card

### Weighted Sampling

- `--use-popularity-weights` - Enable weighted card selection based on popularity
  - When enabled, more popular cards are selected more frequently
  - Cards without weights get a minimum weight to ensure some representation

### Custom Weights Path

- `--popularity-weights-path PATH` - Override the default weights file location
  - Default: `/workspace/fab/src/data/card_popularity_weights.json`

## Usage Examples

### Example 1: Generate data with only the top 300 most popular cards

```bash
python scripts/generate_synthetic_playmat_screenshots.py \
    --num-images 1000 \
    --popularity-min 1 \
    --popularity-max 300
```

### Example 2: Generate data with cards ranked 301-600

```bash
python scripts/generate_synthetic_playmat_screenshots.py \
    --num-images 1000 \
    --popularity-min 301 \
    --popularity-max 600
```

### Example 3: Generate data with cards ranked 601-900

```bash
python scripts/generate_synthetic_playmat_screenshots.py \
    --num-images 1000 \
    --popularity-min 601 \
    --popularity-max 900
```

### Example 4: Generate data with top 300 cards using weighted sampling

```bash
python scripts/generate_synthetic_playmat_screenshots.py \
    --num-images 1000 \
    --popularity-min 1 \
    --popularity-max 300 \
    --use-popularity-weights
```

This will:
- Only use the top 300 most popular cards
- Sample cards with probability proportional to their popularity weights
- More popular cards will appear more frequently in the synthetic data

### Example 5: Use weighted sampling without rank filtering

```bash
python scripts/generate_synthetic_playmat_screenshots.py \
    --num-images 1000 \
    --use-popularity-weights
```

This will use all available cards but bias selection towards more popular ones.

### Example 6: Combine with coverage-guided generation

```bash
python scripts/generate_synthetic_playmat_screenshots.py \
    --num-images 1000 \
    --popularity-min 1 \
    --popularity-max 300 \
    --use-popularity-weights \
    --coverage-guided \
    --makeup-min 5
```

This will:
- Only use top 300 cards
- Use weighted sampling within each selected class
- Ensure at least 5 instances of each card class
- Balance coverage across the dataset

## How It Works

### Rank Calculation

1. Cards are ranked by their `total_weight` from the JSON file (descending)
2. Rank 1 = most popular card
3. Cards with multiple printings have their weights summed and share the same rank

### Filtering Process

1. Script loads the popularity weights JSON file
2. Creates a rank mapping (card name → rank number)
3. Filters `card_files` to only include cards within the specified rank range
4. Cards not in the weights file are excluded when filtering is enabled

### Weighted Sampling

When `--use-popularity-weights` is enabled:
- Each card's selection probability is proportional to its popularity weight
- Cards with higher weights are selected more frequently
- Cards without weights get a minimum weight of 0.001
- Works with both random sampling and coverage-guided generation

### Canonicalization

Card names are canonicalized to handle reprints:
- `Enlightened_Strike_WTR159` → `Enlightened_Strike`
- `Snatch_SEA169` → `Snatch`

This ensures cards with multiple printings are treated as a single entity for ranking and weighting.

## Tips

1. **Start with top cards**: For initial training, use `--popularity-min 1 --popularity-max 300` to focus on the most commonly played cards

2. **Incremental training**: Train on progressively larger sets:
   - Phase 1: Top 300 cards
   - Phase 2: Cards 1-600 (or 301-600 for just the next tier)
   - Phase 3: Cards 1-900 (or 601-900 for just the third tier)

3. **Use weighted sampling**: Add `--use-popularity-weights` to reflect real-world card frequencies in gameplay

4. **Combine with coverage guidance**: Use `--coverage-guided` to ensure all cards in your range get represented, while still biasing towards popular cards

5. **Monitor filtered results**: The script prints how many cards were filtered, e.g.:
   ```
   [init] Filtered 1779 cards to 300 within rank range [1, 300]
   ```

## Troubleshooting

### No cards found in range

If you see:
```
RuntimeError: No cards found within popularity rank range [X, Y]
```

Possible causes:
- The rank range exceeds the number of cards with weights
- No card images exist for cards in that rank range
- The weights file path is incorrect

### Cards not in weights file

Cards without popularity data:
- Are excluded when rank filtering is active
- Get minimum weight (0.001) when weighted sampling is enabled
- Can still be used if no filtering/weighting is applied

## Files Modified

- `scripts/generate_synthetic_playmat_screenshots.py` - Main script with new features
- Added functions:
  - `load_popularity_weights()` - Load and parse weights JSON
  - `filter_cards_by_popularity_rank()` - Filter cards by rank range
  - `weighted_card_choice()` - Select cards using popularity weights
