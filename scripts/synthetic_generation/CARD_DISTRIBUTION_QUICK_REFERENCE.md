# Card Distribution Testing - Quick Reference

## Quick Start

```powershell
# Navigate to workspace root
cd "c:\Users\Michael Murrary\FaBCode"

# Run quick test (1,000 iterations, ~8 seconds)
python scripts\synthetic_generation\run_quick_test.py quick

# Run standard test (25,000 iterations, ~3.5 minutes)
python scripts\synthetic_generation\run_quick_test.py standard

# Compare both selectors (25k each, ~7 minutes)
python scripts\synthetic_generation\run_quick_test.py compare
```

## Pre-configured Scenarios

| Scenario | Iterations | Selector | Time | Description |
|----------|-----------|----------|------|-------------|
| `quick` | 1,000 | weighted | ~8s | Quick validation |
| `standard` | 25,000 | weighted | ~3.5m | Recommended for testing |
| `large` | 50,000 | weighted | ~7m | Deep analysis |
| `quick-smooth` | 1,000 | smooth | ~8s | Quick smooth test |
| `standard-smooth` | 25,000 | smooth | ~3.5m | Full smooth test |
| `compare` | 25,000 | both | ~7m | Compare selectors |
| `compare-quick` | 5,000 | both | ~1.5m | Quick comparison |

## Direct Usage (Advanced)

```powershell
# Custom iterations
python scripts\synthetic_generation\test_card_distribution.py --iterations 10000 --selector weighted

# Silent mode with custom output
python scripts\synthetic_generation\test_card_distribution.py -n 25000 -s weighted -o my_test.json -q

# Test smooth selector
python scripts\synthetic_generation\test_card_distribution.py -n 25000 -s smooth

# Test both selectors
python scripts\synthetic_generation\test_card_distribution.py -n 25000 -s both
```

## What Gets Tested

✅ **Card Selection Logic**
- Format selection (CC/LL/Blitz with 80/15/5 distribution)
- Hero selection based on format
- Weighted vs non-weighted card pools
- Zone-specific filtering (weapons, equipment, pitch, etc.)
- 2H weapon logic
- Combat chain cards
- Pitch zone weighting (80% blue, 15% yellow, 5% red)

❌ **What's Skipped (for speed)**
- Image loading and compositing
- Augmentations (blur, glare, rotation)
- File I/O (no images or labels)
- Positioning calculations

## Key Output Metrics

### Console
- Total simulations completed
- Average cards per playmat (~26-27)
- Unique cards seen (expect ~2,450 out of 2,641)
- Format distribution (should be ~80/15/5)
- Top 20 most selected cards
- Top 20 most selected heroes

### JSON File
- Complete card counts for all cards
- Complete hero counts
- Zone-specific breakdowns
- Detailed statistics

**Location**: `card_distribution_{selector}_{iterations}_{timestamp}.json`

### Formatted Card List (NEW!)
- All cards listed in single column
- Ordered by placement count (highest to lowest)
- Alphabetically sorted on ties
- Easy to read and compare

**Location**: `card_distribution_{selector}_{iterations}_{timestamp}_cards.txt`

**Sample format:**
```
Fyendal's Spring Tunic                                       #    407 ( 1.539%)
Nullrune Gloves                                              #    278 ( 1.051%)
Crown of Providence                                          #    254 ( 0.960%)
```

## Expected Results (25k iterations)

| Metric | Expected Value |
|--------|---------------|
| Unique cards seen | 2,400-2,500 (out of 2,641) |
| Unique heroes seen | 120-122 (all heroes) |
| Average cards/playmat | 25-30 |
| Format CC % | 80-85% |
| Format Blitz % | 15-20% |
| Format LL % | 0-5% |
| Top card | Fyendal's Spring Tunic (~1.4%) |
| Processing rate | 100-150 playmats/sec |

## Common Issues

### Error: File not found
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/card.json'
```
**Solution**: Run from workspace root, not from scripts directory

### Slow performance
**Normal**: 100-150 playmats/second  
**If slower**: Check system resources, use `--quiet` flag

### Unexpected distributions
**Check**:
- Card popularity weights file is up to date
- Card.json is complete
- No errors in card_selector.py logic

## Recommended Workflow

### Before Large Generation
```powershell
# 1. Test card selection
python scripts\synthetic_generation\run_quick_test.py standard

# 2. Review statistics
# Check console output and JSON file

# 3. If satisfied, run production generation
python scripts\synthetic_generation\parallel_generate_dataset.py
```

### Debugging Card Selection
```powershell
# 1. Run baseline test
python scripts\synthetic_generation\run_quick_test.py standard --output before.json

# 2. Make changes to card_selector.py

# 3. Run test again
python scripts\synthetic_generation\run_quick_test.py standard --output after.json

# 4. Compare before.json and after.json
```

### Comparing Selectors
```powershell
# Test both weighted and smooth selectors
python scripts\synthetic_generation\run_quick_test.py compare

# Review both outputs
# Files: card_distribution_weighted_25000_*.json
#        card_distribution_smooth_25000_*.json
```

## Performance Tips

- Use `--quiet` flag to reduce console I/O overhead
- Run from SSD for faster data loading
- Close unnecessary applications
- Expect ~100-150 playmats/sec on typical hardware

## Documentation

- **Full documentation**: `CARD_DISTRIBUTION_TESTING.md`
- **Implementation summary**: `CARD_DISTRIBUTION_TESTING_SUMMARY.md`
- **This guide**: `CARD_DISTRIBUTION_QUICK_REFERENCE.md`

## Examples

```powershell
# Example 1: Quick validation before commit
python scripts\synthetic_generation\run_quick_test.py quick

# Example 2: Full test for code review
python scripts\synthetic_generation\run_quick_test.py standard --quiet

# Example 3: Deep analysis with custom output
python scripts\synthetic_generation\test_card_distribution.py -n 50000 -s weighted -o deep_analysis.json

# Example 4: Compare weighted vs smooth
python scripts\synthetic_generation\run_quick_test.py compare
```

## When to Use

✅ **Use this tool when:**
- Testing card selector changes
- Validating weighted selection logic
- Checking distribution before large runs
- Debugging card selection issues
- Comparing selector strategies

❌ **Don't use for:**
- Testing image generation quality
- Testing augmentation effects
- Testing label file accuracy
- Testing positioning logic

For those, use `test_generation.py` or `Core_Playmat_Generator.py` instead.
