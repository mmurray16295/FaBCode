# Data Organization Structure

## Overview
This document describes the organization of training data and generated images in the FaBCode project.

## Directory Structure

```
data/
├── images/              # Source card images (from Scryfall/FaB API) - GITIGNORED
├── positioning_cache/   # Cached card positioning data - GITIGNORED
├── card.json           # Card metadata
├── card_popularity_weights.json  # Card popularity rankings
├── generated/          # All generated training data - GITIGNORED
│   ├── phase1/
│   ├── phase2/
│   ├── phase3/
│   └── ...
└── phase*/             # Legacy phase directories - GITIGNORED
```

## Generated Data Guidelines

### Recommended: Use `data/generated/` directory
All future generated training data should be placed in `data/generated/` with appropriate subdirectories:

```bash
data/generated/
├── phase1_100classes/
│   ├── train/
│   ├── valid/
│   ├── test/
│   ├── classes.yaml
│   └── data.yaml
├── phase2_500classes/
│   ├── train/
│   ├── valid/
│   ├── test/
│   ├── classes.yaml
│   └── data.yaml
├── phase2_random/
│   ├── train/
│   ├── valid/
│   ├── test/
│   ├── classes.yaml
│   └── data.yaml
└── benchmarks/
    ├── synthetic_benchmark_v1/
    └── synthetic_benchmark_v2/
```

### Legacy: Direct `data/` subdirectories
Currently existing phase directories are in the root of `data/`:
- `data/phase2_500classes/` - Phase 2 synthetic template-based images (30k)
- `data/phase2_random/` - Phase 2 random playmat images (3.3k)
- `data/synthetic/` - Original synthetic data directory

**Note:** These are all gitignored via `data/phase*/` pattern.

## .gitignore Patterns

The following patterns ensure generated data is never committed:

```gitignore
# Ignore all generated training data
data/generated/
data/phase*/
data/synthetic/
data/synthetic*/
data/benchmark_synth*/
data/synthetic_benchmark/

# Ignore all card images and backgrounds
data/images/

# Ignore positioning cache
data/positioning_cache/
```

## Training Phase Naming Convention

Use descriptive names for training phases:

- `phase1_100classes` - Initial training with top 100 cards
- `phase2_500classes` - Expanded to top 500 cards
- `phase3_1000classes` - Further expansion to 1000 cards
- `phase4_all` - Full card set

Include class count or distinguishing features in the directory name.

## Scripts and Data Paths

### Generation Scripts
When creating generation scripts, use the `data/generated/` directory:

```python
OUTPUT_BASE_DIR = '/root/FaBCode/data/generated/phase3_1000classes'
```

Or add `--output-dir` argument:

```bash
python3 scripts/generate_synthetic_playmat_screenshots.py \
    --output-dir data/generated/phase3_1000classes \
    ...
```

### Training Scripts
Reference the `data.yaml` path from generated directories:

```bash
python3 scripts/train_yolo.py \
    --data data/generated/phase2_500classes/data.yaml \
    --weights runs/train/phase1_100classes/weights/best.pt \
    ...
```

## Best Practices

1. **Always use `data/generated/`** for new training data
2. **Include phase and class count** in directory names
3. **Maintain separate directories** for different data types:
   - `synthetic` - Template-based screenshots
   - `random` - Random playmat generation
   - `benchmark` - Testing/validation sets
4. **Document data provenance** - Note how data was generated in logs
5. **Clean up intermediate files** - Remove process directories after merging
6. **Verify gitignore** - Check that new directories are ignored before generating large datasets

## Migration Plan (Future)

When convenient, migrate existing phase directories:

```bash
mkdir -p data/generated
mv data/phase2_500classes data/generated/
mv data/phase2_random data/generated/
mv data/synthetic data/generated/phase1_100classes
```

Then update all `data.yaml` paths and training scripts accordingly.

## Disk Space Management

Generated data can be large:
- Phase 1 (10k images): ~5-7 GB
- Phase 2 (30k images): ~45-50 GB  
- Random data (3.3k images): ~1.5-2 GB

Regularly clean up:
- Old benchmark data
- Intermediate process directories (`phase2_p0`, `phase2_p1`, etc.)
- Failed training runs
- Unnecessary visualizations

Use `du -sh data/generated/*/` to monitor disk usage.
