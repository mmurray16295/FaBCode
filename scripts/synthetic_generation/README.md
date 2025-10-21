# Synthetic Playmat Generation Module

This module contains all scripts related to generating synthetic playmat training images for the FaB Card Detector.

## Core Components

### Primary Generator
- **`Core_Playmat_Generator.py`** - Main synthetic playmat image generator
  - Generates realistic playmat screenshots with cards placed on zone-labeled backgrounds
  - Applies augmentations (blur, glare, perspective transforms, etc.)
  - Creates YOLO format labels for training
  - Uses CardSelector for intelligent card selection

### Card Selection System
- **`card_selector.py`** - Intelligent card selection with format/hero/talent awareness
  - Handles Classic Constructed, Living Legend, and Blitz formats
  - Hero-specific card filtering
  - Talent-based card selection
  - Popularity-weighted sampling

### Augmentation System
- **`augmentations.py`** - Image augmentation functions (blur, glare, perspective, shadows, etc.)
- **`augmentation_config.py`** - Augmentation configuration and probability settings
- **`augmentation_presets.py`** - Pre-defined augmentation presets for different training phases

## Batch Generation Scripts

### Testing & Verification
- **`test_generation.py`** - Consolidated test script with configurable options
  - Generate 1-N images with verification
  - Optional visualization (draw bounding boxes)
  - Comprehensive verification checks
  - Command-line configurable (count, augmentations, presets)

### Production Generation
- **`parallel_generate_dataset.py`** - Parallel generation using multiple processes for large datasets
  - Multi-process parallelization for high-core-count systems
  - Automatic background management (ensures 50% backgrounds of target images)
  - Smart background cycling for diversity
  - Efficient for generating 10k+ images

## Setup & Utilities

### Data Setup
- **`generate_card_data_yaml.py`** - Generate YOLO data.yaml and card class mappings
  - Creates `data/card_name_to_class_id.json` (2,641 card classes)
  - Creates `data/synthetic/data.yaml` for YOLO training

### Shared Utilities
- **`generation_utils.py`** - Shared utilities for background management
  - `ensure_background_variations()`: Auto-generate backgrounds if needed (50% rule)
  - `count_background_variations()`: Count existing backgrounds
  - Used by both test_generation.py and parallel_generate_dataset.py

### Background & Label Utilities
- **`generate_background_variations.py`** - Generate background variations with jostled zone labels
  - Creates large pool of backgrounds with randomized zone positions (±60px, ±10% scale)
  - Helps prevent model from overfitting to exact zone positions
  - Called automatically by generation_utils when needed
- **`propose_slots_simple.py`** - Use YOLO model to propose card slots on unlabeled backgrounds
  - Runs detection on unlabeled backgrounds
  - Generates YOLO label files for new backgrounds
  - Creates overlay previews for verification

### Visualization & Verification
- **`visualize_labels.py`** - Visualize YOLO labels with bounding boxes and card names
  - Draw boxes on existing generated images
  - Verify labels match actual card positions
  - Useful for quality checking datasets

## Testing Scripts

- **`test_card_selector_hero_selection.py`** - Test hero selection logic (Young/Adult matching)
- **`test_card_selector_talents.py`** - Test talent extraction and filtering
- **`empirical_card_test.py`** - Empirically test which cards can be selected by CardSelector

## Usage Examples

### Test generation with verification
```bash
# Single image with visualization (for inspection)
python scripts/synthetic_generation/test_generation.py --count 1 --visualize

# Quick 5-image test
python scripts/synthetic_generation/test_generation.py --count 5

# Full verification test (20 images with checks)
python scripts/synthetic_generation/test_generation.py --count 20 --verify

# Debug mode (no augmentations, with visualization)
python scripts/synthetic_generation/test_generation.py --count 3 --no-augmentations --visualize
```

### Production generation
```bash
# Small to medium batches (1-1000 images) - test_generation.py
python scripts/synthetic_generation/test_generation.py --count 100 --ensure-backgrounds

# Large datasets with parallel processing (10k+ images)
python scripts/synthetic_generation/parallel_generate_dataset.py --num-images 25000 --workers 8

# Pre-generate background variations (optional, auto-runs if needed)
python scripts/synthetic_generation/generate_background_variations.py --num-variations 10000
```

### Setup (run once before generation)
```bash
# Generate card class mappings and data.yaml
python scripts/synthetic_generation/generate_card_data_yaml.py
```

## Dependencies

- PIL/Pillow - Image manipulation
- OpenCV (cv2) - Advanced image processing
- NumPy - Numerical operations
- PyYAML - YAML file handling

## Data Requirements

The generator requires:
- `data/card.json` - Card database
- `data/card_weights_all_printings.json` - Card popularity weights (V2)
- `data/synthetic/backgrounds/` - Background playmat images
- `data/synthetic/backgrounds_labels/` - Zone label files (YOLO format)
- `data/images/` - Card images organized by set

## Output

Generated images are saved to:
- `data/synthetic/train/images/` - Training images
- `data/synthetic/train/labels/` - YOLO labels
- `data/synthetic/valid/images/` - Validation images (if split)
- `data/synthetic/valid/labels/` - Validation labels (if split)
