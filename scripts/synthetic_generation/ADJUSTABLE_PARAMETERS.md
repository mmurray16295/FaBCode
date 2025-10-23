# Comprehensive Adjustable Parameters Guide
## FaB Synthetic Playmat Generation System

**Last Updated:** October 23, 2025  
**Purpose:** Complete reference for tweaking synthetic dataset generation for validation/training sets

---

## Table of Contents
1. [Card Selection Parameters](#1-card-selection-parameters)
2. [Augmentation Configuration](#2-augmentation-configuration)
3. [Image Generation Parameters](#3-image-generation-parameters)
4. [Background Management](#4-background-management)
5. [Windowed Mode Parameters](#5-windowed-mode-parameters)
6. [Parallel Generation Settings](#6-parallel-generation-settings)
7. [Quick Preset Configurations](#7-quick-preset-configurations)

---

## 1. Card Selection Parameters

### 1.1 Card Selector Type
**File:** `Core_Playmat_Generator.py`, `parallel_generate_dataset.py`  
**Parameter:** `--selector` or `selector_type`

- **`'smooth'`** (default): Even distribution across all cards
  - Tracks usage counts globally
  - Selects randomly from least-used legal cards
  - "Draw without replacement" effect
  - Best for: Validation sets, ensuring even coverage
  
- **`'weighted'`**: Popularity-based selection
  - Uses deck usage statistics from `card_weights_all_printings.json`
  - 75% from weighted list, 25% from generic/class/talent pools
  - Formula: `weight = usage_percentage^0.75`
  - Best for: Training sets mimicking real play patterns

**Location:** `card_selector.py` (weighted), `card_selector_smooth.py` (smooth)

### 1.2 Format Selection
**File:** `card_selector.py`, `card_selector_smooth.py`  
**Function:** `select_format()`

```python
WEIGHTED_SELECTION_PROBABILITY = 0.75  # 75% weighted, 25% unweighted

# Format distribution
CC_PROBABILITY = 0.70      # 70% Classic Constructed
BLITZ_PROBABILITY = 0.30   # 30% Blitz
```

### 1.3 Card Pool Distribution (Weighted Selector Only)
**File:** `card_selector.py`  
**Function:** `select_card()`

```python
# Selection probabilities when NOT using weighted cards
# 75% weighted, then remaining 25% split as:
WEIGHTED_PROB = 0.75      # 75% from popularity weights
GENERIC_PROB = 0.08       # 8% generic cards (75-83%)
CLASS_ONLY_PROB = 0.08    # 8% class-only cards (83-91%)
TALENT_ONLY_PROB = 0.04   # 4% talent-only cards (91-95%)
BOTH_PROB = 0.05          # 5% class+talent cards (95-100%)
```

### 1.4 Usage Weight Formula
**File:** `card_selector.py`  
**Function:** `_convert_usage_to_weight()`

```python
# Converts deck usage percentage to selection weight
# Power of 0.75 balances popularity vs diversity
weight = usage_percentage^0.75

# Examples:
# 90% usage -> weight ~52.4
# 50% usage -> weight ~18.8
# 10% usage -> weight ~3.2
#  1% usage -> weight ~1.0
```

**Adjustment:** Change exponent (0.5 = more even, 1.0 = more extreme)

### 1.5 Pitch Card Weighting
**File:** `card_selector.py`  
**Function:** `select_card_for_zone()`

```python
# For Pitch/Pitch 2 zones only
pitch_weighting = True  # Enable pitch-based selection

BLUE_PROB = 0.80    # 80% blue pitch (value 3)
YELLOW_PROB = 0.15  # 15% yellow pitch (value 2)
RED_PROB = 0.05     # 5% red pitch (value 1)
```

### 1.6 Hero Selection
**File:** `card_selector.py`, `card_selector_smooth.py`

```python
# Weighted/unweighted split for heroes
WEIGHTED_SELECTION_PROBABILITY = 0.75  # 75% from weights, 25% from card.json

# Format preferences:
# - Blitz: Prefers Young heroes
# - CC: Prefers Adult (non-Young) heroes
```

---

## 2. Augmentation Configuration

### 2.1 Blur Settings
**File:** `augmentation_config.py`  
**Class:** `BlurConfig`

```python
enabled: bool = True
probability_range: Tuple[float, float] = (0.15, 0.50)  # 15-50% blur per image
kernel_size_range: Tuple[int, int] = (3, 9)  # Must be odd numbers
sigma_range: Tuple[float, float] = (0.5, 2.0)
```

**Effect:** Simulates motion blur or out-of-focus camera

### 2.2 Glare/Lighting Settings
**File:** `augmentation_config.py`  
**Class:** `GlareConfig`

```python
enabled: bool = True
probability: float = 0.25  # 25% of images get glare
intensity_range: Tuple[float, float] = (0.3, 0.7)
radius_range: Tuple[int, int] = (100, 300)  # Pixels
num_spots_range: Tuple[int, int] = (1, 3)
```

**Effect:** Simulates light reflections from overhead lighting

### 2.3 Shadow Settings
**File:** `augmentation_config.py`  
**Class:** `ShadowConfig`

```python
enabled: bool = True
probability: float = 0.50  # 50% of images get shadows
intensity_range: Tuple[float, float] = (0.0, 0.80)  # 0=none, 1=black
radius_range: Tuple[int, int] = (120, 400)  # Pixels
num_spots_range: Tuple[int, int] = (1, 2)
shadow_types: Tuple[str, ...] = ('spot', 'gradient', 'vignette', 'uniform')
```

**Shadow Types:**
- `'spot'`: Localized dark spots (simulates object shadows)
- `'gradient'`: Directional shadows from corners
- `'vignette'`: Edge darkening effect
- `'uniform'`: Overall dimming

### 2.4 Color Adjustment Settings
**File:** `augmentation_config.py`  
**Class:** `ColorAdjustmentConfig`

```python
enabled: bool = True
probability: float = 0.5  # 50% of images

# Ranges (multiply factors)
brightness_range: Tuple[float, float] = (0.935, 1.065)  # Reduced by 35% from original
contrast_range: Tuple[float, float] = (0.935, 1.065)
saturation_range: Tuple[float, float] = (0.9025, 1.0975)
hue_shift_range: Tuple[int, int] = (-3, 3)  # Degrees

# Tinting
tint_probability: float = 0.7  # 70% of color-adjusted images
tint_intensity_range: Tuple[float, float] = (0.0325, 0.0975)  # Reduced by 35%
tint_colors: dict = {
    'red': (0, 0, 255),      # BGR format
    'yellow': (0, 255, 255),
    'blue': (255, 0, 0),
    'green': (0, 255, 0),
    'orange': (0, 165, 255),
    'brown': (42, 82, 165),
    'black': (0, 0, 0),
    'purple': (255, 0, 255),
    'pink': (203, 192, 255)
}
```

**Effect:** Simulates different camera settings, white balance, lighting colors

### 2.5 Card Sleeve Settings
**File:** `augmentation_config.py`  
**Class:** `SleeveConfig`

```python
enabled: bool = True
probability: float = 0.6  # 60% of cards have sleeves
glare_probability: float = 0.4  # 40% of sleeved cards get glare
reflection_intensity: Tuple[float, float] = (0.1, 0.3)
color_tint_probability: float = 0.3  # 30% colored sleeves
```

**Effect:** Simulates plastic card sleeves with reflections

### 2.6 Deck Box Occlusion Settings
**File:** `augmentation_config.py`  
**Class:** `DeckBoxConfig`

```python
enabled: bool = True
probability: float = 0.15  # 15% of images
max_boxes: int = 2
size_range: Tuple[int, int] = (100, 250)  # Pixels
opacity_range: Tuple[float, float] = (0.7, 1.0)
```

**Effect:** Simulates deck boxes/cases partially occluding playmat

### 2.7 General Occluder Settings
**File:** `augmentation_config.py`  
**Class:** `OccluderConfig`

```python
enabled: bool = True
probability: float = 0.25  # 25% of images
max_occluders: int = 3
size_range: Tuple[int, int] = (50, 150)  # Pixels
opacity_range: Tuple[float, float] = (0.6, 0.95)
```

**Effect:** Simulates hands, dice, tokens, counters

### 2.8 Card Position Jitter
**File:** `augmentation_config.py`  
**Class:** `JitterConfig`

```python
enabled: bool = True
probability: float = 0.8  # 80% of cards
translation_range: Tuple[int, int] = (-15, 15)  # Pixels in x/y
rotation_range: Tuple[float, float] = (-8.0, 8.0)  # Degrees
scale_range: Tuple[float, float] = (0.95, 1.05)  # Scale factor
```

**Effect:** Randomizes card placement within zones

### 2.9 Bounding Box Jitter
**File:** `augmentation_config.py`  
**Class:** `BoundingBoxJitterConfig`

```python
enabled: bool = True
probability: float = 0.5  # 50% of boxes
expansion_range: Tuple[float, float] = (-0.05, 0.1)  # % of box size
shift_range: Tuple[int, int] = (-5, 5)  # Pixels in x/y
```

**Effect:** Creates imperfect bounding boxes for robust training

### 2.10 Preset Configurations
**File:** `augmentation_config.py`

```python
# Available preset functions:
DEFAULT_CONFIG = AugmentationConfig()  # Standard settings
create_light_config()    # Lighter augmentations (testing/debugging)
create_heavy_config()    # Heavier augmentations (robust training)
create_no_augmentation_config()  # All disabled (baseline)
```

---

## 3. Image Generation Parameters

### 3.1 Core Playmat Generation
**File:** `Core_Playmat_Generator.py`

```python
# Overlap control
max_overlap_pct: float = 25  # Max 25% overlap between cards
max_attempts: int = 50  # Placement attempts before giving up

# Background cycling
use_cycling: bool = False  # If True, cycles through backgrounds equally
                          # If False, random selection (may reuse)

# Zone jitter (DEPRECATED - use background variations instead)
apply_jitter: bool = False  # Position jitter within zones
jitter_range: int = 25  # ±25 pixels (only if apply_jitter=True)
```

### 3.2 Hard Case Effect
**File:** `Core_Playmat_Generator.py`  
**Function:** `apply_hard_case()`

```python
include_artifacts: bool = True  # Challenging visual artifacts

# Hard case sizing
border_size: int = random.randint(8, 12)  # Pixels on each side

# Border appearance
border_base_color: int = random.randint(200, 230)  # Light grey to white
```

**Effect:** Simulates thick clear plastic hard cases (like top-loaders)

### 3.3 Dice/Counter Occluders
**File:** `Core_Playmat_Generator.py`  
**Function:** `apply_occluders_to_playmat()`

```python
probability_per_card: float = 0.10  # 10% of cards get occluders
second_occluder_probability: float = 0.01  # 1% get two occluders

# Occluder augmentation
blur_intensity_range: Tuple[float, float] = (0.10, 0.30)
hue_shift_range: Tuple[float, float] = (-10, 10)  # Degrees
saturation_adjust_range: Tuple[float, float] = (0.8, 1.2)
brightness_adjust_range: Tuple[float, float] = (0.9, 1.1)

# Placement bias
center_bias: float = 0.3  # Lower = more centered
max_offset_x: float = 0.3  # 30% of card width
max_offset_y: float = 0.3  # 30% of card height
```

### 3.4 Color Strip Degradation
**File:** `augmentations.py`  
**Function:** `apply_color_strip_degradation()`

```python
strip_height_ratio: float = 0.08  # 8% of card height
probability: float = 0.85  # 85% of cards

# Desaturation
saturation_reduction_range: Tuple[float, float] = (0.2, 0.5)  # 20-50%

# Color shift
hue_shift_range: Tuple[int, int] = (-30, 30)  # Degrees

# Brightness variation
brightness_adjust_range: Tuple[float, float] = (0.7, 1.3)

# Noise addition
noise_intensity_range: Tuple[float, float] = (0.02, 0.08)

# Pitch symbol degradation (circular mask)
pitch_symbol_ratio: float = 0.5  # 50% of strip height
```

**Effect:** Makes pitch color strips unreliable for classification

### 3.5 Light Source Positioning
**File:** `Core_Playmat_Generator.py`

```python
# Light source for glare intensity calculation
# Position determines card-by-card glare intensity

# Positioning options (chosen randomly):
# - Top: (random_x, 0)
# - Bottom: (random_x, img_height)
# - Left: (0, random_y)
# - Right: (img_width, random_y)
# - Center: (img_width/2, img_height/2)

# Glare intensity formula
# Uses inverse exponential falloff from light source
falloff_exponent: float = -5  # Steeper = faster dropoff
glare_intensity_min: float = 0.05
glare_intensity_max: float = 0.35
```

### 3.6 Shadow Source Positioning
**File:** `Core_Playmat_Generator.py`

```python
# Shadow source determines shadow type and direction
# Positioned in one of 4 quadrants:
# - top-left, top-right, bottom-left, bottom-right

# Shadow intensity varies per card based on configuration
# Types: 'spot', 'gradient', 'vignette', 'uniform'
```

---

## 4. Background Management

### 4.1 Background Variations
**File:** `generate_background_variations.py`

```python
# Background jitter (replaces card-level jitter)
jitter_range: int = 25  # ±25 pixels for zone positions

# Variation generation
num_variations: int = 1000  # Generate 1000+ backgrounds
base_backgrounds: int = 28  # Source templates

# Ratio with images
target_ratio: float = 1.0  # 1:1 backgrounds to images
```

**Effect:** Pre-generates varied playmat templates with jittered zones

### 4.2 Background Cycling
**File:** `generation_utils.py`, `Core_Playmat_Generator.py`

```python
use_cycling: bool = True  # Recommended for parallel generation

# When cycling:
# - Each background used ~equally
# - Shuffled once, then cycled through
# - Resets when exhausted

# When random:
# - May reuse same backgrounds frequently
# - True random selection each time
```

---

## 5. Windowed Mode Parameters

### 5.1 Windowed Mode Settings
**File:** `windowed_mode.py`  
**Function:** `should_apply_windowed_mode()`, `apply_windowed_mode()`

```python
# Enable/disable
WINDOWED_MODE_ENABLED: bool = True
WINDOWED_MODE_PROBABILITY: float = 0.05  # 5% of images

# Scaling
scale_range: Tuple[float, float] = (0.45, 0.75)  # 45-75% of original size

# Background noise
white_bias: float = 0.65  # 65% white rectangles (simulates UI)
num_rectangles_per_10k_pixels: int = 1  # Density

# Rectangle sizing
rect_width_range: Tuple[int, int] = (10, 150)
rect_height_range: Tuple[int, int] = (10, 100)

# Color distribution
white_prob: float = 0.65      # 65% white-ish
red_prob: float = 0.04        # 4% red (20% of colored)
blue_prob: float = 0.025      # 2.5% blue (12.5% of colored)
gray_prob: float = 0.025      # 2.5% gray (12.5% of colored)
black_prob: float = 0.025     # 2.5% black (12.5% of colored)
random_prob: float = 0.235    # 23.5% random (remaining)
```

**Effect:** Simulates game running in browser/windowed mode with desktop visible

---

## 6. Parallel Generation Settings

### 6.1 Parallelization
**File:** `parallel_generate_dataset.py`

```python
num_processes: int = 80  # Parallel worker processes
# Recommendation: 60-80% of CPU cores for best performance
# Example: 128 cores -> 80-100 processes

batch_size: int = 1  # Images per batch (keep at 1 for max parallelism)
```

### 6.2 Performance Benchmarking
**File:** `parallel_generate_dataset.py`  
**Function:** `benchmark_parallelization()`

```python
test_counts = [10, 20, 30, 40, 50, 60, 80, 100, 120]
test_images_per_count = 20  # Images per test
```

### 6.3 Background Pre-Generation
**File:** `generation_utils.py`

```python
batch_size: int = 1000  # Generate backgrounds in batches
# Prevents OOM during large generation runs

ensure_1_to_1_ratio: bool = True  # One background per target image
```

---

## 7. Quick Preset Configurations

### 7.1 Easy Validation Set (Recommended Starting Point)
```python
# Use smooth selector for even distribution
selector_type = 'smooth'

# Light augmentations
augmentation_config = create_light_config()
# Overrides:
augmentation_config.blur.probability = 0.15        # Reduce blur
augmentation_config.glare.probability = 0.15       # Reduce glare
augmentation_config.shadow.probability = 0.20      # Light shadows
augmentation_config.color.probability = 0.30       # Moderate color
augmentation_config.sleeves.probability = 0.50     # Half with sleeves
augmentation_config.deck_boxes.probability = 0.05  # Minimal occlusion
augmentation_config.occluders.probability = 0.10   # Few occluders

# Windowed mode
WINDOWED_MODE_PROBABILITY = 0.02  # Only 2% windowed

# Color strip degradation
strip_degradation_probability = 0.70  # Most readable
```

### 7.2 Realistic Validation Set (Recommended for Final Testing)
```python
# Use smooth selector
selector_type = 'smooth'

# Standard augmentations (default config)
augmentation_config = DEFAULT_CONFIG

# Moderate windowed mode
WINDOWED_MODE_PROBABILITY = 0.05  # 5% windowed

# Standard color strip degradation
strip_degradation_probability = 0.85  # 85%
```

### 7.3 Hard Training Set (Current Adversarial Training)
```python
# Use weighted selector for realistic distribution
selector_type = 'weighted'

# Heavy augmentations
augmentation_config = create_heavy_config()

# High windowed mode
WINDOWED_MODE_PROBABILITY = 0.10  # 10% windowed

# Aggressive color strip degradation
strip_degradation_probability = 0.95  # 95%
```

### 7.4 Minimal Augmentation (Debugging/Baseline)
```python
# Use smooth selector
selector_type = 'smooth'

# No augmentations
augmentation_config = create_no_augmentation_config()

# No windowed mode
WINDOWED_MODE_PROBABILITY = 0.0

# No color strip degradation
strip_degradation_probability = 0.0
```

---

## 8. File Reference Summary

### Core Generation Scripts
- `Core_Playmat_Generator.py` - Main generation script, single image
- `parallel_generate_dataset.py` - Parallel batch generation wrapper

### Card Selection
- `card_selector.py` - Weighted popularity-based selection
- `card_selector_smooth.py` - Even distribution selection with tracking

### Augmentation System
- `augmentation_config.py` - Configuration dataclasses and presets
- `augmentations.py` - Augmentation functions (blur, glare, shadow, color)
- `augmentation_presets.py` - Named preset configurations with examples

### Supporting Scripts
- `windowed_mode.py` - Windowed mode simulation
- `generation_utils.py` - Shared utilities (background management)
- `generate_background_variations.py` - Pre-generates jittered backgrounds

### Data Files
- `data/card.json` - Complete card database
- `data/card_weights_all_printings.json` - Popularity weights
- `data/card_name_to_class_id.json` - YOLO class ID mapping
- `smooth_selector_state.json` - Usage tracking for smooth selector

---

## 9. Recommended Parameter Changes for Validation Set

Based on discussion to create easier, more realistic validation set:

### Changes from Current Hard Training Set:

1. **Selector Type:** Keep `'smooth'` for even distribution
2. **Blur:** Reduce from 50% max to 30% max probability
3. **Glare:** Reduce from 25% to 15% probability
4. **Shadow:** Reduce intensity from 0-80% to 0-40% range
5. **Color adjustments:** Keep moderate (already reduced by 35%)
6. **Sleeves:** Reduce from 60% to 40% probability
7. **Occluders:** Reduce from 10% per card to 5% per card
8. **Windowed mode:** Reduce from 5% to 2% probability
9. **Color strip degradation:** Keep at 85% (this is important for avoiding shortcut learning)

### Suggested New Preset: "Realistic Validation"
```python
# Copy to augmentation_config.py:
def create_realistic_validation_config() -> AugmentationConfig:
    """Realistic validation set with moderate difficulty"""
    config = AugmentationConfig()
    config.blur.probability_range = (0.15, 0.30)
    config.glare.probability = 0.15
    config.shadow.probability = 0.35
    config.shadow.intensity_range = (0.0, 0.40)
    config.color.probability = 0.5
    config.sleeves.probability = 0.40
    config.sleeves.glare_probability = 0.30
    config.deck_boxes.probability = 0.10
    config.occluders.probability = 0.15
    config.jitter.probability = 0.8
    config.bbox_jitter.probability = 0.5
    return config
```

---

## 10. Usage Examples

### Generate 2000 Easy Validation Images:
```bash
cd /workspace/FaBCode

# Ensure backgrounds exist
python scripts/synthetic_generation/generate_background_variations.py --num-variations 2000

# Generate dataset
python scripts/synthetic_generation/parallel_generate_dataset.py \
    --num-images 2000 \
    --num-processes 80 \
    --selector smooth
```

Then manually adjust `augmentation_config.py` to use `create_light_config()` or custom preset.

### Generate 30000 Additional Training Images:
```bash
# Ensure backgrounds
python scripts/synthetic_generation/generate_background_variations.py --num-variations 30000

# Generate dataset with weighted selector
python scripts/synthetic_generation/parallel_generate_dataset.py \
    --num-images 30000 \
    --num-processes 80 \
    --selector weighted
```

---

## Notes

- All parameters are currently **hardcoded** in the scripts
- To change parameters, **edit the source files** before generation
- Consider creating a **config file system** for future flexibility
- The `augmentation_presets.py` file provides examples of successful combinations
- **Test small batches** (10-50 images) before generating large datasets
- Use `test_generation.py` for quick single-image testing with visualization

---

**End of Document**
