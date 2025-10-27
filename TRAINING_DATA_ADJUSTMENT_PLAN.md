# Training Data Adjustment Plan - Phase 4

## Problem Assessment

Current Phase 3 model (90.5% mAP) is performing worse on recent games compared to previous games. Root cause: training data optimized too heavily for edge cases (difficult augmentations) at the expense of clearly visible common scenarios.

## Current "Difficult" Parameters

### 1. **Card Rotation** (Core_Playmat_Generator.py)
- **Hero Cards** (lines 1219-1233):
  - Base rotation: ±90° (left/right orientation)
  - Random variance: `±6°` around base
  - Additional taps: 6% chance of +90°/180°/270° (tapped/flipped cards)
  
- **Standard Zone Cards** (lines 1277-1291):
  - Base rotation: ±90° (left/right orientation)
  - Random variance: `±6°` around base
  - Additional taps: 6% chance of +90°/180°/270°
  
- **Combat Chain Cards** (line 1307):
  - **Full random rotation**: `-180° to +180°` ⚠️ **TOO DIFFICULT**
  - No constraints, cards can be at any angle

### 2. **Card Overlap** (lines 1320-1321)
- **Training mode**: `65%` max overlap allowed ⚠️ **TOO DIFFICULT**
- **Validation mode**: `45%` max overlap
- Higher overlap = more occlusion = harder to detect

### 3. **Shadow Intensity** (augmentation_config.py, lines 31-39)
- **Range**: `0.0 to 0.80` (0-80% darkness)
- **Probability**: 50% of images get shadows
- **Types**: spot, gradient, vignette, uniform
- Can create very dark regions that obscure cards

### 4. **Color Adjustments** (augmentation_config.py, lines 43-70)
**Current ranges** (already reduced by 35% from original):
- Brightness: `0.935 to 1.065` (multiply factor)
- Contrast: `0.935 to 1.065`
- Saturation: `0.9025 to 1.0975`
- Hue shift: `-3° to +3°`
- Tint intensity: `0.0325 to 0.0975` (3.25% to 9.75%)
- Tint probability: 70% of color-adjusted images

### 5. **Blur Intensity** (augmentation_config.py, lines 11-17)
- **Range**: `15% to 50%` blur intensity per image
- Kernel size: 3 to 9 (odd numbers)
- Sigma: 0.5 to 2.0

## Proposed Adjustments for "Easier" Training Data

### Priority 1: Glare Reduction ⭐ CRITICAL
**Current**: 25% probability, intensity 0.3-0.7, radius 100-300px, 1-3 spots
**Proposed**: 8% probability, intensity 0.2-0.4, radius 80-200px, 1-2 spots

**Reasoning**: Glare creates bright washout that obscures card details. 25% is too frequent and too intense. Most real gameplay has minimal glare. Reducing to 8% makes it occasional, smaller spots prevent large washout areas.

```python
# augmentation_config.py, lines 20-26
probability: float = 0.08  # Was 0.25
intensity_range: Tuple[float, float] = (0.2, 0.4)  # Was (0.3, 0.7)
radius_range: Tuple[int, int] = (80, 200)  # Was (100, 300)
num_spots_range: Tuple[int, int] = (1, 2)  # Was (1, 3)
```

### Priority 2: Shadow Intensity Reduction ⭐ CRITICAL
**Current**: 0.0 to 0.80 (up to 80% darkness)
**Proposed**: 0.0 to 0.50 (max 50% darkness)

**Reasoning**: 80% darkness creates near-black regions. Typical indoor gaming has decent lighting. Keep shadows for realism but reduce extreme darkness.

```python
# augmentation_config.py, line 36
intensity_range: Tuple[float, float] = (0.0, 0.50)  # Was (0.0, 0.80)
```

### Priority 3: Card Overlap Reduction
**Current Training**: 65% max overlap
**Proposed Training**: 45% max overlap (match validation mode)

**Reasoning**: 65% overlap creates heavily occluded cards that are rare in actual gameplay. Players naturally space cards for readability.

```python
# Core_Playmat_Generator.py, lines 1320-1321
max_overlap = 45  # Use realistic overlap for both modes (was 65 for training)
```

### Priority 4: Blur Intensity Reduction
**Current**: 15% to 50% blur intensity
**Proposed**: 15% to 35% blur intensity

**Reasoning**: 50% blur makes cards very difficult to read. Most cameras maintain reasonable focus during gameplay. Reduce maximum blur while keeping realistic motion blur.

```python
# augmentation_config.py, line 13
probability_range: Tuple[float, float] = (0.15, 0.35)  # Was (0.15, 0.50)
```

### ~~Priority 5: Hero/Standard Card Rotation Variance~~ **REVERTED**
**Status**: KEEPING ORIGINAL ROTATION - rotation is NOT the problem

### ~~Priority 1: Combat Chain Rotation~~ **REVERTED**
**Status**: KEEPING ORIGINAL ROTATION (-180° to +180°) - rotation is NOT the problem

## Implementation Changes

### File 1: `augmentation_config.py`

**Lines 20-26** - Drastically reduce glare (MOST CRITICAL):
```python
probability: float = 0.08  # Was 0.25 (reduced by 68%)
intensity_range: Tuple[float, float] = (0.2, 0.4)  # Was (0.3, 0.7)
radius_range: Tuple[int, int] = (80, 200)  # Was (100, 300)
num_spots_range: Tuple[int, int] = (1, 2)  # Was (1, 3)
```

**Line 13** - Reduce max blur:
```python
probability_range: Tuple[float, float] = (0.15, 0.35)  # Was (0.15, 0.50)
```

**Line 36** - Reduce max shadow darkness:
```python
intensity_range: Tuple[float, float] = (0.0, 0.50)  # Was (0.0, 0.80)
```

### File 2: `Core_Playmat_Generator.py`

**Lines 1320-1321** - Reduce overlap (keep realistic for both modes):
```python
max_overlap = 45  # Use realistic overlap for both modes (was 65 for training)
```

**NO ROTATION CHANGES** - Rotation is NOT the problem, keeping all original rotation logic.

## Expected Improvements

1. **Glare Reduction**: MASSIVE improvement - 68% less glare (8% vs 25%), smaller spots, lower intensity
2. **Shadow Reduction**: Less extreme darkness (50% max vs 80% max) = better visibility
3. **Card Overlap**: Less occlusion = better detection of partially visible cards
4. **Blur Reduction**: Less extreme blur = cards more readable
5. **Overall Confidence**: Higher confidence scores on clearly visible cards with normal lighting

**Rotation UNCHANGED** - keeping all original rotation logic (heroes, standard zones, combat chain)

## Testing Plan

### Step 1: Generate Test Batch
```bash
cd "c:\VS Code\FaB Code\scripts\synthetic_generation"
python generate_dataset_25k.py --num_images 100 --output_dir "../../data/test_easier_augmentations"
```

### Step 2: Visual Inspection
- Open 20-30 random images from test batch
- Verify cards are more realistic and readable
- Check that difficulty is reduced but still has variety
- Ensure weighted card selection still working

### Step 3: Compare to Phase 3 Data
- Open some Phase 3 training images for comparison
- Confirm new data is "easier" (more visible, less occlusion, better angles)
- Verify still has augmentation variety (not too easy)

### Step 4: Full Dataset Generation
If test batch looks good:
```bash
python generate_dataset_25k.py --num_images 25000
```
Runtime: ~7 hours (1.07s per image)

### Step 5: Phase 4 Training on RunPod
- Upload dataset (~40-50GB compressed)
- Start YOLO11x training (48 hours)
- Monitor via TensorBoard
- Estimated cost: $35-50

## Success Metrics

- ✅ Glare drastically reduced (8% probability vs 25%, smaller/weaker spots)
- ✅ Less extreme shadows (50% max vs 80% max darkness)
- ✅ Less card overlap (45% max vs 65% previous)
- ✅ Lower max blur (35% vs 50%)
- ✅ Realistic card placement and visibility
- ✅ Maintained augmentation variety
- ✅ Better detection on clearly visible cards during gameplay
- ✅ Rotation UNCHANGED (keeping original rotation logic)

## Rollback Plan

If Phase 4 model performs worse:
1. Keep current Phase 3 model weights
2. Identify which specific parameter changes caused regression
3. Adjust parameters to middle ground
4. Generate new dataset with tweaked parameters
5. Run Phase 5 training

## Notes

- Changes are **CONSERVATIVE** - reducing difficulty while maintaining realistic variation
- Focus on **most impactful changes** first (combat chain rotation, overlap)
- Weighted card selection system **UNCHANGED** (maintains proper class distribution)
- Color adjustments **UNCHANGED** (already reduced by 35% in previous iteration)
- Glare effects **UNCHANGED** (realistic as-is)
- Occluder system **UNCHANGED** (hands, dice, tokens)
- Sleeve effects **UNCHANGED** (realistic as-is)

