# Hard Case Implementation

## Overview
Added synthetic hard case effect to replace sleeves for Equipment, Weapon, and Hero cards with 50% probability.

## Features

### Visual Characteristics
Hard cases simulate clear plastic cases with:
- **Clearish white halo**: 5-7px border with semi-transparent white (RGB 240-255)
- **Edge brightness gradient**: Brighter near edges, fading inward (simulates thick plastic)
- **Extra glare spots**: 2-4 random glare reflections (10-25px radius)
  - 60% chance of edge/corner placement
  - 40% chance of random placement
- **Black edge artifacts**: 1-2 thin black lines (1-2px) parallel to each edge (simulates case seams)
- **Surface artifacts**: 3-8 small scratches/dust marks (2-6px lines)

### Application Rules
1. **Card Type Filter**: Only applies to:
   - Equipment cards
   - Weapon cards
   - Hero cards

2. **Mutually Exclusive**: Hard cases and sleeves are mutually exclusive
   - 50% chance: Hard cases (Equipment/Weapon/Hero only)
   - 50% chance: Sleeves (all cards) OR bare cards

3. **Combat Chain Exclusion**: Combat chain cards always use sleeves/bare, never hard cases

## Implementation Details

### New Function
```python
def apply_hard_case(card_img)
```
- Creates 5-7px border around card
- Applies clearish white base color
- Adds brightness gradient at edges
- Places 2-4 glare spots with circular falloff
- Draws 1-2 black lines parallel to each edge
- Adds 3-8 small surface artifacts

### Modified Functions
1. **apply_card_augmentations()**: Added `use_hard_case` parameter
   - Checks flag before applying hard case or sleeve
   - Hard case applied if `use_hard_case=True`
   - Sleeve applied if `sleeve_color is not None` and `use_hard_case=False`

2. **place_hero_card()**: Added `use_hard_case` parameter
   - Heroes always get hard case when flag is True

3. **place_standard_card()**: Added `use_hard_case` parameter
   - Checks card types (Equipment/Weapon) before applying

4. **place_combat_chain_card()**: Added `use_hard_case` parameter
   - Always passes `False` (combat chain cards excluded)

### Configuration
In `main()` function:
```python
use_hard_case = random.random() < 0.5  # 50% probability
```

## Testing Results
- Hard cases successfully render with clearish white halo and edge artifacts
- Equipment, Weapon, and Hero cards receive hard cases when flag is True
- Action/Attack/Defense cards in combat chain zones never get hard cases
- Approximately 50/50 distribution between hard case and sleeve scenarios
- Hard cases are mutually exclusive with sleeves (correct behavior)

## Usage
Hard case effect is automatically applied during synthetic image generation:
```bash
python scripts/synthetic_generation/Core_Playmat_Generator.py
```

Output shows one of:
- `Protection: Hard cases (Equipment/Weapon/Hero only)`
- `Sleeve color: RGB(R, G, B)`
- `Sleeve: None (bare cards)`

## Visual Examples
Hard cases appear as:
- Thicker border (5-7px vs 3px for sleeves)
- Bright white/clearish appearance around edges
- Multiple glare reflections on surface
- Thin black seam lines at edges
- Small scratches/dust marks on surface
