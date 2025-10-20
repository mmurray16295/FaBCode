# Phase 1 Synthetic Data Generation Plan

## Overview
Training a FaB card detection model from scratch using curriculum learning approach.
**Goal**: 95%+ accuracy on top ~1500 cards, with steep dropoff acceptable after that.

## Training Philosophy: Curriculum Learning
Start with fewer classes so the model learns "what is a card" before scaling to fine-grained classification.

---

## Phase 1: Learn "What is a Card?" (100 Classes)

### Target
- **Classes**: Top 100 most popular cards (ranks 1-100)
- **Total Images**: 3,000
  - 2,700 (90%) from `generate_synthetic_playmat_screenshots.py`
  - 300 (10%) from `generate_random_playmat.py`

### Rationale
- Small enough class count for quick convergence
- Large enough for diversity (100 different card arts)
- High images-per-class ratio (~30 average, top cards get 50+)
- Weighted sampling ensures most popular cards are well-represented
- 10% random backgrounds prevent overfitting to 51 labeled templates

### Commands

#### Step 1: Generate Synthetic Playmat Data (90%)
```bash
cd /root/FaBCode
python3 scripts/generate_synthetic_playmat_screenshots.py \
    --num-images 2700 \
    --popularity-min 1 \
    --popularity-max 100 \
    --use-popularity-weights \
    --card-dirs data/images/*/
```

**What this does:**
- Uses 51 hand-labeled YouTube backgrounds (35 train / 10 valid / 6 test)
- Places cards in realistic playmat positions (from positioning cache)
- Applies screen capture simulation (JPEG, blur, chroma subsample, etc.)
- Adds occluders (arms, dice, tokens)
- Weights sampling so most popular cards appear more frequently

#### Step 2: Generate Random Playmat Data (10%)
```bash
cd /root/FaBCode
python3 scripts/generate_random_playmat.py \
    --num-images 300 \
    --popularity-min 1 \
    --popularity-max 100 \
    --use-popularity-weights \
    --card-dirs data/images/*/
```

**What this does:**
- Generates procedural backgrounds (solid colors, gradients, textures, etc.)
- Places cards in completely random positions (no template cache)
- Applies screen capture simulation (newly added!)
- Adds occluders (newly added!)
- Provides position diversity to prevent overfitting

### Output
- Images: `/root/FaBCode/data/synthetic/{train,valid,test}/images/`
- Labels: `/root/FaBCode/data/synthetic/{train,valid,test}/labels/`
- Class index: `/root/FaBCode/data/synthetic/classes.yaml`
- Data config: `/root/FaBCode/data/synthetic/data.yaml`

### Expected Training Results
- **50-100 epochs** until validation loss plateaus
- **>90% accuracy** on these 100 cards
- High confidence bounding boxes
- Clean validation curves (no overfitting)

### Success Criteria
If Phase 1 achieves >90% accuracy on 100 cards, proceed to Phase 2.
If not, indicates a fundamental problem (wrong model, bad data, etc.)

---

## Phase 2: Expand Understanding (300 Classes)

### Target
- **Classes**: Top 300 cards (ranks 1-300)
- **Total Images**: 5,000+
  - 4,500 (90%) synthetic playmat
  - 500 (10%) random playmat

### Commands
```bash
# Synthetic (90%)
python3 scripts/generate_synthetic_playmat_screenshots.py \
    --num-images 4500 \
    --popularity-min 1 \
    --popularity-max 300 \
    --use-popularity-weights \
    --card-dirs data/images/*/

# Random (10%)
python3 scripts/generate_random_playmat.py \
    --num-images 500 \
    --popularity-min 1 \
    --popularity-max 300 \
    --use-popularity-weights \
    --card-dirs data/images/*/
```

### Training
- **Start from Phase 1 weights** (fine-tune)
- 30-50 epochs
- Model now knows what cards are, learns finer distinctions

---

## Phase 3: Scale to Mid-Tier (750 Classes)

### Target
- **Classes**: Top 750 cards (ranks 1-750)
- **Total Images**: 12,000+
  - 10,200 (85%) synthetic playmat
  - 1,800 (15%) random playmat

### Commands
```bash
# Synthetic (85%)
python3 scripts/generate_synthetic_playmat_screenshots.py \
    --num-images 10200 \
    --popularity-min 1 \
    --popularity-max 750 \
    --use-popularity-weights \
    --card-dirs data/images/*/

# Random (15%)
python3 scripts/generate_random_playmat.py \
    --num-images 1800 \
    --popularity-min 1 \
    --popularity-max 750 \
    --use-popularity-weights \
    --card-dirs data/images/*/
```

**Note**: Increased random % (15%) due to more repetition risk with 12k images.

---

## Phase 4: Final Target (1500 Classes)

### Target
- **Classes**: Top 1500 cards (ranks 1-1500)
- **Total Images**: 20,000+
  - 17,000 (85%) synthetic playmat
  - 3,000 (15%) random playmat

### Commands
```bash
# Synthetic (85%)
python3 scripts/generate_synthetic_playmat_screenshots.py \
    --num-images 17000 \
    --popularity-min 1 \
    --popularity-max 1500 \
    --use-popularity-weights \
    --card-dirs data/images/*/

# Random (15%)
python3 scripts/generate_random_playmat.py \
    --num-images 3000 \
    --popularity-min 1 \
    --popularity-max 1500 \
    --use-popularity-weights \
    --card-dirs data/images/*/
```

### Expected Results
- **95%+ accuracy on top 1500 cards**
- Steep dropoff acceptable for cards 1501+
- Model excels at most commonly played cards

---

## Why This Approach Works

### 1. Curriculum Learning
- Model builds understanding incrementally
- Each phase starts from good initialization
- Prevents "drowning" in too many similar classes

### 2. Weighted Sampling
- Most popular cards get more training examples
- Matches real-world distribution
- Model learns what it will see most in production

### 3. Mixed Data Sources
- **Synthetic playmat (85-90%)**: Realistic domain, screen capture effects
- **Random playmat (10-15%)**: Position diversity, regularization
- Prevents overfitting to 51 background templates

### 4. Progressive Mixing
- Phase 1 (3k images): 90/10 split (less overfitting risk)
- Phases 3-4 (12k-20k images): 85/15 split (more regularization needed)

---

## Scripts Enhanced for This Plan

### `generate_synthetic_playmat_screenshots.py`
- ✅ Paths updated to Linux
- ✅ Uses real YouTube backgrounds
- ✅ Positioning cache for realistic card placement
- ✅ Screen capture simulation
- ✅ Occluders (arms, dice, tokens)
- ✅ Supports popularity filtering and weighting

### `generate_random_playmat.py`
- ✅ Paths updated to Linux
- ✅ `--card-dirs` argument added
- ✅ **NEW**: Screen capture simulation added
- ✅ **NEW**: Occluders added
- ✅ Procedural backgrounds (textures match FaB playmats)
- ✅ Truly random positioning (no cache)
- ✅ Full 0-359° rotation (players face each other + tapping)
- ✅ Supports popularity filtering and weighting

---

## Current Status

### ✅ Completed
1. Dependencies installed (opencv, ultralytics, etc.)
2. Background images split (35 train / 10 valid / 6 test)
3. Paths fixed in both generation scripts
4. Random playmat script enhanced with screen capture + occluders

### 🔄 In Progress
- Downloading all card images (currently set 35/86)

### ⏳ Pending
- Run Phase 1 generation (waiting for downloads)
- Train Phase 1 model
- Evaluate and proceed to Phase 2-4

---

## Key Design Decisions

### Full 0-359° Rotation ✓
- Players face each other (90°/270° base rotation)
- Cards get tapped (±90° additional rotation)
- Result: All angles possible and necessary

### Varied Backgrounds ✓
- Real backgrounds: YouTube gameplay (realistic domain)
- Random backgrounds: Procedural textures (prevent overfitting)
- Both simulate FaB playmat variety (solid colors + hero/card art)

### Zone Rules NOT Enforced ✓
- Cards can appear anywhere on mat
- Realistic for:
  - Players sometimes misplace cards
  - Camera angles showing cards outside zones
  - Simpler training objective = better accuracy

### No `generate_random_playmat.py` Exclusively ✗
- Would miss screen capture artifacts (critical for production)
- Would miss realistic camera perspective/lighting
- Random used only for regularization, not as primary data source

---

## Next Steps

1. **Wait for downloads** to complete (~45 more sets)
2. **Run Phase 1 generation** (commands above)
3. **Train Phase 1 model** using `yolo11n.pt` base
4. **Evaluate**: Should see >90% accuracy on 100 cards
5. **Proceed to Phase 2** if successful

