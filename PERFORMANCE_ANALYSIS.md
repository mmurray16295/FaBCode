# Performance Analysis - Synthetic Image Generation

## Summary Statistics (10 Runs)

**Average Total Time:** ~0.96 seconds per image  
**Range:** 0.807s - 1.139s  
**Throughput:** ~3,750 images/hour, ~90,000 images/day

---

## Time Breakdown by Operation

| Operation | Avg Time | % of Total | Notes |
|-----------|----------|------------|-------|
| **place_cards** | 0.66s | **69.4%** | 🔴 PRIMARY BOTTLENECK |
| window_transformations | 0.14s | 14.7% | Variable (5-24%), depends on random effect |
| initialization | 0.092s | 9.6% | CardSelector + data loading |
| load_background_labels | 0.010s | 1.0% | Efficient |
| Other operations | 0.05s | 5.3% | Various small tasks |

---

## Key Findings

### 1. **Card Placement is the Bottleneck (69.4%)**

The `place_cards` operation takes nearly 70% of total time. This includes:
- Applying augmentations (blur, glare, color, sleeves, hard cases)
- Image transformations (rotation, scaling)
- Compositing cards onto playmat
- Per-card processing for ~20-25 cards

**Breakdown within card placement:**
- **Image augmentations**: Each card goes through multiple PIL/CV2 operations
  - Blur (Gaussian blur on full card image)
  - Glare (spot-based brightness adjustment)
  - Color adjustment (brightness, contrast, saturation, hue shift, tint)
  - Sleeve application (border + color overlay)
  - Hard case application (border, glare spots, edge lines, blur - 50% of eligible cards)
- **Rotation**: PIL rotation with expand=True creates larger canvas
- **Alpha compositing**: Pasting RGBA onto RGB playmat

### 2. **Window Transformations are Variable (5-24%)**

The window color transformation varies significantly:
- Depends on random effect chosen (posterize, saturation, hue shift, etc.)
- Some effects are more expensive than others
- Could be optimized but not the primary concern

### 3. **Initialization is One-Time Cost (9.6%)**

Loading card database and weights happens once:
- 0.092s is acceptable for ~4,200 cards
- Could cache in memory for batch processing
- Not a concern for single image generation

---

## Optimization Opportunities

### 🔴 HIGH IMPACT (Target: 30-50% speedup)

#### 1. **Parallel Card Processing**
**Current:** Sequential processing of 20-25 cards  
**Proposed:** Batch process cards in parallel using multiprocessing/threading  
**Expected gain:** 30-40% (0.66s → 0.40s)  
**Implementation:**
```python
from concurrent.futures import ThreadPoolExecutor

# Process multiple cards simultaneously
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(apply_augmentations, card) for card in cards]
    results = [f.result() for f in futures]
```

**Challenges:**
- PIL/CV2 thread safety
- Need to ensure zone placement order is maintained
- May need process pool instead of thread pool for true parallelism

#### 2. **Optimize Hard Case Effect**
**Current:** Multiple CV2 operations per eligible card (50% chance)  
- Create 8-12px border
- Add 3-6 glare spots
- Draw 2-3 lines per edge (4 edges)
- Apply Gaussian blur

**Proposed optimizations:**
- Pre-generate hard case templates at common card sizes
- Cache blurred edge line patterns
- Reduce number of operations per card

**Expected gain:** 10-15% (0.66s → 0.56s)

#### 3. **Reduce Image Size During Processing**
**Current:** Full resolution processing (cards are ~750x1050px)  
**Proposed:** 
- Scale down before augmentations
- Apply effects at lower resolution
- Scale up only at final composite

**Expected gain:** 20-30% (0.66s → 0.46s)  
**Risk:** May affect visual quality - need testing

### 🟡 MEDIUM IMPACT (Target: 10-20% speedup)

#### 4. **Cache Common Augmentation Results**
- Sleeve overlays (9 common colors)
- Glare patterns (reused across cards in same image)
- Color adjustments (same for all cards in image)

**Expected gain:** 5-10%

#### 5. **Optimize Window Transformations**
- Skip expensive effects occasionally
- Use faster numpy operations instead of PIL
- Pre-compute transformation matrices

**Expected gain:** 5-10% (but variable impact)

#### 6. **Batch Image Loading**
- Already implemented (image_cache)
- Could pre-load common cards across multiple images
- Persistent cache between script runs

**Expected gain:** 2-5%

### 🟢 LOW IMPACT (Target: <5% speedup)

#### 7. **Optimize Sleeve Application**
- Sleeve code is already efficient
- 3px border is minimal overhead
- Not worth optimizing further

#### 8. **Reduce Debug Printing**
- Current: ~20-30 print statements per image
- Could reduce or make conditional
- Minimal impact (~1-2%)

---

## Recommended Action Plan

### Phase 1: Quick Wins (1-2 hours)
1. ✅ **Add timing instrumentation** (COMPLETED)
2. Profile hard case function specifically
3. Reduce hard case complexity (fewer glare spots, simpler blur)
4. Cache sleeve/hard case templates

**Expected:** 10-15% speedup → 0.82s per image (~4,400/hour)

### Phase 2: Parallel Processing (4-6 hours)
1. Implement parallel card augmentation
2. Test thread safety with PIL/CV2
3. Benchmark with 2, 4, 8 workers

**Expected:** 30-40% cumulative speedup → 0.58s per image (~6,200/hour)

### Phase 3: Advanced Optimizations (8-12 hours)
1. Implement multi-resolution processing
2. Pre-compute common augmentation patterns
3. Optimize window transformations
4. Consider Rust/C++ extensions for hot paths

**Expected:** 50-60% cumulative speedup → 0.38s per image (~9,500/hour)

---

## Cost-Benefit Analysis

| Optimization | Time to Implement | Expected Speedup | ROI |
|--------------|-------------------|------------------|-----|
| Hard case simplification | 1-2 hours | 10-15% | ⭐⭐⭐⭐⭐ |
| Parallel processing | 4-6 hours | 30-40% | ⭐⭐⭐⭐ |
| Multi-resolution | 8-12 hours | 20-30% | ⭐⭐⭐ |
| Window optimization | 2-3 hours | 5-10% | ⭐⭐ |
| Cache enhancements | 3-4 hours | 5-10% | ⭐⭐ |

---

## Current Performance is Good!

**Context:**
- **1.07s per image** is actually quite fast for synthetic data generation
- Includes complex augmentations (blur, glare, color, sleeves, hard cases, occluders)
- Generates high-quality labeled images with realistic variations
- Already optimized (image caching, zone lookups, batch operations)

**When to optimize:**
- If generating >100k images (worth the dev time)
- If running on slower hardware
- If need real-time generation

**When NOT to optimize:**
- Current speed is acceptable for dataset creation
- Development time > compute time savings
- Risk of introducing bugs

---

## Profiling Commands

```powershell
# Single run with full timing
python scripts/test_generate_simple.py

# 10 runs with timing summary
for ($i=1; $i -le 10; $i++) { 
    python scripts/test_generate_simple.py 2>&1 | Select-String -Pattern "TIMING BREAKDOWN:|TOTAL" 
}

# 100 images for throughput test
$startTime = Get-Date
for ($i=1; $i -le 100; $i++) { 
    python scripts/test_generate_simple.py 2>&1 | Out-Null 
}
$duration = (Get-Date) - $startTime
Write-Host "Total: $($duration.TotalSeconds)s, Avg: $([math]::Round($duration.TotalSeconds/100, 2))s/image"
```

---

## Hardware Context

This analysis was performed on a system with:
- Unknown CPU (likely multi-core modern processor)
- Python 3.x with PIL, OpenCV, NumPy
- Windows PowerShell environment
- SSD storage (fast image I/O)

Performance may vary significantly on:
- Cloud GPUs (RunPod, Colab) - may be faster or slower
- CPU-only servers
- Systems with slower disk I/O
