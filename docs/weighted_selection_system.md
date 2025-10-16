# Weighted Card Selection System

## Overview
The card selection system uses popularity weights from competitive deck data to create realistic synthetic playmats while maintaining variety.

## How It Works

### 1. Data Sources
- **card.json**: Complete card database with all properties
- **card_popularity_weights_by_hero.json**: Deck usage statistics per hero

### 2. Selection Distribution
- **90%** from weighted pool (cards that appear in competitive decks)
- **10%** from non-weighted pools:
  - 4% Generic cards
  - 1% Class-only cards
  - 1% Talent-only cards
  - 4% Class + Talent cards

### 3. Weight Conversion
Deck usage percentages are converted to selection weights using **power of 0.75 function**:

```
weight = usage_percentage ^ 0.75
```

This provides excellent balance between popularity and variety:
- 90% usage → weight ~29.4
- 50% usage → weight ~18.8
- 10% usage → weight ~5.6
- 1% usage → weight ~1.0

### Why Power of 0.75?
We tested three approaches:
- **Linear** (weight = usage): Too extreme, popular cards dominate completely
- **Square Root** (weight = sqrt(usage)): Good but not weighted enough (~9.5 for 90% usage)
- **Power of 0.75**: ✅ Perfect balance - strongly respects popularity while maintaining variety

The 0.75 exponent provides about **3x stronger weighting** for popular cards compared to square root, 
while still ensuring all cards have a chance to appear.

## Example Results

### Florian Rotwood Harbinger
**Equipment Selection (10,000 trials)**

| Card | Deck Usage | Weight | Selection Rate |
|------|-----------|--------|----------------|
| Grasp of the Arknight | 90.8% | 29.42 | 2.7% |
| Well Grounded | 85.6% | 28.15 | 2.5% |
| Face Purgatory | 82.0% | 27.26 | 2.4% |
| Dyadic Carapace | 76.6% | 25.88 | 2.4% |
| Nullrune Boots | 40.5% | 16.06 | 1.5% |
| Amethyst Tiara | 10.2% | 5.67 | 0.5% |

**Key Observations:**
- Popular cards (90% usage) selected ~5x more than uncommon cards (10% usage)
- Excellent differentiation: top cards appear 2.5-2.7%, mid-tier ~1.5%, low-tier ~0.5%
- All cards have reasonable chance to appear
- No card dominates (even 90% usage card only appears 2.7% of time)
- Variety maintained across equipment slots

## Benefits

1. **Realistic**: Popular cards appear more frequently (matching real play)
2. **Variety**: All legal cards can appear, preventing repetitive datasets
3. **Balanced**: No single card dominates selections
4. **Format-aware**: Cards filtered to match format legality
5. **Mathematically sound**: Avoids the trap of "90% usage = 90% selection"

## Implementation Details

### _convert_usage_to_weight()
```python
def _convert_usage_to_weight(self, usage_percentage: float) -> float:
    import math
    return math.pow(usage_percentage, 0.75)
```

### select_card()
```python
def select_card(self, card_pools: Dict[str, List[Dict]]) -> Optional[Dict]:
    roll = random.random()
    
    if roll < 0.90:  # 90% weighted
        cards = card_pools['weighted']
        weights = [self._convert_usage_to_weight(c.get('usage_percentage', 1.0)) 
                  for c in cards]
        return random.choices(cards, weights=weights, k=1)[0]
    elif roll < 0.94:  # 4% generic
        return random.choice(card_pools['generic'])
    # ... etc
```

## Validation

All tests confirm:
- ✅ 90% from weighted, 10% from non-weighted
- ✅ Popular cards selected more frequently
- ✅ All cards appear across large samples
- ✅ No mathematical errors (usage % correctly converted to weights)
- ✅ Format filtering working correctly
- ✅ Zone-specific filtering maintained

## Future Enhancements

Possible improvements:
- Adjust 90/10 split based on hero pool size
- Different weight functions per card type
- Meta-game shifts over time
- Rarity-based adjustments
