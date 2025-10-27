# FaB Card Detector Phase 4

## Overview
Phase 4 introduces **hero-aware detection** with pre-filtering and confidence boosting based on competitive meta data.

## Key Features

### 1. Hero Detection
- Automatically detects which hero(es) are playing
- Supports dual-hero games
- Dynamic threshold adjustment (starts at 0.69, lowers to 0.40 until heroes found)

### 2. Pre-Detection Filtering
- Uses YOLO's `classes=` parameter to restrict detection to legal cards only
- Reduces GPU compute by ~85% (only detects ~200-500 legal cards vs ~3,100 total)
- Prevents illegal card guesses entirely (before inference, not after)

### 3. Confidence Boosting
- Loads `card_weights_all_printings.json` from fabrec.gg competitive meta
- Boosts confidence for high-usage cards in detected hero's deck:
  - **70%+ usage**: +20% confidence boost
  - **50-70% usage**: +15% confidence boost
  - **30-50% usage**: +10% confidence boost
  - **15-30% usage**: +5% confidence boost
  - **<15% or not in deck**: -8% penalty
- Only adjusts borderline detections (0.20-0.45 range)

### 4. Combined Toggle
- Single "Active Hero Weight Adjustment" checkbox enables both:
  - Pre-detection filtering (restricts YOLO classes)
  - Post-detection confidence boosting (adjusts scores)
- Easy A/B testing: toggle on/off to compare performance

## Usage

### GUI Mode (Recommended)
```bash
cd "c:\VS Code\FaB Code\scripts\computer_vision_application"
python fab_detector_app_phase_4.py
```

### Command-Line Mode
```bash
python fab_detector_app_phase_4.py --weights "path/to/model.pt" --conf 0.69 --hero-filtering
```

## Architecture

### Detection Pipeline
```
1. Screen Capture (mss)
   ↓
2. Hero Detection (if enabled)
   - Check for hero cards
   - Build legal card pool based on class/talent matching
   - Generate legal class IDs for YOLO filtering
   ↓
3. YOLO Detection
   - If hero filtering OFF: detect all 3,100 classes
   - If hero filtering ON: detect only legal class IDs (~200-500)
   ↓
4. Confidence Boosting (if enabled)
   - Load card usage % from meta data
   - Boost legal cards based on deck membership
   - Apply multiplier (0.92x to 1.20x)
   ↓
5. Geometry Filtering
   - Aspect ratio: 0.6 - 2.2
   - Area: 0.4% - 15% of screen
   ↓
6. Display & Card Preview
```

### Hero Matching Logic
Cards are legal for a hero if:
- **Generic or Token**: Always legal
- **Class cards**: At least ONE card class matches hero's classes (OR logic)
- **Talent cards**: At least ONE card talent matches hero's talents (OR logic)
- **No talent hero**: Excludes cards with ANY talent

Special handling:
- "Essence of X" keywords grant additional talent access
- Heroes themselves always pass through filters

## Configuration

### Files Required
- `data/card.json` - Card metadata (types, classes, talents)
- `data/card_weights_all_printings.json` - Meta usage statistics
- `confidence_booster.py` - Confidence adjustment module (must be in same directory)

### Settings Persistence
Settings saved to `detector_config_phase4.json`:
- Model path
- Confidence threshold
- IOU threshold
- Hero filtering enabled state

## Testing

### A/B Comparison
1. Run detection with hero filtering **OFF**
2. Note FPS and detection accuracy
3. Toggle hero filtering **ON**
4. Compare performance:
   - FPS should increase (less YOLO compute)
   - Accuracy should improve (fewer false positives)
   - Confidence scores should better reflect meta relevance

### Expected Results
- **FPS gain**: 15-30% (depending on hero pool size)
- **False positives**: Reduced by 80-90% (illegal cards blocked)
- **True positives**: Confidence adjusted +5% to +20% for meta cards

## Troubleshooting

### Hero not detected
- Lower initial threshold (GUI default 0.69)
- Wait ~5 seconds for dynamic threshold adjustment
- Check that hero card image is visible and unobstructed

### ConfidenceBooster not loading
- Verify `confidence_booster.py` is in same directory
- Check `data/card_weights_all_printings.json` exists
- Review console output for error messages

### Pre-filtering not working
- Ensure hero detected (watch console for "[hero] Detected Hero 1")
- Verify legal pool built (console shows pool size)
- Check class IDs generated (console shows legal class count)

## Development Notes

### Archived Code
- Old Phase 4 prototype moved to `archived/temp_check/`
- Added to `.gitignore` to prevent tracking

### Future Enhancements
- Persistent hero detection across sessions
- Format-specific filtering (CC vs Blitz)
- Manual hero override option
- Confidence boost visualization (colored boxes)

## Credits
- Meta data: fabrec.gg (scraped Oct 21, 2025)
- Detection model: YOLOv11
- Screen capture: mss library
