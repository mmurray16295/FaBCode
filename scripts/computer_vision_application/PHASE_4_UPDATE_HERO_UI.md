# Phase 4 Update: Hero Override UI & Browse Path Fix

## Changes Made (October 27, 2025)

### 1. Hero Override UI Added ✅

**New Features:**
- **Manual Hero Selection:** Dropdown menus for Player 1 and Player 2
  - Default: "Auto Detect" (uses AI detection)
  - Override: Select specific hero from full hero list
  
- **Live Status Display:**
  - Shows "(Auto-Detected)" when AI finds a hero
  - Shows "(Manual Override)" when user selects hero manually
  - Updates in real-time during detection
  
- **Apply Override Button:**
  - Click to force the detector to use selected heroes
  - Takes effect immediately during detection
  - Rebuilds legal card pool based on selection
  
- **Reset Button:**
  - "Reset Auto-Detection" button
  - Clears both heroes and returns to auto-detect mode
  - Useful between games

**UI Location:**
The hero controls are now in the "Phase 4: Hero-Aware Filtering & Override" section:
```
┌─ Phase 4: Hero-Aware Filtering & Override ───────────┐
│ ☑ Active Hero Weight Adjustment                      │
│                                                       │
│ When enabled:                                         │
│  • Detects active hero(es) from gameplay             │
│  • Restricts YOLO to only legal cards                │
│  • Boosts confidence for meta-relevant cards         │
│  • Reduces wasted GPU compute                        │
│                                                       │
│ ───────────────────────────────────────────────────── │
│ Hero Override (Optional):                            │
│                                                       │
│ Player 1: [Dropdown: Auto Detect ▼] (Auto-Detected) │
│ Player 2: [Dropdown: Auto Detect ▼]                  │
│                                                       │
│ [Apply Hero Override] [Reset Auto-Detection]         │
└───────────────────────────────────────────────────────┘
```

### 2. Browse Dialog Path Fix ✅

**Problem:** Browse button opened at C:\ drive root

**Solution:** Now intelligently defaults to:
1. Current model's directory (if valid)
2. `models/` folder (if exists)
3. `runs/train/` folder (if exists)
4. Current directory (fallback)

**Code:**
```python
def _browse_model(self):
    initial_dir = None
    current_path = self.model_path.get()
    
    if current_path and os.path.exists(current_path):
        initial_dir = os.path.dirname(os.path.abspath(current_path))
    elif os.path.exists("models"):
        initial_dir = os.path.abspath("models")
    elif os.path.exists("runs/train"):
        initial_dir = os.path.abspath("runs/train")
    
    filename = filedialog.askopenfilename(
        title="Select Model Weights",
        initialdir=initial_dir,  # ← Smart default!
        filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")]
    )
```

### 3. Architecture Changes

**CardDetector Class:**
- Added `gui` parameter to `__init__()` for GUI communication
- Added `hero_override_active` flag to track manual vs auto mode
- New method: `_update_gui_hero_status()` - Updates GUI hero dropdowns and status labels in real-time

**DetectorGUI Class:**
- Added `_load_hero_names()` - Loads hero list from card.json
- Added `_apply_hero_override()` - Applies manual hero selection
- Added `_reset_hero_detection()` - Clears heroes and returns to auto-detect
- Fixed `_browse_model()` - Smart initial directory selection

**Hero Detection Flow:**
```
1. User enables "Active Hero Weight Adjustment" checkbox
2. Detection starts in "Auto Detect" mode
3. AI detects heroes → GUI updates automatically
4. User can override: Select hero from dropdown → Click "Apply"
5. Detector rebuilds legal pool based on override
6. Between games: Click "Reset Auto-Detection"
```

### 4. Testing Instructions

**Test Hero Override:**
1. Launch Phase 4 app
2. Enable "Active Hero Weight Adjustment"
3. Start detection
4. Wait for auto-detection OR manually select hero
5. Verify status label updates:
   - "(Auto-Detected)" for AI-detected heroes
   - "(Manual Override)" for user-selected heroes

**Test Reset:**
1. With heroes detected/overridden
2. Click "Reset Auto-Detection"
3. Verify both dropdowns return to "Auto Detect"
4. Status labels clear

**Test Browse Path:**
1. Click "Browse" next to Model Weights
2. Verify dialog opens in model directory (not C:\)
3. Select different model
4. Click "Browse" again
5. Verify dialog opens in that model's directory

### 5. Key Files Modified

- `fab_detector_app_phase_4.py`:
  - Added hero override UI (lines ~145-185)
  - Added helper methods (_load_hero_names, _apply_hero_override, _reset_hero_detection)
  - Fixed _browse_model() with smart initial directory
  - Updated CardDetector.__init__() to accept gui parameter
  - Enhanced _apply_hero_filtering() to check for manual overrides
  - Added _update_gui_hero_status() for live UI updates

### 6. Benefits

**Hero Override:**
- ✅ Correct AI mistakes without restarting
- ✅ Test specific hero matchups
- ✅ Force detection when auto-detect struggles
- ✅ See live detection status

**Browse Path Fix:**
- ✅ Saves clicks navigating to model folder
- ✅ Remembers last model location
- ✅ More professional UX

### 7. Backward Compatibility

- All changes are **additive** - no existing functionality removed
- Auto-detect still works when dropdowns set to "Auto Detect"
- Manual override is **optional** - detector works without it
- GUI gracefully handles missing card.json (empty hero list)

### 8. Known Limitations

- Hero override requires detection to be running
- Dropdown only shows heroes from card.json (must be loaded)
- Reset doesn't clear legal pool until next detection cycle
- Status label updates may have ~100ms delay (GUI thread scheduling)

### 9. Future Enhancements

- [ ] Keyboard shortcut for reset (F9)
- [ ] Persistent hero selection across sessions
- [ ] Hero images in dropdown
- [ ] Confidence threshold override per hero
- [ ] Export hero detection history

---

**Status:** ✅ Ready for Testing
**Version:** Phase 4.1
**Date:** October 27, 2025
