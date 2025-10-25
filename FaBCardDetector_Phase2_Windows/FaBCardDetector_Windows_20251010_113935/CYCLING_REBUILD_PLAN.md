# Alternative Cycling Rebuild Plan

## Problem Statement
The CV model sometimes misidentifies cards, but then correctly identifies them in later frames. Users need a way to cycle through all detected alternatives for a given card position to find the correct one.

## Current Issues
1. **Click detection unreliable** - CV2 mouse callbacks may not work in transparent overlay mode
2. **Pinning too automatic** - Pins on hover, conflicts with linger behavior
3. **No visual feedback** - User doesn't know how many alternatives exist
4. **State confusion** - Multiple tracking variables (bbox_alternatives, pinned_alternatives, etc.)
5. **Accumulation timing unclear** - When does it start/stop collecting?

## Proposed Solution

### User Flow
```
1. User hovers card → Preview shows with "Press SPACE to lock"
2. User presses SPACE → Preview locks, message "Collecting alternatives... (3s)"
3. System accumulates for 3 seconds → All detections in that region saved
4. Message changes to "Alternative 1 of 5 - Use ← → to cycle"
5. User presses ← or → to cycle through alternatives
6. User presses ESC or clicks outside to unlock
```

### Why This Works
- **Deliberate action required** - No accidental pinning
- **Clear timing** - 3 second buffer is explicit
- **Visual feedback** - User always knows status
- **Keyboard controls** - Reliable in all modes (windowed/overlay/transparent)
- **Simple state** - One boolean: locked or not

### Implementation

#### New State Variables
```python
# Replace all current cycling state with:
self.cycling_locked = False  # Is cycling mode active?
self.cycling_bbox = None  # (x1, y1, x2, y2) of locked bbox
self.cycling_alternatives = []  # [(class_id, conf, name), ...]
self.cycling_index = 0  # Current position in alternatives
self.cycling_start_time = 0  # When did we start accumulating?
self.cycling_buffer_duration = 3.0  # Seconds to accumulate
```

#### New Methods
```python
def _start_cycling(self):
    """Lock current hovered card and start accumulating alternatives."""
    
def _stop_cycling(self):
    """Unlock and clear all cycling state."""
    
def _cycle_alternative(self, direction):
    """Cycle to next/previous alternative (direction: 1 or -1)."""
    
def _update_cycling_status_text(self):
    """Return status text to display on preview."""
```

#### Keyboard Bindings
```python
# In main loop after cv2.waitKey()
if key == ord(' '):  # SPACE
    if self.cycling_locked:
        self._stop_cycling()
    else:
        self._start_cycling()
elif key == 83 or key == 2555904:  # Right arrow (Windows)
    if self.cycling_locked:
        self._cycle_alternative(1)
elif key == 81 or key == 2424832:  # Left arrow (Windows)
    if self.cycling_locked:
        self._cycle_alternative(-1)
elif key == 27:  # ESC
    if self.cycling_locked:
        self._stop_cycling()
```

#### Visual Feedback
```python
# On CardPreviewWindow or CV2 preview, add text overlay:
status_text = self._update_cycling_status_text()

# Returns:
# "Press SPACE to lock" (not locked, hovering)
# "Collecting alternatives... 3 found (2s left)" (locked, accumulating)
# "Alternative 2 of 5 - Use ← → arrows" (locked, accumulated)
# "Locked - ESC to unlock" (locked, no alternatives)
```

### Benefits
1. **Works in all modes** - Keyboard always reliable
2. **Clear intent** - User must press SPACE deliberately  
3. **Visible status** - Always know what's happening
4. **Simple state** - One locked flag, one list
5. **Buffer time** - 3 seconds to gather alternatives
6. **No conflicts** - Separate from hover/linger logic

### Integration Points

#### CardPreviewWindow Changes
- Add text overlay to display status
- Show status at top of card image
- Semi-transparent background for readability

#### Main Loop Changes  
- Check keyboard after `cv2.waitKey()`
- Update alternatives during accumulation period
- Draw status text on CV2 preview as well (for windowed mode)

#### Hero Filtering Integration
- Alternatives list should respect legal card filtering
- Don't add illegal cards to alternatives

### Testing Plan
1. Test SPACE to lock in windowed mode
2. Test SPACE to lock in overlay mode  
3. Test arrow keys cycle in both modes
4. Test ESC unlocks in both modes
5. Test accumulation period (exactly 3 seconds)
6. Test status text updates correctly
7. Test with cards that flicker between 2-3 detections
8. Test with stable detection (no alternatives)
9. Test hero filtering applies to alternatives

### Alternative: Mouse Cycling (if keyboard-only is not ideal)
If you really want mouse clicking:
- **Middle click** to lock (more deliberate than left/right)
- **Left click** to cycle backward
- **Right click** to cycle forward
- **Middle click again** to unlock

This avoids conflicts with window moving/dragging.

### Performance Considerations
- Limit alternatives to 20 (already done)
- Clear alternatives when unlocking (prevent stale data)
- Only accumulate while locked (save processing)

### Code Organization
```python
# Group all cycling code together
class CyclingManager:
    """Handles alternative detection cycling."""
    def __init__(self, max_alternatives=20, buffer_duration=3.0):
        ...
    
    def lock(self, bbox):
        """Start accumulation for given bbox."""
        ...
    
    def unlock(self):
        """Stop and clear."""
        ...
    
    def update(self, current_time, boxes, clss, confs, names):
        """Update alternatives if locked and accumulating."""
        ...
    
    def cycle(self, direction):
        """Cycle to next/prev alternative."""
        ...
    
    def get_status_text(self):
        """Return current status for display."""
        ...
    
    def get_current_card(self):
        """Return currently selected card name."""
        ...
```

This encapsulation makes it easier to test and maintain.

## Next Steps
1. Review this plan
2. Decide: keyboard-only or middle-click option?
3. Implement CyclingManager class
4. Integrate into main loop
5. Add visual feedback to previews
6. Test thoroughly

Would you like me to implement this new approach?
