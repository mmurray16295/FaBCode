# Alternative Detection Cycling - Original Logic Backup
**Date:** October 24, 2025  
**Status:** Buggy implementation - saved for reference before rebuild

## Original Concept
When hovering over a bbox, pin a region and accumulate all detection alternatives that appear in that region over time. Allow user to cycle through alternatives with left/right click.

## Current Implementation Issues
- Pinning behavior conflicts with linger timeouts
- Alternative accumulation doesn't have bounds checking
- Click detection may not be registering properly
- State management between pinned region and stable card is confusing

## Original Code Sections

### 1. Pinned Region State (from __init__)
```python
# Alternative detections for right-click cycling
self.bbox_alternatives = {}  # Maps bbox location -> list of (class_id, conf, name)
self.current_alternative_index = {}  # Maps bbox location -> current index

# Pinned region for accumulating alternatives
self.pinned_region = None  # (exp_x1, exp_y1, exp_x2, exp_y2) - expanded bbox
self.pinned_alternatives = []  # List of (class_id, conf, name) tuples
self.pinned_alternative_index = 0  # Current position in alternatives list
```

### 2. Update Pinned Region Alternatives (lines 1389-1419)
```python
def _update_pinned_region_alternatives(self, boxes, clss, confs, names):
    """Update alternatives for the currently pinned region (if any).
    
    When a card is hovered, we pin an expanded region around it and accumulate
    all detections that appear in that region across frames.
    """
    if not hasattr(self, 'pinned_region'):
        self.pinned_region = None
        self.pinned_alternatives = []
    
    # If we have a pinned region, look for detections that fall within it
    if self.pinned_region is not None:
        exp_x1, exp_y1, exp_x2, exp_y2 = self.pinned_region
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(v) for v in box]
            class_id = int(clss[i])
            conf = float(confs[i])
            
            if isinstance(names, dict):
                name = names.get(class_id, str(class_id))
            else:
                name = str(class_id)
            
            # Check if this detection overlaps with the pinned region
            if not (x2 < exp_x1 or x1 > exp_x2 or y2 < exp_y1 or y1 > exp_y2):
                # It overlaps - now check if card is legal
                
                # Skip hero cards from filtering
                if not self._is_hero_card(name):
                    # Apply legal card filtering if heroes are detected
                    if self.legal_card_names is not None:
                        card_name_lower = name.replace('_', ' ').lower().strip()
                        if card_name_lower not in self.legal_card_names:
                            # Illegal card - skip it
                            continue
                
                # Card is legal (or is a hero card) - check if we already have it
                card_tuple = (class_id, conf, name)
                
                # Check if this card name is already in alternatives (avoid duplicates)
                if not any(alt[2] == name for alt in self.pinned_alternatives):
                    self.pinned_alternatives.append(card_tuple)
                    # Re-sort by confidence
                    self.pinned_alternatives.sort(key=lambda x: x[1], reverse=True)
                    print(f"[pinned] Added {name} ({conf:.2f}) to region. Total: {len(self.pinned_alternatives)}")
```

### 3. Cycle Alternative Detection (lines 1713-1780)
```python
def _cycle_alternative_detection(self, direction=1):
    """Cycle to the next/previous alternative detection for the pinned region.
    
    Args:
        direction: 1 for forward (right-click), -1 for backward (left-click)
    """
    try:
        print(f"[cycle] Method called with direction={direction}")
        
        # Check if we have pinned alternatives
        if not hasattr(self, 'pinned_alternatives') or not self.pinned_alternatives:
            print("[cycle] No pinned alternatives available")
            return
        
        if len(self.pinned_alternatives) <= 1:
            print(f"[cycle] Only 1 alternative available, cannot cycle")
            return
        
        # Ensure we have an index
        if not hasattr(self, 'pinned_alternative_index'):
            self.pinned_alternative_index = 0
        
        # Move index in specified direction (with wrapping)
        self.pinned_alternative_index = (self.pinned_alternative_index + direction) % len(self.pinned_alternatives)
        
        # Get the new detection
        class_id, conf, name = self.pinned_alternatives[self.pinned_alternative_index]
        
        direction_str = "forward" if direction > 0 else "backward"
        print(f"[cycle] Cycled {direction_str} to alternative {self.pinned_alternative_index + 1}/{len(self.pinned_alternatives)}: {name} ({conf:.2f})")
        
        # Update stable card and force reload
        self.stable_card_name = name
        
        # Update hover time to keep preview visible after cycling
        self.preview_last_hover_time = time.time()
        
        # Keep the bbox from the original detection (for display purposes)
        if self.stable_card_bbox:
            x1, y1, x2, y2, _ = self.stable_card_bbox
            self.stable_card_bbox = (x1, y1, x2, y2, name)
        
        # Force card image reload
        image_url = self.get_image_url_by_name(name)
        if image_url:
            try:
                response = requests.get(image_url, timeout=2)
                if response.status_code == 200:
                    pil_image = Image.open(BytesIO(response.content))
                    self.stable_card_pil = pil_image
                    
                    # Update CV2 preview image
                    card_img = pil_image.resize((self.args.card_size[0], self.args.card_size[1]))
                    card_np = np.array(card_img)
                    card_np = cv2.cvtColor(card_np, cv2.COLOR_RGBA2BGRA)
                    self.stable_card_image = card_np
                    print(f"[cycle] Updated CV2 preview for {name}")
                    
                    # Update Tkinter preview if we have the preview window
                    if hasattr(self, 'card_preview_window') and self.card_preview_window:
                        gx, gy = self._get_global_mouse_pos()
                        if self.stable_card_bbox:
                            x1, y1, x2, y2, _ = self.stable_card_bbox
                            bbox = (x1, y1, x2, y2)
                            self.card_preview_window.update_card(pil_image, gx, gy, bbox)
            except Exception as e:
                print(f"[cycle] Failed to load image for {name}: {e}")
    except Exception as e:
        print(f"[cycle] EXCEPTION in _cycle_alternative_detection: {e}")
        import traceback
        traceback.print_exc()
```

### 4. Mouse Callback (lines 1421-1433)
```python
def _mouse_callback(self, event, x, y, flags, param):
    """Mouse callback for window."""
    print(f"[MOUSE] Event: {event}, Position: ({x}, {y}), Flags: {flags}")
    if event == cv2.EVENT_MOUSEMOVE:
        self.mouse_pos = (x, y)
    elif event == cv2.EVENT_LBUTTONDOWN:
        print(f"[MOUSE] LEFT CLICK detected at ({x}, {y})")
        # Left-click: cycle backward to previous alternative
        self._cycle_alternative_detection(direction=-1)
    elif event == cv2.EVENT_RBUTTONDOWN:
        print(f"[MOUSE] RIGHT CLICK detected at ({x}, {y})")
        # Right-click: cycle forward to next alternative
        self._cycle_alternative_detection(direction=1)
```

## Rebuild Plan

### Issues to Fix:
1. **Click detection not working** - CV2 window may not be capturing clicks properly in transparent overlay mode
2. **Pinning conflicts with linger** - Preview disappears before alternatives accumulate
3. **No visual feedback** - User doesn't know how many alternatives are available
4. **State confusion** - Multiple tracking variables (bbox_alternatives vs pinned_alternatives)

### Better Approach:
1. **Single source of truth** for pinned state
2. **Clear visual indicator** showing "Alternative X of Y" 
3. **Lock preview on first click** instead of on hover (prevents accidental pinning)
4. **Accumulate alternatives for N seconds** after pinning (configurable buffer time)
5. **Use keyboard shortcuts** instead of mouse clicks if CV2 clicks don't work in overlay mode
6. **Show all alternatives in a list** on the preview window itself

### Proposed New Flow:
1. Hover over card → shows preview with "Click to pin"
2. Click on preview → locks it and starts accumulating alternatives for 3 seconds
3. Display shows "Collecting alternatives... X found"
4. After collection, show "Alternative 1 of X - Use ← → arrows to cycle"
5. Click outside or press ESC to unpin
