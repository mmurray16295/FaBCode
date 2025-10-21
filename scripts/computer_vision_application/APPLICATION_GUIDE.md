# FaB Card Detector - Application Guide

## Overview

The FaB Card Detector provides real-time Flesh and Blood card recognition with two distinct modes optimized for different use cases.

## Applications Included

### 1. `fab_detector_app.py` (GUI Application) ⭐ RECOMMENDED
**User-friendly graphical interface for configuring and running detection.**

Features:
- Easy-to-use GUI for all settings
- Settings persist between sessions
- Start/stop detection with buttons
- Real-time performance monitoring
- No command-line knowledge needed

Launch with:
```bash
python fab_detector_app.py
```

### 2. `live_detector.py` (Command-Line Tool)
**Flexible command-line tool for advanced users and scripting.**

Features:
- Webcam detection
- Screen capture
- Image file testing
- Scriptable and automatable

Launch with:
```bash
python live_detector.py --source webcam
```

### 3. `screen_detect.py` (Legacy/Advanced)
**Original implementation with advanced Windows-specific features.**

Features:
- Transparent overlays with chroma-key
- Click-through windows
- Video file playback
- Overlay masking to prevent feedback loops

## Detection Modes Explained

### Windowed Mode
**Best for: Testing, debugging, learning what the model sees**

What you see:
- Full captured screen in a window
- Green boxes around every detected card
- Card names labeled above boxes
- FPS and detection count
- Card preview image appears when hovering over a box

Use cases:
- Testing model accuracy
- Seeing all cards at once
- Understanding detection confidence
- Debugging false positives/negatives

### Transparent Overlay Mode
**Best for: Playing online, minimal interference**

What you see:
- Mostly invisible window
- Only card preview image appears when hovering over a detected card
- FPS counter
- No boxes or labels (unless hovering)

Use cases:
- Playing online games via webcam
- Watching match videos
- Identifying opponent's cards during gameplay
- Minimal screen clutter

Features:
- **Transparency**: Background is invisible (Windows chroma-key)
- **Click-through**: Mouse clicks pass through the overlay
- **Always-on-top**: Stays visible above other windows
- **Multi-monitor**: Show overlay on different screen than capture

## Multi-Monitor Setup Guide

### Recommended Configurations

**Dual Monitor - Gaming Setup:**
```
Monitor 1 (Primary): Your game/video
Monitor 2 (Secondary): Detector overlay

Settings:
- Capture Monitor: 1 (where the game is)
- Display Monitor: 2 (where you're looking)
- Mode: Windowed (see all detections)
```

**Dual Monitor - Streaming Setup:**
```
Monitor 1: OBS/Streaming software
Monitor 2: Game capture

Settings:
- Capture Monitor: 2 (game screen)
- Display Monitor: 1 (you see detections, viewers don't)
- Mode: Transparent Overlay (doesn't interfere with game)
```

**Single Monitor:**
```
Monitor 1: Everything

Settings:
- Capture Monitor: 1
- Display Monitor: 1 (or leave default)
- Mode: Transparent Overlay (minimal interference)
- Enable: Click-through mode
```

## Settings Guide

### Confidence Threshold (0.1 - 0.95)
**How sure the model must be before showing a detection.**

- **Low (0.25)**: Shows more detections, more false positives
- **Recommended (0.69)**: Balanced from F1 optimization
- **High (0.85)**: Only very confident detections, may miss some cards

Adjust based on:
- Lighting conditions (lower in dim lighting)
- Camera quality (lower for poor quality)
- Card condition (lower for played/damaged cards)

### IOU Threshold (0.1 - 0.95)
**Controls overlapping detection suppression (NMS).**

- **Low (0.30)**: Keeps more overlapping boxes (may show duplicates)
- **Recommended (0.50)**: Standard NMS behavior
- **High (0.70)**: Very aggressive suppression (may merge nearby cards)

Adjust based on:
- Card density (higher when cards are packed together)
- Typical card spacing in your setup

### Card Preview Size
**Size of the card image shown when hovering.**

- **Small (200x280)**: Less screen space, faster loading
- **Recommended (300x420)**: Good balance of visibility and size
- **Large (400x560)**: Easiest to read, takes more space

Adjust based on:
- Screen resolution
- How close you sit to monitor
- Whether you need to read card text

## Performance Optimization

### For Best FPS:

1. **Lower confidence threshold** (fewer inference passes)
2. **Reduce card preview size** (faster image rendering)
3. **Use GPU** (install PyTorch with CUDA)
4. **Close other applications** (more system resources)
5. **Lower capture resolution** (modify code if needed)

### Expected Performance:

| Hardware | Windowed Mode | Overlay Mode |
|----------|---------------|--------------|
| CPU Only (i5/i7) | 5-10 FPS | 8-15 FPS |
| GTX 1060 | 25-35 FPS | 30-45 FPS |
| RTX 3060 | 40-60 FPS | 50-80 FPS |
| RTX 4090 | 80-120 FPS | 100-150 FPS |

## Troubleshooting

### "Model not found"
**Solution**: Click "Browse" and select the `models/best.pt` file

### No detections showing
**Possible causes**:
1. Confidence threshold too high (lower to 0.25)
2. Cards not in training data
3. Poor lighting or camera quality
4. Cards too small or far from camera

**Solutions**:
- Lower confidence threshold
- Improve lighting
- Move camera closer to cards
- Check if cards are in the model's training set

### Overlay window not transparent
**Cause**: Transparency only works on Windows

**Solutions**:
- Use Windows OS, or
- Use windowed mode instead, or
- Accept black background overlay

### Detection is slow/laggy
**Solutions**:
1. Install GPU-enabled PyTorch
2. Close other applications
3. Reduce card preview size
4. Lower confidence threshold slightly
5. Use windowed mode (simpler rendering)

### Mouse clicks not working (overlay mode)
**Cause**: Click-through mode is enabled

**Solution**: Disable "Click-through mode" in GUI settings

### Card preview shows wrong card
**Possible causes**:
1. Similar card names in database
2. Card has multiple printings
3. Model misidentified the card

**Solutions**:
- Check confidence score
- Ensure good lighting
- Retrain model with more data for that card

### Overlay appears on wrong monitor
**Solution**: Change "Display Monitor" setting to correct monitor number

## Advanced Usage

### Running from Command Line

Full control with all options:
```bash
python live_detector.py --source screen --conf 0.5 --model models/best.pt
```

Batch process images:
```bash
for img in examples/*.jpg; do
    python live_detector.py --source "$img" --save "output/$(basename $img)"
done
```

### Creating Shortcuts

**Windows:**
1. Right-click `fab_detector_app.py`
2. "Create shortcut"
3. Move shortcut to Desktop
4. Rename to "FaB Detector"

**macOS/Linux:**
Create `fab_detector.command` file:
```bash
#!/bin/bash
cd "$(dirname "$0")"
python3 fab_detector_app.py
```
Make executable: `chmod +x fab_detector.command`

### Integration with OBS

To overlay detections on a stream:
1. Run detector in windowed mode
2. In OBS: Add "Window Capture" source
3. Select "FaB Card Detector" window
4. Use chroma key filter to remove background

## Tips & Best Practices

### For Online Play
- Use transparent overlay mode
- Enable click-through if overlay blocks clicks
- Place overlay on secondary monitor if possible
- Keep confidence high (0.7+) to avoid false positives

### For Learning/Analysis
- Use windowed mode to see all detections
- Lower confidence (0.4) to catch difficult cards
- Take screenshots of interesting detections
- Monitor FPS to ensure smooth experience

### For Streaming
- Run detector on second monitor (not visible to viewers)
- Use high confidence (0.8+) to avoid confusion
- Consider manual verification before announcing cards
- Test setup before going live

### For Testing the Model
- Use windowed mode with all boxes visible
- Try different lighting conditions
- Test with various card conditions (mint, played, damaged)
- Note which cards are misidentified
- Use confidence threshold around 0.25 to see edge cases

## Safety & Fair Play

**Important Notes:**
- This tool is for personal learning and analysis
- Always check tournament rules before use in competitive play
- Many online platforms prohibit detection software
- Use responsibly and ethically
- Model accuracy is not 100% - always verify

## Model Information

### Current Model (Phase 1)
- **Classes**: 100 (most popular FaB cards)
- **Architecture**: YOLOv11 nano
- **Training**: Synthetic playmat screenshots
- **Performance**: 99% mAP50 on validation set

### Limitations
- Only detects cards in training set (top 100)
- Best on standard playmat backgrounds
- May struggle with extreme angles or occlusion
- Lighting conditions affect accuracy

### Future Models
- Phase 2: 500 classes (mission-critical cards)
- Phase 3: 1100 classes (all competitive cards)
- Phase 4: 2000+ classes (comprehensive coverage)

## Support

### Getting Help
1. Check this guide thoroughly
2. Review troubleshooting section
3. Test with known-good images first
4. Check model file exists and is correct

### Reporting Issues
When reporting problems, include:
- Operating system
- Python version
- GPU model (if applicable)
- Detection mode being used
- Screenshot of issue
- Model file path

## License & Credits

Model trained on FaB card images and synthetic data.
For personal/educational use only.

Flesh and Blood TCG © Legend Story Studios
