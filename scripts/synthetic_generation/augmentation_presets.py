"""
Augmentation Preset Templates
==============================
This file stores successful augmentation parameter combinations discovered during testing.
Each preset represents a set of parameters that produced good-looking synthetic playmat images.

Use these as:
- Reference for good parameter ranges
- Starting points for further experimentation
- Baseline configurations for training data generation
"""

from dataclasses import dataclass
from typing import Dict, Any
from datetime import datetime


@dataclass
class AugmentationPreset:
    """A named set of augmentation parameters that produced good results"""
    name: str
    description: str
    date_discovered: str
    image_filename: str
    
    # Blur parameters
    blur_intensity: float
    
    # Glare parameters
    num_glare_spots: int
    light_source_pos: tuple  # (x, y)
    
    # Color adjustment parameters
    brightness: float
    contrast: float
    saturation: float
    hue_shift: int
    tint_color: str = None
    tint_intensity: float = None
    
    # Notes about what makes this preset effective
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert preset to dictionary for easy inspection"""
        return {
            'name': self.name,
            'description': self.description,
            'date_discovered': self.date_discovered,
            'image_filename': self.image_filename,
            'blur_intensity': self.blur_intensity,
            'num_glare_spots': self.num_glare_spots,
            'light_source_pos': self.light_source_pos,
            'brightness': self.brightness,
            'contrast': self.contrast,
            'saturation': self.saturation,
            'hue_shift': self.hue_shift,
            'tint_color': self.tint_color,
            'tint_intensity': self.tint_intensity,
            'notes': self.notes
        }


# ============================================================================
# PRESET LIBRARY
# ============================================================================

PRESETS = {}


# Preset 1: "Natural Overhead Lighting"
PRESETS['natural_overhead'] = AugmentationPreset(
    name="Natural Overhead Lighting",
    description="Moderate blur with single glare spot, reduced contrast, white tint for realistic overhead lighting",
    date_discovered="2025-10-16",
    image_filename="test_playmat_20251016_105447_488.jpg",
    
    # Blur: moderate
    blur_intensity=0.282,  # 28.2%
    
    # Glare: single spot at top center
    num_glare_spots=1,
    light_source_pos=(927, 107),  # Top center of 1919x1097 image
    
    # Color: slightly darker, softer contrast, slightly more saturated, white tint
    brightness=0.95,
    contrast=0.91,
    saturation=1.07,
    hue_shift=3,
    tint_color='white',
    tint_intensity=0.147,  # 14.7%
    
    notes="""
    Why this works:
    - Moderate blur (28.2%) keeps cards readable but realistic
    - Single glare spot minimizes washout, most cards remain clear
    - Reduced contrast (0.91) mimics camera sensor compression
    - Slight darkening (0.95) prevents overexposure
    - White tint simulates overhead fluorescent/LED lighting
    - Increased saturation (1.07) compensates for white tint, maintains color visibility
    - Light source at top center creates natural shadow patterns
    
    Best for: Indoor tournament/shop play environment with overhead lighting
    """
)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_preset(name: str) -> AugmentationPreset:
    """Retrieve a preset by name"""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(PRESETS.keys())}")
    return PRESETS[name]


def list_presets() -> list:
    """Get list of all available preset names"""
    return list(PRESETS.keys())


def get_preset_ranges(preset_name: str, variance: float = 0.1) -> Dict[str, tuple]:
    """
    Generate parameter ranges based on a preset with specified variance.
    
    Args:
        preset_name: Name of the preset to base ranges on
        variance: How much to vary from preset values (0.1 = ±10%)
        
    Returns:
        Dictionary of parameter ranges suitable for random sampling
    """
    preset = get_preset(preset_name)
    
    ranges = {
        'blur_intensity_range': (
            max(0.15, preset.blur_intensity * (1 - variance)),
            min(0.50, preset.blur_intensity * (1 + variance))
        ),
        'brightness_range': (
            max(0.8, preset.brightness * (1 - variance)),
            min(1.2, preset.brightness * (1 + variance))
        ),
        'contrast_range': (
            max(0.8, preset.contrast * (1 - variance)),
            min(1.2, preset.contrast * (1 + variance))
        ),
        'saturation_range': (
            max(0.85, preset.saturation * (1 - variance)),
            min(1.15, preset.saturation * (1 + variance))
        ),
        'hue_shift_range': (
            int(preset.hue_shift - 5),
            int(preset.hue_shift + 5)
        ),
    }
    
    return ranges


def print_preset_summary(preset_name: str):
    """Print a human-readable summary of a preset"""
    preset = get_preset(preset_name)
    print(f"\n{'='*70}")
    print(f"Preset: {preset.name}")
    print(f"{'='*70}")
    print(f"Description: {preset.description}")
    print(f"Date: {preset.date_discovered}")
    print(f"Reference Image: {preset.image_filename}")
    print(f"\nParameters:")
    print(f"  Blur:        {preset.blur_intensity:.1%}")
    print(f"  Glare Spots: {preset.num_glare_spots}")
    print(f"  Light Pos:   {preset.light_source_pos}")
    print(f"  Brightness:  {preset.brightness:.2f}")
    print(f"  Contrast:    {preset.contrast:.2f}")
    print(f"  Saturation:  {preset.saturation:.2f}")
    print(f"  Hue Shift:   {preset.hue_shift}°")
    if preset.tint_color:
        print(f"  Tint:        {preset.tint_color} @ {preset.tint_intensity:.1%}")
    print(f"\nNotes:\n{preset.notes}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # Demo: print all presets
    print("\n" + "="*70)
    print("AUGMENTATION PRESET LIBRARY")
    print("="*70)
    
    for preset_name in list_presets():
        print_preset_summary(preset_name)
        
    # Demo: generate ranges with variance
    print("\nExample: Parameter ranges with ±10% variance from 'natural_overhead':")
    ranges = get_preset_ranges('natural_overhead', variance=0.1)
    for param, range_val in ranges.items():
        print(f"  {param}: {range_val}")
