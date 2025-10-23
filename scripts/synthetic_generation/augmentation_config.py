"""
Augmentation configuration for synthetic playmat generation.
Centralizes all augmentation parameters and probability settings.
"""

from dataclasses import dataclass
from typing import Tuple
import random
import numpy as np


@dataclass
class BlurConfig:
    """Configuration for blur augmentation"""
    enabled: bool = True
    probability_range: Tuple[float, float] = (0.15, 0.50)  # 15-50% blur intensity per image
    kernel_size_range: Tuple[int, int] = (3, 9)  # Must be odd numbers
    sigma_range: Tuple[float, float] = (0.5, 2.0)


@dataclass
class GlareConfig:
    """Configuration for glare/lighting effects"""
    enabled: bool = True
    probability: float = 0.25  # 25% of images get glare
    intensity_range: Tuple[float, float] = (0.3, 0.7)
    radius_range: Tuple[int, int] = (100, 300)  # Glare spot radius in pixels
    num_spots_range: Tuple[int, int] = (1, 3)  # Number of glare spots


@dataclass
class ShadowConfig:
    """Configuration for shadow/darkness effects"""
    enabled: bool = True
    probability: float = 0.50  # 50% of images get shadows
    intensity_range: Tuple[float, float] = (0.0, 0.80)  # 0-80% darkness (0.0 = no shadow, 1.0 = completely black)
    radius_range: Tuple[int, int] = (120, 400)  # Shadow region radius in pixels
    num_spots_range: Tuple[int, int] = (1, 2)  # Number of shadow spots
    # Shadow types: 'spot' (localized), 'gradient' (directional), 'vignette' (edges), 'uniform' (overall dimming)
    shadow_types: Tuple[str, ...] = ('spot', 'gradient', 'vignette', 'uniform')


@dataclass
class ColorAdjustmentConfig:
    """Configuration for color/brightness/contrast adjustments"""
    enabled: bool = True
    probability: float = 0.5  # 50% of images get color adjustments
    brightness_range: Tuple[float, float] = (0.935, 1.065)  # Multiply factor (reduced by 35%: was 0.9-1.1)
    contrast_range: Tuple[float, float] = (0.935, 1.065)  # Multiply factor (reduced by 35%: was 0.9-1.1)
    saturation_range: Tuple[float, float] = (0.9025, 1.0975)  # Multiply factor (reduced by 35%: was 0.85-1.15)
    hue_shift_range: Tuple[int, int] = (-3, 3)  # Degrees (reduced by 35%: was -5 to 5)
    # Color tinting parameters
    tint_probability: float = 0.7  # 70% of color-adjusted images get a tint
    tint_intensity_range: Tuple[float, float] = (0.0325, 0.0975)  # How strong the tint is (reduced by 35%: was 0.05-0.15)
    # Tint colors: red, yellow, blue, green, orange, brown, black, white, purple, pink
    tint_colors: dict = None  # Will be initialized in __post_init__
    
    def __post_init__(self):
        if self.tint_colors is None:
            # Define tint colors in BGR format (OpenCV convention) as tuples
            # Note: white removed since glare already applies white/bright effects
            self.tint_colors = {
                'red': (0, 0, 255),
                'yellow': (0, 255, 255),
                'blue': (255, 0, 0),
                'green': (0, 255, 0),
                'orange': (0, 165, 255),
                'brown': (42, 82, 165),
                'black': (0, 0, 0),
                'purple': (255, 0, 255),
                'pink': (203, 192, 255)
            }


@dataclass
class SleeveConfig:
    """Configuration for card sleeve effects"""
    enabled: bool = True
    probability: float = 0.89  # 89% of images have sleeves (8/9 ratio from original logic)
    glare_probability: float = 0.4  # 40% of sleeved cards have glare
    reflection_intensity: Tuple[float, float] = (0.1, 0.3)
    color_tint_probability: float = 0.3  # 30% have colored sleeves


@dataclass
class DeckBoxConfig:
    """Configuration for deck box/case occlusions"""
    enabled: bool = True
    probability: float = 0.15  # 15% of images have deck boxes
    max_boxes: int = 2  # Maximum number of boxes per image
    size_range: Tuple[int, int] = (100, 250)  # Width/height in pixels
    opacity_range: Tuple[float, float] = (0.7, 1.0)


@dataclass
class OccluderConfig:
    """Configuration for other occluding objects (hands, dice, tokens)"""
    enabled: bool = True
    probability: float = 0.25  # 25% of images have occluders
    max_occluders: int = 3
    size_range: Tuple[int, int] = (50, 150)
    opacity_range: Tuple[float, float] = (0.6, 0.95)


@dataclass
class JitterConfig:
    """Configuration for card position jitter"""
    enabled: bool = True
    probability: float = 0.8  # 80% of cards get position jitter
    translation_range: Tuple[int, int] = (-15, 15)  # Pixels in x/y
    rotation_range: Tuple[float, float] = (-8.0, 8.0)  # Degrees
    scale_range: Tuple[float, float] = (0.95, 1.05)  # Scale factor


@dataclass
class BoundingBoxJitterConfig:
    """Configuration for bounding box adjustments"""
    enabled: bool = True
    probability: float = 0.5  # 50% of boxes get adjustments
    expansion_range: Tuple[float, float] = (-0.05, 0.1)  # Percentage of box size
    shift_range: Tuple[int, int] = (-5, 5)  # Pixels in x/y


@dataclass
class AugmentationConfig:
    """Master configuration for all augmentations"""
    blur: BlurConfig = None
    glare: GlareConfig = None
    shadow: ShadowConfig = None
    color: ColorAdjustmentConfig = None
    sleeves: SleeveConfig = None
    deck_boxes: DeckBoxConfig = None
    occluders: OccluderConfig = None
    jitter: JitterConfig = None
    bbox_jitter: BoundingBoxJitterConfig = None
    
    # Global settings
    seed: int = None  # Set for reproducibility
    
    def __post_init__(self):
        # Initialize with defaults if None
        if self.blur is None:
            self.blur = BlurConfig()
        if self.glare is None:
            self.glare = GlareConfig()
        if self.shadow is None:
            self.shadow = ShadowConfig()
        if self.color is None:
            self.color = ColorAdjustmentConfig()
        if self.sleeves is None:
            self.sleeves = SleeveConfig()
        if self.deck_boxes is None:
            self.deck_boxes = DeckBoxConfig()
        if self.occluders is None:
            self.occluders = OccluderConfig()
        if self.jitter is None:
            self.jitter = JitterConfig()
        if self.bbox_jitter is None:
            self.bbox_jitter = BoundingBoxJitterConfig()
        
        if self.seed is not None:
            random.seed(self.seed)


# Default configuration instance
DEFAULT_CONFIG = AugmentationConfig()


def get_config() -> AugmentationConfig:
    """Get the default augmentation configuration"""
    return DEFAULT_CONFIG


def create_light_config() -> AugmentationConfig:
    """Create a configuration with lighter augmentations (for testing/debugging)"""
    config = AugmentationConfig()
    config.blur.probability = 0.1
    config.glare.probability = 0.1
    config.shadow.probability = 0.1
    config.color.probability = 0.2
    config.sleeves.probability = 0.3
    config.deck_boxes.probability = 0.05
    config.occluders.probability = 0.1
    config.jitter.probability = 0.5
    config.bbox_jitter.probability = 0.3
    return config


def create_heavy_config() -> AugmentationConfig:
    """Create a configuration with heavier augmentations (for robust training)"""
    config = AugmentationConfig()
    config.blur.probability = 0.5
    config.glare.probability = 0.4
    config.shadow.probability = 0.5
    config.color.probability = 0.7
    config.sleeves.probability = 0.8
    config.deck_boxes.probability = 0.25
    config.occluders.probability = 0.4
    config.jitter.probability = 0.9
    config.bbox_jitter.probability = 0.7
    return config


def create_no_augmentation_config() -> AugmentationConfig:
    """Create a configuration with all augmentations disabled (baseline)"""
    config = AugmentationConfig()
    config.blur.enabled = False
    config.glare.enabled = False
    config.color.enabled = False
    config.sleeves.enabled = False
    config.deck_boxes.enabled = False
    config.occluders.enabled = False
    config.jitter.enabled = False
    config.bbox_jitter.enabled = False
    return config


def create_realistic_validation_config() -> AugmentationConfig:
    """
    Create a realistic validation configuration (moderate difficulty).
    
    Designed for validation sets that provide good feedback on real-world performance
    without the extreme adversarial conditions of hard training data.
    
    Key differences from DEFAULT_CONFIG:
    - Reduced blur intensity (30% max vs 50% max)
    - High glare probability (70% - almost all games have some glare) but much lower intensity (max 0.14 vs 0.7)
    - Lighter shadows (0-20% intensity vs 0-80%)
    - 100% sleeves (all competitive games use sleeves)
    - Reduced sleeve glare (30% vs 40%)
    - Deck boxes kept at 15% (hard cases in realistic frequency)
    - Occluders kept at 25% (dice, counters common in real games)
    - Standard jitter and bbox adjustments maintained
    - Color adjustments kept moderate (already reduced by 35% from original)
    """
    config = AugmentationConfig()
    
    # Blur: Reduce max intensity from 50% to 30%
    config.blur.probability_range = (0.15, 0.30)
    
    # Glare: High probability (70%) but much reduced intensity (10x less max)
    config.glare.probability = 0.70  # Almost every game has some glare from lighting
    config.glare.intensity_range = (0.1, 0.2)  # Max reduced from 0.7 to 0.2 (3.5x less)
    
    # Shadow: Reduce max intensity to 20%
    config.shadow.probability = 0.35
    config.shadow.intensity_range = (0.0, 0.20)  # Max 20% vs 80%
    
    # Color: Keep moderate (already well-tuned)
    config.color.probability = 0.5
    
    # Sleeves: Keep default (~60% probability)
    # Sleeve glare: Reduced from 40% to 30%
    config.sleeves.glare_probability = 0.30
    
    # Deck boxes: Keep at 15% (hard cases are realistic)
    config.deck_boxes.probability = 0.15
    
    # Occluders: Keep at 25% (dice/counters are common)
    config.occluders.probability = 0.25
    
    # Jitter: Keep standard for natural card placement variation
    config.jitter.probability = 0.8
    
    # Bbox jitter: Keep standard for robustness
    config.bbox_jitter.probability = 0.5
    
    return config
