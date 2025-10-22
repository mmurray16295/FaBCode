"""
Image augmentation functions for synthetic playmat generation.
Each function applies a specific augmentation based on configuration parameters.
"""

import cv2
import numpy as np
import random
from typing import Tuple, List, Optional
from augmentation_config import (
    AugmentationConfig,
    BlurConfig,
    GlareConfig,
    ShadowConfig,
    ColorAdjustmentConfig,
    SleeveConfig,
    DeckBoxConfig,
    OccluderConfig,
    JitterConfig,
    BoundingBoxJitterConfig
)


def apply_blur(image: np.ndarray, config: BlurConfig, blur_intensity: float = None) -> np.ndarray:
    """
    Apply Gaussian blur to simulate motion blur or out-of-focus effects.
    
    Args:
        image: Input image (BGR format from OpenCV)
        config: BlurConfig with intensity settings
        blur_intensity: Fixed blur intensity (0.0-1.0). If None, no blur is applied.
        
    Returns:
        Blurred image (or original if blur_intensity is None)
    """
    if not config.enabled or blur_intensity is None or blur_intensity == 0.0:
        return image
    
    # Map blur_intensity (0.0-1.0) to kernel size and sigma
    # Higher intensity = larger kernel and sigma
    min_kernel, max_kernel = config.kernel_size_range
    min_sigma, max_sigma = config.sigma_range
    
    # Calculate kernel size (ensure odd number)
    kernel_range = (max_kernel - min_kernel) // 2
    kernel_size = min_kernel + 2 * int(blur_intensity * kernel_range)
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure odd
    kernel_size = max(min_kernel, min(kernel_size, max_kernel))  # Clamp to range
    
    # Calculate sigma
    sigma = min_sigma + blur_intensity * (max_sigma - min_sigma)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    return blurred


def apply_glare(image: np.ndarray, config: GlareConfig, glare_intensity: float = None, glare_pattern: list = None) -> np.ndarray:
    """
    Apply glare/lighting effects to simulate reflections and light spots.
    
    Args:
        image: Input image (BGR format from OpenCV)
        config: GlareConfig with intensity and spot settings
        glare_intensity: Fixed glare intensity (0.0-1.0). If None, no glare is applied.
        glare_pattern: List of dicts with 'x_ratio', 'y_ratio', 'radius_ratio' for consistent glare positions across cards
        
    Returns:
        Image with glare effects applied (or original if glare_intensity is None)
    """
    if not config.enabled or (glare_intensity is None and glare_pattern is None) or glare_intensity == 0.0:
        return image
    
    # Create a copy to work with
    result = image.copy().astype(np.float32)
    h, w = result.shape[:2]
    
    # Use consistent pattern if provided, otherwise generate random spots
    if glare_pattern is not None:
        # Apply the same glare pattern to this card (positions scale with card dimensions)
        for spot in glare_pattern:
            center_x = int(spot['x_ratio'] * w)
            center_y = int(spot['y_ratio'] * h)
            radius = int(spot['radius_ratio'] * min(w, h))
            
            # Use the card's specific glare intensity directly (already calculated based on distance)
            intensity = glare_intensity
            
            # Create glare mask (Gaussian distribution)
            y_grid, x_grid = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
            
            # Gaussian falloff
            sigma = radius / 2.0
            glare_mask = np.exp(-(dist_from_center**2) / (2 * sigma**2))
            
            # Apply glare (brighten pixels) - vectorized across all channels
            result += glare_mask[:, :, np.newaxis] * intensity * 255
    else:
        # Original random behavior for backwards compatibility
        num_spots = int(glare_intensity * config.num_spots_range[1]) + 1
        num_spots = min(num_spots, config.num_spots_range[1])
        
        for _ in range(num_spots):
            # Random position for glare spot (anywhere on card)
            center_x = random.randint(0, w - 1)
            center_y = random.randint(0, h - 1)
            
            # Radius and intensity scale with glare_intensity
            radius = int(config.radius_range[0] + glare_intensity * (config.radius_range[1] - config.radius_range[0]))
            intensity = config.intensity_range[0] + glare_intensity * (config.intensity_range[1] - config.intensity_range[0])
            
            # Create glare mask (Gaussian distribution)
            y_grid, x_grid = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
            
            # Gaussian falloff
            sigma = radius / 2.0
            glare_mask = np.exp(-(dist_from_center**2) / (2 * sigma**2))
            
            # Apply glare (brighten pixels) - vectorized across all channels
            result += glare_mask[:, :, np.newaxis] * intensity * 255
    
    # Clip values to valid range and convert back to uint8
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


def apply_shadow(image: np.ndarray, config: 'ShadowConfig', shadow_intensity: float = None, 
                 shadow_pattern: list = None, shadow_type: str = None, shadow_direction: str = None) -> np.ndarray:
    """
    Apply shadow/darkness effects to simulate poor lighting conditions, shadows, and dark environments.
    
    Args:
        image: Input image (BGR format from OpenCV)
        config: ShadowConfig with intensity and pattern settings
        shadow_intensity: Fixed shadow intensity (0.0-1.0). If None, no shadow is applied.
                         0.0 = no shadow, 1.0 = completely black
        shadow_pattern: List of dicts with 'x_ratio', 'y_ratio', 'radius_ratio' for consistent shadow positions
        shadow_type: Type of shadow ('spot', 'gradient', 'vignette', 'uniform'). If None, randomly chosen.
        
    Returns:
        Image with shadow effects applied (or original if shadow_intensity is None)
    """
    if not config.enabled or shadow_intensity is None or shadow_intensity == 0.0:
        return image
    
    # Create a copy to work with
    result = image.copy().astype(np.float32)
    h, w = result.shape[:2]
    
    # Choose shadow type if not provided
    if shadow_type is None:
        shadow_type = random.choice(config.shadow_types)
    
    if shadow_type == 'spot':
        # Localized shadow spots (similar to glare but darkening)
        if shadow_pattern is not None:
            # Apply consistent shadow pattern
            for spot in shadow_pattern:
                center_x = int(spot['x_ratio'] * w)
                center_y = int(spot['y_ratio'] * h)
                radius = int(spot['radius_ratio'] * min(w, h))
                
                # Use the card's specific shadow intensity
                intensity = shadow_intensity
                
                # Create shadow mask (Gaussian distribution)
                y_grid, x_grid = np.ogrid[:h, :w]
                dist_from_center = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
                
                # Gaussian falloff
                sigma = radius / 2.0
                shadow_mask = np.exp(-(dist_from_center**2) / (2 * sigma**2))
                
                # Apply shadow (darken pixels) - vectorized across all channels
                result *= (1.0 - shadow_mask[:, :, np.newaxis] * intensity)
        else:
            # Random shadow spots
            num_spots = int(shadow_intensity * config.num_spots_range[1]) + 1
            num_spots = min(num_spots, config.num_spots_range[1])
            
            for _ in range(num_spots):
                # Random position for shadow spot
                center_x = random.randint(0, w - 1)
                center_y = random.randint(0, h - 1)
                
                # Radius scales with shadow_intensity
                radius = int(config.radius_range[0] + shadow_intensity * 
                           (config.radius_range[1] - config.radius_range[0]))
                
                # Create shadow mask (Gaussian distribution)
                y_grid, x_grid = np.ogrid[:h, :w]
                dist_from_center = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
                
                # Gaussian falloff
                sigma = radius / 2.0
                shadow_mask = np.exp(-(dist_from_center**2) / (2 * sigma**2))
                
                # Apply shadow (darken pixels) - vectorized across all channels
                result *= (1.0 - shadow_mask[:, :, np.newaxis] * shadow_intensity)
    
    elif shadow_type == 'gradient':
        # Directional shadow based on shadow source position (consistent across all cards)
        # Direction is determined from shadow source quadrant, passed as shadow_direction parameter
        direction = shadow_direction if shadow_direction else random.choice(['top-left', 'top-right', 'bottom-left', 'bottom-right'])
        
        if direction == 'top-left':
            # Shadow strongest at top-left, fades toward bottom-right
            gradient_x = np.linspace(shadow_intensity, 0, w)
            gradient_y = np.linspace(shadow_intensity, 0, h).reshape(-1, 1)
            shadow_mask = np.sqrt(np.tile(gradient_x, (h, 1))**2 + np.tile(gradient_y, (1, w))**2) / np.sqrt(2)
            shadow_mask = np.clip(shadow_mask, 0, shadow_intensity)
        elif direction == 'top-right':
            # Shadow strongest at top-right, fades toward bottom-left
            gradient_x = np.linspace(0, shadow_intensity, w)
            gradient_y = np.linspace(shadow_intensity, 0, h).reshape(-1, 1)
            shadow_mask = np.sqrt(np.tile(gradient_x, (h, 1))**2 + np.tile(gradient_y, (1, w))**2) / np.sqrt(2)
            shadow_mask = np.clip(shadow_mask, 0, shadow_intensity)
        elif direction == 'bottom-left':
            # Shadow strongest at bottom-left, fades toward top-right
            gradient_x = np.linspace(shadow_intensity, 0, w)
            gradient_y = np.linspace(0, shadow_intensity, h).reshape(-1, 1)
            shadow_mask = np.sqrt(np.tile(gradient_x, (h, 1))**2 + np.tile(gradient_y, (1, w))**2) / np.sqrt(2)
            shadow_mask = np.clip(shadow_mask, 0, shadow_intensity)
        else:  # bottom-right
            # Shadow strongest at bottom-right, fades toward top-left
            gradient_x = np.linspace(0, shadow_intensity, w)
            gradient_y = np.linspace(0, shadow_intensity, h).reshape(-1, 1)
            shadow_mask = np.sqrt(np.tile(gradient_x, (h, 1))**2 + np.tile(gradient_y, (1, w))**2) / np.sqrt(2)
            shadow_mask = np.clip(shadow_mask, 0, shadow_intensity)
        
        # Apply gradient shadow - vectorized across all channels
        result *= (1.0 - shadow_mask[:, :, np.newaxis])
    
    elif shadow_type == 'vignette':
        # Edge darkening (vignette effect)
        # Create distance from center
        center_x, center_y = w / 2, h / 2
        y_grid, x_grid = np.ogrid[:h, :w]
        
        # Normalized distance from center (0 at center, 1 at corners)
        dist_from_center = np.sqrt(((x_grid - center_x) / (w / 2))**2 + 
                                   ((y_grid - center_y) / (h / 2))**2)
        
        # Create vignette mask (darker at edges)
        # Use quadratic falloff for smooth transition
        vignette_mask = np.clip(dist_from_center ** 2, 0, 1) * shadow_intensity
        
        # Apply vignette - vectorized across all channels
        result *= (1.0 - vignette_mask[:, :, np.newaxis])
    
    else:  # uniform
        # Uniform darkening across entire card
        darkening_factor = 1.0 - shadow_intensity
        result *= darkening_factor
    
    # Clip values to valid range and convert back to uint8
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


def apply_color_strip_degradation(image: np.ndarray, strip_height_ratio: float = 0.08, probability: float = 0.85) -> np.ndarray:
    """
    Degrade the color strip at the top of FaB cards to make pitch value less obvious.
    
    This applies multiple degradation techniques to make the color strip harder to rely on:
    1. Strong desaturation to reduce color intensity
    2. Random color shift/tint to make the color less reliable
    3. Brightness variation to simulate lighting/glare
    4. Noise addition for realism
    5. Circular degradation over top-left pitch symbol
    
    Note: Blur is NOT applied here as it creates an artificial rectangular pattern.
    Global blur from the main augmentation pipeline will handle overall blur naturally.
    
    Args:
        image: Input image (BGR format)
        strip_height_ratio: Height of the color strip as ratio of image height (default 0.08 = 8%)
        probability: Probability of applying degradation (default 0.85 = 85%)
        
    Returns:
        Image with degraded color strip and pitch symbol
    """
    if random.random() > probability:
        return image
    
    height, width = image.shape[:2]
    strip_height = int(height * strip_height_ratio)
    
    # Create a copy to work with
    result = image.copy()
    strip = result[:strip_height, :].copy()
    
    # Convert to HSV for color manipulation
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # 1. STRONG desaturation (reduce color intensity) - PRIMARY degradation
    # Much more aggressive - often turns colors nearly grey
    desaturation = random.uniform(0.1, 0.6)  # Reduce saturation by 40-90%
    hsv[:, :, 1] *= desaturation
    
    # 2. Add random color tint/shift to make color less reliable
    # Higher probability and wider range
    hue_shift = 0  # Default no shift
    if random.random() < 0.85:  # 85% chance (up from 70%)
        hue_shift = random.uniform(-30, 30)  # Much larger hue shift (was ±20)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    
    # 3. Random brightness variation (simulate overexposure or shadow)
    # Wider range for more extreme variations
    brightness_adjust = random.uniform(0.5, 1.6)  # Wider range (was 0.6-1.4)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_adjust, 0, 255)
    
    # Convert back to BGR
    strip = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # 4. Add noise for realism (higher probability and intensity)
    if random.random() < 0.6:  # 60% chance (up from 50%)
        noise = np.random.normal(0, 15, strip.shape).astype(np.float32)  # More noise (was 12)
        strip = np.clip(strip.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    # 5. Occasionally add a color overlay to further confuse the color
    if random.random() < 0.4:  # 40% chance - new technique
        # Random color overlay
        overlay_color = np.random.randint(0, 256, 3).astype(np.float32)
        overlay_strength = random.uniform(0.1, 0.3)
        strip = strip.astype(np.float32)
        strip = strip * (1 - overlay_strength) + overlay_color * overlay_strength
        strip = np.clip(strip, 0, 255).astype(np.uint8)
    
    # Apply the degraded strip back to the image with gradient blending
    # Use a smooth gradient to avoid hard edges
    blend_height = min(25, strip_height // 2)  # Even larger blend zone (was 20px)
    
    # Create a gradient mask that goes from 1.0 at top to 0.0 at blend_height
    for i in range(blend_height):
        # Smooth cosine falloff for more natural blending
        alpha = 0.5 * (1 + np.cos(np.pi * i / blend_height))
        row_idx = strip_height - blend_height + i
        if row_idx < strip_height and row_idx >= 0:
            result[row_idx, :] = (strip[row_idx, :] * alpha + result[row_idx, :] * (1 - alpha)).astype(np.uint8)
    
    # Apply the non-blended part
    result[:strip_height - blend_height, :] = strip[:strip_height - blend_height, :]
    
    # 6. Apply circular degradation to top-left corner (pitch symbol area)
    # Pitch symbol is typically in top-left corner, roughly 10-12% of card width/height
    circle_center_x = int(width * 0.10)  # 10% from left edge
    circle_center_y = int(height * 0.10)  # 10% from top edge
    circle_radius = int(min(width, height) * 0.12)  # 12% of smaller dimension
    
    # Create circular region with same degradation parameters
    # Extract circular region
    y_min = max(0, circle_center_y - circle_radius)
    y_max = min(height, circle_center_y + circle_radius)
    x_min = max(0, circle_center_x - circle_radius)
    x_max = min(width, circle_center_x + circle_radius)
    
    # Create a mask for the circular region with smooth falloff
    y_coords, x_coords = np.ogrid[y_min:y_max, x_min:x_max]
    distances = np.sqrt((x_coords - circle_center_x)**2 + (y_coords - circle_center_y)**2)
    
    # Smooth falloff mask (fully degraded at center, fades to 0 at radius)
    circle_mask = np.clip(1.0 - (distances / circle_radius), 0, 1).astype(np.float32)
    # Apply cosine falloff for smoother transition
    circle_mask = 0.5 * (1 + np.cos(np.pi * (1 - circle_mask)))
    
    # Extract and degrade the circular region
    circle_region = result[y_min:y_max, x_min:x_max].copy()
    circle_hsv = cv2.cvtColor(circle_region, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # Apply same degradation as strip (use same parameters for consistency)
    circle_hsv[:, :, 1] *= desaturation
    if hue_shift != 0:  # Apply hue shift if it was applied to the strip
        circle_hsv[:, :, 0] = (circle_hsv[:, :, 0] + hue_shift) % 180
    circle_hsv[:, :, 2] = np.clip(circle_hsv[:, :, 2] * brightness_adjust, 0, 255)
    
    circle_degraded = cv2.cvtColor(circle_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # Apply noise to circle
    if random.random() < 0.6:
        noise = np.random.normal(0, 15, circle_degraded.shape).astype(np.float32)
        circle_degraded = np.clip(circle_degraded.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    # Blend the degraded circular region back using the mask - vectorized across all channels
    circle_mask_3d = circle_mask[:, :, np.newaxis]
    result[y_min:y_max, x_min:x_max] = (
        circle_degraded * circle_mask_3d +
        result[y_min:y_max, x_min:x_max] * (1 - circle_mask_3d)
    ).astype(np.uint8)
    
    return result
    
    # Apply the non-blended part
    result[:strip_height - blend_height, :] = strip[:strip_height - blend_height, :]
    
    return result


def apply_color_adjustment(image: np.ndarray, config: ColorAdjustmentConfig, 
                          brightness: float = None, contrast: float = None,
                          saturation: float = None, hue_shift: int = None,
                          tint_color: str = None, tint_intensity: float = None) -> np.ndarray:
    """
    Apply color adjustments to simulate different lighting conditions and camera settings.
    Reduces color saturation and normalizes brightness to make pitch circles harder to distinguish.
    
    Args:
        image: Input image (BGR format from OpenCV)
        config: ColorAdjustmentConfig with adjustment ranges
        brightness: Brightness multiplier (1.0 = no change). If None, randomly chosen from config.
        contrast: Contrast multiplier (1.0 = no change). If None, randomly chosen from config.
        saturation: Saturation multiplier (1.0 = no change, 0.0 = grayscale). If None, randomly chosen from config.
        hue_shift: Hue shift in degrees (-180 to 180). If None, randomly chosen from config.
        tint_color: Color name for tinting (e.g., 'red', 'blue'). If None, randomly chosen.
        tint_intensity: Strength of tint (0.0-1.0). If None, randomly chosen from config.
        
    Returns:
        Color-adjusted image
    """
    if not config.enabled:
        return image
    
    # If no parameters provided, use random values from config ranges
    if brightness is None:
        brightness = random.uniform(*config.brightness_range)
    if contrast is None:
        contrast = random.uniform(*config.contrast_range)
    if saturation is None:
        saturation = random.uniform(*config.saturation_range)
    if hue_shift is None:
        hue_shift = random.randint(*config.hue_shift_range)
    
    result = image.astype(np.float32)
    
    # Apply brightness adjustment (simple multiplication)
    result = result * brightness
    
    # Apply contrast adjustment (around mid-point 127.5)
    result = (result - 127.5) * contrast + 127.5
    
    # Convert to HSV for saturation and hue adjustments
    result = np.clip(result, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # Apply saturation adjustment
    # Lower saturation makes colors more similar (red/black circles harder to distinguish)
    hsv[:, :, 1] = hsv[:, :, 1] * saturation
    
    # Apply hue shift
    if hue_shift != 0:
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    
    # Clip and convert back to BGR
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)  # Ensure saturation is in valid range
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32)
    
    # Apply color tint if specified or randomly with probability
    apply_tint = tint_color is not None or (tint_intensity is not None) or (random.random() < config.tint_probability)
    if apply_tint:
        if tint_color is None:
            tint_color = random.choice(list(config.tint_colors.keys()))
        if tint_intensity is None:
            tint_intensity = random.uniform(*config.tint_intensity_range)
        
        # Get the tint color and convert to numpy array
        tint = np.array(config.tint_colors[tint_color], dtype=np.float32)
        
        # Blend the tint with the image
        # Higher tint_intensity = more tint influence
        result = result * (1.0 - tint_intensity) + tint * tint_intensity
    
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


# Test function to verify blur works correctly
def _test_blur():
    """Test the blur augmentation with a sample image"""
    # Create a test image (500x500 with some patterns)
    test_image = np.zeros((500, 500, 3), dtype=np.uint8)
    cv2.rectangle(test_image, (100, 100), (400, 400), (255, 255, 255), -1)
    cv2.circle(test_image, (250, 250), 50, (0, 0, 255), -1)
    
    # Apply blur with default config
    from augmentation_config import BlurConfig
    config = BlurConfig()
    
    # Test multiple times to see probabilistic behavior
    print(f"Testing blur with probability={config.probability}")
    applied_count = 0
    for i in range(10):
        result = apply_blur(test_image.copy(), config)
        if not np.array_equal(result, test_image):
            applied_count += 1
    
    print(f"Blur applied {applied_count}/10 times (expected ~{int(config.probability * 10)})")
    
    # Force apply blur for visual check
    config.probability = 1.0
    blurred = apply_blur(test_image, config)
    
    # Save for visual inspection
    import os
    test_dir = '../data/Background Perfecting/test'
    os.makedirs(test_dir, exist_ok=True)
    cv2.imwrite(os.path.join(test_dir, 'test_blur_original.png'), test_image)
    cv2.imwrite(os.path.join(test_dir, 'test_blur_applied.png'), blurred)
    print(f"Saved images to {test_dir}/")


if __name__ == "__main__":
    _test_blur()
