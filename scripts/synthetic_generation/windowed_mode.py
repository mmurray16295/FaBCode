"""
Windowed mode simulation for synthetic training data.

Simulates the game running in windowed mode on a desktop/browser background
by scaling down the game window and placing it on a noisy background that
resembles a YouTube page or desktop environment.
"""

import random
from PIL import Image, ImageDraw
import numpy as np


def generate_noisy_background(width, height, white_bias=0.65):
    """
    Generate a noisy background resembling a YouTube page or desktop.
    
    Creates a scattered pattern of small multi-colored rectangles with
    a bias towards white (simulating thumbnails, text, UI elements, etc.)
    
    Args:
        width: Width of background image
        height: Height of background image
        white_bias: Probability that a rectangle will be white (0.0-1.0)
    
    Returns:
        PIL Image with noisy background pattern
    """
    # Create base canvas
    background = Image.new('RGB', (width, height), color=(240, 240, 240))
    draw = ImageDraw.Draw(background)
    
    # Calculate number of rectangles based on area (roughly 1 rect per 100x100 pixels)
    num_rectangles = int((width * height) / 10000)
    
    for _ in range(num_rectangles):
        # Random rectangle size (small to medium)
        rect_width = random.randint(10, 150)
        rect_height = random.randint(10, 100)
        
        # Random position
        x = random.randint(0, width - rect_width)
        y = random.randint(0, height - rect_height)
        
        # Choose color (65% white, 35% random colors)
        if random.random() < white_bias:
            # White-ish colors (simulating text areas, thumbnails, etc.)
            brightness = random.randint(230, 255)
            color = (brightness, brightness, brightness)
        else:
            # Random colors (simulating UI elements, thumbnails, branding, etc.)
            # Bias towards common YouTube/web colors
            color_choice = random.random()
            if color_choice < 0.2:
                # Red-ish (YouTube brand, alerts)
                color = (random.randint(200, 255), random.randint(0, 100), random.randint(0, 100))
            elif color_choice < 0.3:
                # Blue-ish (links, buttons)
                color = (random.randint(0, 100), random.randint(0, 150), random.randint(200, 255))
            elif color_choice < 0.4:
                # Gray-ish (UI elements)
                gray = random.randint(100, 200)
                color = (gray, gray, gray)
            elif color_choice < 0.5:
                # Black-ish (text)
                darkness = random.randint(0, 80)
                color = (darkness, darkness, darkness)
            else:
                # Truly random (other UI elements)
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        
        # Draw rectangle
        draw.rectangle([x, y, x + rect_width, y + rect_height], fill=color)
    
    return background


def apply_windowed_mode(image, labels, original_width, original_height, 
                       scale_range=(0.45, 0.75), white_bias=0.65):
    """
    Apply windowed mode simulation to a generated playmat image.
    
    Scales down the game window and places it on a noisy background,
    simulating the game running in windowed mode on a desktop/browser.
    
    Args:
        image: PIL Image of the generated playmat
        labels: List of label dictionaries with bounding box info
        original_width: Original image width (for label normalization)
        original_height: Original image height (for label normalization)
        scale_range: Tuple of (min_scale, max_scale) for window size (0.45 = 45%, 0.75 = 75%)
        white_bias: Probability of white rectangles in background (default 0.65)
    
    Returns:
        Tuple of (windowed_image, updated_labels) where labels have adjusted coordinates
    """
    # Choose random scale factor
    scale = random.uniform(scale_range[0], scale_range[1])
    
    # Calculate scaled window size
    scaled_width = int(original_width * scale)
    scaled_height = int(original_height * scale)
    
    # Resize the game window
    scaled_image = image.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
    
    # Generate noisy background at original size
    background = generate_noisy_background(original_width, original_height, white_bias)
    
    # Choose random position for scaled window (ensure it fits)
    max_x = original_width - scaled_width
    max_y = original_height - scaled_height
    paste_x = random.randint(0, max_x) if max_x > 0 else 0
    paste_y = random.randint(0, max_y) if max_y > 0 else 0
    
    # Paste scaled window onto background
    background.paste(scaled_image, (paste_x, paste_y))
    
    # Update all label coordinates
    updated_labels = []
    for label in labels:
        # Labels are in YOLO format: [class_id, center_x, center_y, width, height]
        # All values are normalized (0.0-1.0)
        class_id = label[0]
        center_x_norm = label[1]
        center_y_norm = label[2]
        width_norm = label[3]
        height_norm = label[4]
        
        # Convert to absolute pixel coordinates in original image
        center_x_abs = center_x_norm * original_width
        center_y_abs = center_y_norm * original_height
        width_abs = width_norm * original_width
        height_abs = height_norm * original_height
        
        # Scale down the card
        center_x_scaled = center_x_abs * scale
        center_y_scaled = center_y_abs * scale
        width_scaled = width_abs * scale
        height_scaled = height_abs * scale
        
        # Offset by paste position
        center_x_final = center_x_scaled + paste_x
        center_y_final = center_y_scaled + paste_y
        
        # Convert back to normalized coordinates
        center_x_norm_new = center_x_final / original_width
        center_y_norm_new = center_y_final / original_height
        width_norm_new = width_scaled / original_width
        height_norm_new = height_scaled / original_height
        
        # Clamp to valid range [0, 1]
        center_x_norm_new = max(0.0, min(1.0, center_x_norm_new))
        center_y_norm_new = max(0.0, min(1.0, center_y_norm_new))
        width_norm_new = max(0.0, min(1.0, width_norm_new))
        height_norm_new = max(0.0, min(1.0, height_norm_new))
        
        updated_labels.append([class_id, center_x_norm_new, center_y_norm_new, 
                              width_norm_new, height_norm_new])
    
    return background, updated_labels


def should_apply_windowed_mode(probability=0.15):
    """
    Randomly decide whether to apply windowed mode based on probability.
    
    Args:
        probability: Chance of applying windowed mode (default 0.15 = 15%)
    
    Returns:
        Boolean indicating whether to apply windowed mode
    """
    return random.random() < probability


if __name__ == '__main__':
    # Test the noisy background generator
    print("Testing windowed mode background generation...")
    
    # Generate and save a sample background
    test_bg = generate_noisy_background(1920, 1080, white_bias=0.65)
    test_bg.save('test_windowed_background.png')
    print("✓ Saved test_windowed_background.png")
    
    # Test with a simple image
    from PIL import Image, ImageDraw
    test_img = Image.new('RGB', (1920, 1080), color=(50, 100, 150))
    draw = ImageDraw.Draw(test_img)
    draw.rectangle([860, 490, 1060, 590], fill=(255, 0, 0))
    draw.text((900, 520), "TEST WINDOW", fill=(255, 255, 255))
    
    # Create mock labels for the red rectangle
    # YOLO format: [class_id, center_x_norm, center_y_norm, width_norm, height_norm]
    mock_labels = [
        [0, 0.5, 0.5, 0.104, 0.093]  # Red rectangle in center
    ]
    
    # Apply windowed mode
    windowed_img, updated_labels = apply_windowed_mode(
        test_img, mock_labels, 1920, 1080, 
        scale_range=(0.6, 0.9), white_bias=0.65
    )
    
    windowed_img.save('test_windowed_result.png')
    print("✓ Saved test_windowed_result.png")
    print(f"✓ Original label: {mock_labels[0]}")
    print(f"✓ Updated label:  {updated_labels[0]}")
    print("\nWindowed mode simulation working correctly!")
