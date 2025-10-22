"""
Shared utility functions for synthetic image generation.
Used by both test_generation.py and parallel_generate_dataset.py to ensure consistency.
"""

import sys
import subprocess
from pathlib import Path


def count_background_variations(background_dir):
    """
    Count existing background variation files.
    
    Args:
        background_dir: Path to directory containing background images
        
    Returns:
        Number of background image files found
    """
    bg_path = Path(background_dir)
    if not bg_path.exists():
        return 0
    
    jpg_files = list(bg_path.glob('*.jpg'))
    png_files = list(bg_path.glob('*.png'))
    return len(jpg_files) + len(png_files)


def ensure_background_variations(target_images, background_dir='data/synthetic/backgrounds/images', verbose=True):
    """
    Ensure sufficient background variations exist (1:1 ratio with target images).
    Generates additional backgrounds if needed using generate_background_variations.py.
    
    This function is called by both test_generation.py and parallel_generate_dataset.py
    to ensure consistent behavior across all generation modes.
    
    Args:
        target_images: Total number of images to generate
        background_dir: Directory containing background variations
        verbose: Print progress messages
    
    Returns:
        Number of background variations available
    """
    required_backgrounds = target_images  # 1:1 ratio
    existing_backgrounds = count_background_variations(background_dir)
    
    if verbose:
        print(f"\nBackground Variation Check:")
        print(f"  Target images: {target_images:,}")
        print(f"  Required backgrounds (1:1): {required_backgrounds:,}")
        print(f"  Existing backgrounds: {existing_backgrounds:,}")
    
    if existing_backgrounds >= required_backgrounds:
        if verbose:
            print(f"  ✓ Sufficient backgrounds available")
        return existing_backgrounds
    
    # Need to generate more backgrounds
    needed = required_backgrounds - existing_backgrounds
    if verbose:
        print(f"  ⚠ Need {needed:,} more background variations")
        print(f"\nGenerating background variations...")
        print(f"  (This is a one-time cost for this batch size)")
    
    try:
        # Determine script location (works from any caller location)
        script_dir = Path(__file__).parent
        bg_gen_script = script_dir / 'generate_background_variations.py'
        
        if not bg_gen_script.exists():
            if verbose:
                print(f"  ✗ Background generation script not found: {bg_gen_script}")
                print(f"  Continuing with {existing_backgrounds:,} backgrounds...")
            return existing_backgrounds
        
        # Call generate_background_variations.py
        result = subprocess.run([
            sys.executable, 
            str(bg_gen_script),
            '--num-variations', str(needed)
        ], capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            new_count = count_background_variations(background_dir)
            if verbose:
                print(f"  ✓ Generated {needed:,} background variations")
                print(f"  Total backgrounds available: {new_count:,}")
            return new_count
        else:
            if verbose:
                print(f"  ✗ Background generation failed")
                if result.stderr:
                    print(f"  Error: {result.stderr[:200]}")
                print(f"  Continuing with {existing_backgrounds:,} backgrounds...")
            return existing_backgrounds
            
    except subprocess.TimeoutExpired:
        if verbose:
            print(f"  ✗ Background generation timed out (>10 minutes)")
            print(f"  Continuing with {existing_backgrounds:,} backgrounds...")
        return existing_backgrounds
    except Exception as e:
        if verbose:
            print(f"  ✗ Error generating backgrounds: {e}")
            print(f"  Continuing with {existing_backgrounds:,} backgrounds...")
        return existing_backgrounds


def print_background_usage_stats(num_images, num_backgrounds):
    """
    Print statistics about background usage.
    
    Args:
        num_images: Number of images being generated
        num_backgrounds: Number of backgrounds available
    """
    if num_backgrounds > 0:
        usage_per_bg = num_images / num_backgrounds
        print(f"\nBackground Usage:")
        print(f"  Using {num_backgrounds:,} background variations")
        print(f"  Each background will be used ~{usage_per_bg:.1f}x on average")
        
        if usage_per_bg < 1.5:
            print(f"  ⚠ Warning: Low reuse rate. Consider using fewer backgrounds for efficiency.")
        elif usage_per_bg > 3.0:
            print(f"  ⚠ Warning: High reuse rate. Consider generating more backgrounds for diversity.")
