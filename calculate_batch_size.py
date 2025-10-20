#!/usr/bin/env python3
"""
Batch Size Calculator for YOLOv11x Training
Calculates optimal batch size based on GPU memory
"""

import torch
import argparse


def calculate_batch_size(gpu_memory_gb=None, image_size=1280, use_amp=True, verbose=True):
    """
    Calculate recommended batch size for YOLOv11x training
    
    Args:
        gpu_memory_gb: GPU memory in GB (auto-detect if None)
        image_size: Input image size (default: 1280)
        use_amp: Use Automatic Mixed Precision (default: True)
        verbose: Print detailed information
    
    Returns:
        Recommended batch size
    """
    
    # Memory estimates for YOLOv11x
    # These are approximate values based on empirical testing
    MODEL_MEMORY = {
        'yolo11n': 0.5,   # Nano - smallest
        'yolo11s': 1.0,   # Small
        'yolo11m': 2.0,   # Medium
        'yolo11l': 3.0,   # Large
        'yolo11x': 4.0,   # Extra Large - largest
    }
    
    # Memory per image (GB) at different resolutions
    # Format: {image_size: (with_amp, without_amp)}
    IMAGE_MEMORY = {
        640:  (0.3, 0.5),
        1280: (0.7, 1.2),
        1920: (1.5, 2.5),
    }
    
    base_memory = MODEL_MEMORY['yolo11x']
    
    # Get per-image memory
    if image_size in IMAGE_MEMORY:
        per_image_mem = IMAGE_MEMORY[image_size][0 if use_amp else 1]
    else:
        # Estimate for custom size
        scale = (image_size / 1280) ** 2
        per_image_mem = (0.7 if use_amp else 1.2) * scale
    
    # Auto-detect GPU if not specified
    if gpu_memory_gb is None:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu_memory_gb = props.total_memory / (1024**3)
            gpu_name = torch.cuda.get_device_name(0)
            if verbose:
                print(f"✓ Detected GPU: {gpu_name}")
                print(f"✓ Total Memory: {gpu_memory_gb:.2f} GB")
        else:
            raise RuntimeError("No GPU detected! Please specify --gpu-memory manually.")
    
    # Calculate available memory (leave 15% headroom for system/overhead)
    available_memory = gpu_memory_gb * 0.85
    
    # Calculate max batch size
    max_batch = int((available_memory - base_memory) / per_image_mem)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"BATCH SIZE CALCULATION - YOLOv11x")
        print(f"{'='*60}")
        print(f"Image Size: {image_size}x{image_size}")
        print(f"Mixed Precision (AMP): {'Enabled' if use_amp else 'Disabled'}")
        print(f"GPU Memory: {gpu_memory_gb:.2f} GB")
        print(f"Available Memory (85%): {available_memory:.2f} GB")
        print(f"\nMemory Breakdown:")
        print(f"  Base Model: ~{base_memory:.1f} GB")
        print(f"  Per Image: ~{per_image_mem:.2f} GB")
        print(f"\nMaximum Batch Size: {max_batch}")
        print(f"{'='*60}\n")
        
        # Show common batch size recommendations
        print("Recommended Batch Sizes:")
        print(f"{'Batch':>6} | {'Memory':>8} | {'GPU %':>6} | Status")
        print(f"{'-'*6}-+-{'-'*8}-+-{'-'*6}-+-------")
        
        for bs in [4, 8, 12, 16, 20, 24, 32, 40, 48, 64]:
            mem_needed = base_memory + bs * per_image_mem
            gpu_percent = (mem_needed / gpu_memory_gb) * 100
            
            if mem_needed <= available_memory:
                status = "✓ Safe"
                if gpu_percent > 75:
                    status = "✓ Good"
                if gpu_percent > 80:
                    status = "⚠ High"
            else:
                status = "✗ OOM Risk"
            
            print(f"{bs:6d} | {mem_needed:6.2f} GB | {gpu_percent:5.1f}% | {status}")
        
        print(f"\n{'='*60}")
        print("RECOMMENDATIONS:")
        print(f"{'='*60}")
        
        # Recommend based on memory
        if max_batch >= 32:
            recommended = 32
            alt1, alt2 = 24, 16
        elif max_batch >= 24:
            recommended = 24
            alt1, alt2 = 16, 12
        elif max_batch >= 16:
            recommended = 16
            alt1, alt2 = 12, 8
        elif max_batch >= 12:
            recommended = 12
            alt1, alt2 = 8, 4
        elif max_batch >= 8:
            recommended = 8
            alt1, alt2 = 4, 2
        else:
            recommended = 4
            alt1, alt2 = 2, 1
        
        print(f"  Primary: batch={recommended} (best balance)")
        print(f"  Safe: batch={alt1} (more headroom)")
        print(f"  Conservative: batch={alt2} (maximum safety)")
        print(f"\nNote: Start with the safe option and increase if stable.")
        print(f"      Larger batches = better GPU utilization & faster training")
        print(f"      But risk OOM (Out of Memory) errors")
        print(f"{'='*60}\n")
    
    return max_batch


def main():
    parser = argparse.ArgumentParser(description='Calculate optimal batch size for YOLO training')
    parser.add_argument('--gpu-memory', type=float, default=None,
                       help='GPU memory in GB (auto-detect if not specified)')
    parser.add_argument('--image-size', type=int, default=1280,
                       help='Training image size (default: 1280)')
    parser.add_argument('--no-amp', action='store_true',
                       help='Disable mixed precision (not recommended)')
    parser.add_argument('--quiet', action='store_true',
                       help='Only output the recommended batch size')
    
    args = parser.parse_args()
    
    try:
        batch = calculate_batch_size(
            gpu_memory_gb=args.gpu_memory,
            image_size=args.image_size,
            use_amp=not args.no_amp,
            verbose=not args.quiet
        )
        
        if args.quiet:
            print(batch)
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
