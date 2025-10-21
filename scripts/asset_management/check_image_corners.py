"""Quick script to check what's actually in the corners of card images."""

from PIL import Image
import numpy as np
import sys

if len(sys.argv) < 2:
    print("Usage: python check_image_corners.py <image_path>")
    sys.exit(1)

img_path = sys.argv[1]
img = Image.open(img_path)

print(f"\n=== Image: {img_path.split('\\')[-1]} ===")
print(f"Mode: {img.mode}")
print(f"Size: {img.size}")

arr = np.array(img)
print(f"Array shape: {arr.shape}")

if len(arr.shape) == 3 and arr.shape[2] == 4:
    print("\n✓ Has alpha channel")
    print(f"\nAlpha channel stats:")
    print(f"  Min: {arr[:,:,3].min()}")
    print(f"  Max: {arr[:,:,3].max()}")
    print(f"  Mean: {arr[:,:,3].mean():.1f}")
    
    # Check if alpha is mostly opaque everywhere
    mostly_opaque = (arr[:,:,3] > 250).sum() / arr[:,:,3].size * 100
    print(f"  % pixels with alpha > 250: {mostly_opaque:.1f}%")
    
    print(f"\nCorner RGB values (ignoring alpha):")
    print(f"  Top-left:     {arr[:10, :10, :3].mean(axis=(0,1))}")
    print(f"  Top-right:    {arr[:10, -10:, :3].mean(axis=(0,1))}")
    print(f"  Bottom-left:  {arr[-10:, :10, :3].mean(axis=(0,1))}")
    print(f"  Bottom-right: {arr[-10:, -10:, :3].mean(axis=(0,1))}")
    
    print(f"\nCorner Alpha values:")
    print(f"  Top-left:     {arr[:10, :10, 3].mean():.1f}")
    print(f"  Top-right:    {arr[:10, -10:, 3].mean():.1f}")
    print(f"  Bottom-left:  {arr[-10:, :10, 3].mean():.1f}")
    print(f"  Bottom-right: {arr[-10:, -10:, 3].mean():.1f}")
    
else:
    print("\n✗ No alpha channel (RGB only)")
    print(f"\nCorner RGB values:")
    print(f"  Top-left:     {arr[:10, :10].mean(axis=(0,1))}")
    print(f"  Top-right:    {arr[:10, -10:].mean(axis=(0,1))}")
    print(f"  Bottom-left:  {arr[-10:, :10].mean(axis=(0,1))}")
    print(f"  Bottom-right: {arr[-10:, -10:].mean(axis=(0,1))}")

# Show actual corner pixels
print(f"\n=== First 5x5 pixel corner samples ===")
print(f"Top-left corner (first 5x5):")
if len(arr.shape) == 3 and arr.shape[2] == 4:
    for i in range(5):
        row = []
        for j in range(5):
            r, g, b, a = arr[i, j]
            row.append(f"({r:3d},{g:3d},{b:3d},A:{a:3d})")
        print("  " + " ".join(row))
else:
    for i in range(5):
        print(f"  {arr[i, :5]}")
