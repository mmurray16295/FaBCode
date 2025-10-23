"""
Parallel Dataset Generation Script
Generates large datasets efficiently using multiple parallel processes
Optimized for high-CPU environments (120 cores available)
"""

import os
import sys
import time
import subprocess
import multiprocessing as mp
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from generation_utils import ensure_background_variations, print_background_usage_stats

# Get repo root (2 levels up from this script)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

def generate_batch(args):
    """Generate a batch of images in a separate process"""
    batch_id, num_images, base_seed, selector_type, validation_mode, split = args
    
    # Build base command with selector type
    cmd = [sys.executable, "scripts/synthetic_generation/Core_Playmat_Generator.py"]
    if selector_type:
        cmd.extend(["--selector", selector_type])
    if validation_mode:
        cmd.append("--validation_mode")
    if split:
        cmd.extend(["--split", split])
    
    # Core_Playmat_Generator generates ONE image per call, so we loop
    successful_images = 0
    for i in range(num_images):
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env={**os.environ, "PYTHONUNBUFFERED": "1"}
        )
        
        if result.returncode == 0:
            successful_images += 1
        else:
            print(f"\n❌ Batch {batch_id} image {i+1}/{num_images} FAILED!")
            print(f"Command: {' '.join(cmd)}")
            print(f"STDERR: {result.stderr[:500]}")  # First 500 chars
            break  # Stop on first failure
    
    return {
        'batch_id': batch_id,
        'success': successful_images == num_images,
        'images_generated': successful_images
    }

def parallel_generate(total_images, num_processes, test_mode=False, selector_type='smooth', validation_mode=False, split='train'):
    """
    Generate images in parallel using multiple processes
    
    Args:
        total_images: Total number of images to generate
        num_processes: Number of parallel processes to run
        test_mode: If True, only generate a small test batch
        selector_type: 'smooth' (even distribution, default) or 'weighted' (popularity-based)
        validation_mode: If True, use realistic validation settings (0% artifacts, 45% overlap)
        split: 'train', 'valid', or 'test' - which split to save images to
    """
    
    print("=" * 80)
    print("PARALLEL DATASET GENERATION")
    print("=" * 80)
    print(f"Total images: {total_images:,}")
    print(f"Parallel processes: {num_processes}")
    print(f"Images per process: {total_images // num_processes}")
    print(f"CPU cores available: {mp.cpu_count()}")
    print("=" * 80)
    
    if test_mode:
        print("\n⚠ TEST MODE: Generating small batch to measure performance")
        total_images = num_processes * 5  # 5 images per process
        print(f"Test batch size: {total_images} images")
    
    # Ensure sufficient background variations exist BEFORE spawning workers
    # This prevents race conditions where multiple workers try to generate backgrounds
    print("\n" + "=" * 80)
    print("BACKGROUND MANAGEMENT")
    print("=" * 80)
    background_dir = str(REPO_ROOT / 'data' / 'synthetic' / 'backgrounds' / 'images')
    num_backgrounds = ensure_background_variations(total_images, background_dir=background_dir, verbose=True)
    print_background_usage_stats(total_images, num_backgrounds)
    print("=" * 80)
    
    start_time = time.time()
    
    # Create work batches - each process generates 1 image at a time
    # This allows us to saturate all CPUs
    work_items = [(i, 1, i * 12345, selector_type, validation_mode, split) for i in range(total_images)]
    
    print(f"\nStarting generation with {num_processes} parallel processes...")
    print("Progress updates every 100 images...")
    
    completed = 0
    successful = 0
    
    with mp.Pool(processes=num_processes) as pool:
        for result in pool.imap_unordered(generate_batch, work_items):
            if result['success']:
                successful += result['images_generated']
            completed += 1
            
            if completed % 100 == 0 or completed == total_images:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = (total_images - completed) / rate if rate > 0 else 0
                
                print(f"  [{completed:,}/{total_images:,}] "
                      f"({100*completed/total_images:.1f}%) | "
                      f"Rate: {rate:.1f} img/s | "
                      f"Remaining: {remaining/60:.1f} min")
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print(f"Total images: {successful:,}/{total_images:,}")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Average rate: {total_images/total_time:.1f} images/second")
    print(f"Average time per image: {total_time/total_images:.2f} seconds")
    print("=" * 80)
    
    return total_time, total_images / total_time

def benchmark_parallelization():
    """
    Test different parallelization levels to find optimal setting
    """
    print("=" * 80)
    print("BENCHMARKING OPTIMAL PARALLELIZATION")
    print("=" * 80)
    print("\nTesting different process counts to find optimal speed...")
    print("Each test generates 20 images (distributed across processes)")
    print()
    
    # Test different process counts
    test_counts = [10, 20, 30, 40, 50, 60, 80, 100, 120]
    results = []
    
    for num_procs in test_counts:
        print(f"\nTesting {num_procs} processes...")
        
        # Generate 20 images total
        work_items = [(i, 1, i * 12345) for i in range(20)]
        
        start = time.time()
        
        with mp.Pool(processes=num_procs) as pool:
            completed = 0
            for result in pool.imap_unordered(generate_batch, work_items):
                completed += 1
        
        elapsed = time.time() - start
        rate = 20 / elapsed
        
        results.append({
            'processes': num_procs,
            'time': elapsed,
            'rate': rate
        })
        
        print(f"  ✓ {num_procs} processes: {elapsed:.1f}s ({rate:.2f} img/s)")
    
    # Find optimal
    best = max(results, key=lambda x: x['rate'])
    
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    
    for r in results:
        marker = " ← BEST" if r == best else ""
        print(f"  {r['processes']:3d} processes: {r['rate']:6.2f} img/s{marker}")
    
    print("\n" + "=" * 80)
    print(f"RECOMMENDATION: Use {best['processes']} parallel processes")
    print(f"Expected rate: {best['rate']:.2f} images/second")
    print(f"Time for 125,000 images: {125000/best['rate']/3600:.1f} hours")
    print("=" * 80)
    
    return best['processes']

def count_existing_images():
    """Count images already generated"""
    synthetic_dir = REPO_ROOT / "data" / "synthetic"
    
    counts = {'train': 0, 'valid': 0, 'test': 0}
    
    for split in ['train', 'valid', 'test']:
        images_dir = synthetic_dir / split / 'images'
        if images_dir.exists():
            counts[split] = len(list(images_dir.glob("*.jpg")))
    
    total = sum(counts.values())
    
    return counts, total

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Parallel dataset generation")
    parser.add_argument("--images", type=int, default=125000, 
                       help="Total images to generate (default: 125000)")
    parser.add_argument("--processes", type=int, default=None,
                       help="Number of parallel processes (auto-detect if not specified)")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run benchmark to find optimal process count")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from existing images")
    parser.add_argument("--selector", type=str, choices=['weighted', 'smooth'], default='smooth',
                       help="Card selector type: 'smooth' (even distribution, default) or 'weighted' (popularity)")
    parser.add_argument("--validation_mode", action="store_true",
                       help="Use realistic validation settings (0%% artifacts, 45%% overlap)")
    parser.add_argument("--split", type=str, choices=['train', 'valid', 'test'], default='train',
                       help="Which split to save images to: 'train', 'valid', or 'test' (default: train)")
    parser.add_argument("--yes", "-y", action="store_true",
                       help="Auto-confirm all prompts (for background execution)")
    
    args = parser.parse_args()
    
    # Show existing images
    existing_counts, existing_total = count_existing_images()
    if existing_total > 0:
        print(f"\nExisting images found: {existing_total:,}")
        print(f"  Train: {existing_counts['train']:,}")
        print(f"  Valid: {existing_counts['valid']:,}")
        print(f"  Test: {existing_counts['test']:,}")
        
        if args.resume:
            # Check only the split we're targeting
            split_existing = existing_counts[args.split]
            remaining = args.images - split_existing
            if remaining <= 0:
                print(f"\n✓ Already have {split_existing:,} images in '{args.split}' split (target: {args.images:,})")
                sys.exit(0)
            print(f"\nResuming: '{args.split}' split has {split_existing:,} images, will generate {remaining:,} more")
            args.images = remaining
        else:
            if not args.yes:
                response = input("\nClear existing images and start fresh? (yes/no): ").strip().lower()
            else:
                response = 'yes'
                print("\nAuto-clearing existing images (--yes flag)")
            
            if response in ['yes', 'y']:
                print("Clearing existing images...")
                for split in ['train', 'valid', 'test']:
                    for subdir in ['images', 'labels']:
                        dir_path = REPO_ROOT / "data" / "synthetic" / split / subdir
                        if dir_path.exists():
                            for f in dir_path.glob("*.jpg"):
                                f.unlink()
                            for f in dir_path.glob("*.txt"):
                                f.unlink()
                print("✓ Cleared")
            else:
                print("Keeping existing images. Use --resume to add more images.")
                sys.exit(0)
    
    # Benchmark if requested
    if args.benchmark:
        print(f"\nUsing selector: {args.selector.upper()}")
        optimal_procs = benchmark_parallelization()
        
        if not args.yes:
            response = input(f"\nProceed with {args.images:,} images using {optimal_procs} processes? (yes/no): ").strip().lower()
        else:
            response = 'yes'
            print(f"\nAuto-proceeding with {args.images:,} images using {optimal_procs} processes (--yes flag)")
        
        if response not in ['yes', 'y']:
            print("Cancelled.")
            sys.exit(0)
        
        args.processes = optimal_procs
    
    # Auto-detect optimal process count if not specified
    if args.processes is None:
        # Default: 10 processes (safe for memory constraints)
        args.processes = 10
        print(f"\nUsing default: {args.processes} processes ({mp.cpu_count()} cores available)")
    
    # Confirm before large generation
    if args.images >= 10000 and not args.yes:
        est_time = args.images / (args.processes * 0.5) / 3600  # Conservative estimate: 0.5 img/s per process
        response = input(f"\nGenerate {args.images:,} images with {args.processes} processes? "
                        f"(Selector: {args.selector.upper()}) "
                        f"(est. {est_time:.1f} hours)\n(yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Cancelled.")
            sys.exit(0)
    elif args.images >= 10000:
        est_time = args.images / (args.processes * 0.5) / 3600
        print(f"\nAuto-starting generation: {args.images:,} images with {args.processes} processes (Selector: {args.selector.upper()}) (est. {est_time:.1f} hours)")
    
    # Generate!
    total_time, rate = parallel_generate(args.images, args.processes, selector_type=args.selector, validation_mode=args.validation_mode, split=args.split)
    
    # Final summary
    final_counts, final_total = count_existing_images()
    print(f"\n✓ FINAL DATASET:")
    print(f"  Train: {final_counts['train']:,}")
    print(f"  Valid: {final_counts['valid']:,}")
    print(f"  Test: {final_counts['test']:,}")
    print(f"  TOTAL: {final_total:,} images")
