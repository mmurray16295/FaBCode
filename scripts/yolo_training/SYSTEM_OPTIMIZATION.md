# System Optimization & Resource Detection

## Overview

Both image generation scripts now include automatic system resource detection and optimization for maximum performance across different hardware configurations.

## Features

### 1. Automatic GPU Detection
- Detects NVIDIA GPUs using PyTorch CUDA
- Falls back gracefully to CPU-only mode if no GPU is available
- Reports available GPU model and count on startup

### 2. CPU Core Detection & Auto-Threading
- Automatically detects available CPU cores
- Calculates optimal thread count based on system size:
  - **96+ cores**: Up to 16 threads per process
  - **32-96 cores**: Up to 12 threads per process  
  - **16-32 cores**: Up to 8 threads per process
  - **<16 cores**: 4+ threads per process

### 3. Parallel Card Processing
- Uses ThreadPoolExecutor to process multiple cards simultaneously
- Each card's operations (load, resize, rotate, blur, composite) run in parallel
- Provides 2-4x speedup on typical workloads

### 4. Manual Override Support
- Use `--threads N` to manually specify thread count
- Useful for fine-tuning performance or avoiding resource contention

## System Detection Output

When scripts start, you'll see:
```
[system] Detected 1 GPU(s): NVIDIA GeForce RTX 5090
[system] Available CPU cores: 120
```

For CPU-only systems:
```
[system] No GPU detected, using CPU only
[system] Available CPU cores: 16
```

## Performance Examples

### Current Runpod Setup (RTX 5090, 120 CPU cores)
- **Auto-detected threads**: 15 (120 cores / 8)
- **Optimal for**: Running training + generation simultaneously
- **Expected generation speed**: ~10-15 images/sec with parallelization

### Desktop Workstation (No GPU, 16 CPU cores)
- **Auto-detected threads**: 8 (16 cores / 2)
- **CPU-only mode**: Fully functional
- **Expected generation speed**: ~3-5 images/sec

### Laptop (No GPU, 8 CPU cores)
- **Auto-detected threads**: 4 (8 cores / 2)
- **CPU-only mode**: Fully functional
- **Expected generation speed**: ~1-2 images/sec

## Usage Examples

### Use Auto-Detected Settings
```bash
python3 scripts/generate_synthetic_playmat_screenshots.py \
    --num-images 1000 \
    --use-popularity-weights \
    --popularity-min 1 \
    --popularity-max 500
```

### Override Thread Count (e.g., to avoid training interference)
```bash
python3 scripts/generate_synthetic_playmat_screenshots.py \
    --num-images 1000 \
    --threads 8 \
    --use-popularity-weights \
    --popularity-min 1 \
    --popularity-max 500
```

### Run on CPU-Only Machine
No special flags needed - scripts automatically detect and adapt:
```bash
python3 scripts/generate_random_playmat.py \
    --num-images 500 \
    --card-dirs data/images/* \
    --use-popularity-weights
```

## Resource Management Tips

### When Training Simultaneously
1. **Monitor GPU usage**: Training should maintain 70-90% GPU utilization
2. **Monitor CPU usage**: If >95%, reduce generation threads with `--threads`
3. **Recommended**: Use 10-12 threads during training, 15-20 when generation-only

### For Maximum Generation Speed
1. Stop all training processes
2. Use auto-detected thread count (default)
3. Run multiple parallel generation processes if generating >10k images

### For Stability on Shared Systems
1. Use `--threads 4` to minimize resource usage
2. Run with `nice -n 19` for lowest priority
3. Monitor with `htop` or `top` to avoid overwhelming system

## Weight Dampening

Both scripts now include popularity weight dampening (WEIGHT_DAMPENING_POWER = 0.92):
- Reduces extreme weight ratios from 23.5x to 18.2x
- Creates more balanced training data distribution
- Top 100 cards: 41.3% of data (vs 43.8% without dampening)
- Ranks 101-500: 58.7% of data (vs 56.2% without dampening)

## Technical Details

### Thread Pool Implementation
- Uses `concurrent.futures.ThreadPoolExecutor`
- Thread count configurable via `SYSTEM_RESOURCES['optimal_threads']`
- Each thread processes one card at a time
- Main thread handles image composition and I/O

### GPU Detection Logic
```python
try:
    import torch
    if torch.cuda.is_available():
        gpu_available = True
        gpu_count = torch.cuda.device_count()
except ImportError:
    # PyTorch not installed, use CPU only
    gpu_available = False
```

### CPU Core Calculation
```python
cpu_count = multiprocessing.cpu_count()

if cpu_count >= 96:
    optimal_threads = min(16, cpu_count // 8)
elif cpu_count >= 32:
    optimal_threads = min(12, cpu_count // 4)
elif cpu_count >= 16:
    optimal_threads = min(8, cpu_count // 2)
else:
    optimal_threads = max(4, cpu_count // 2)
```

## Future Enhancements

Potential optimizations for future versions:
1. **GPU-accelerated image processing**: Use CUDA for resize/rotate operations
2. **Batch processing**: Process multiple images simultaneously on GPU
3. **Mixed precision**: Use FP16 for faster GPU operations
4. **Distributed generation**: Split work across multiple machines
5. **Smart scheduling**: Adjust thread count based on real-time load

## Compatibility

- **Python**: 3.7+
- **Required**: PIL/Pillow, numpy, yaml
- **Optional**: torch (for GPU detection, falls back gracefully if missing)
- **OS**: Linux, macOS, Windows

## Troubleshooting

### "No GPU detected" but I have a GPU
- Ensure PyTorch is installed: `pip install torch`
- Check CUDA drivers: `nvidia-smi`
- GPU detection is informational only - scripts work fine on CPU

### Generation is slow
- Check thread count in startup output
- Try increasing with `--threads` (e.g., `--threads 20`)
- Monitor CPU usage - if <50%, increase threads
- If 100%, system is saturated (optimal)

### Training slowed down after starting generation
- Reduce generation threads: `--threads 8`
- Run generation with lower priority: `nice -n 19 python3 ...`
- Wait for training to complete, then generate

### Out of memory errors
- Reduce thread count: `--threads 4`
- Generate fewer images per run
- Monitor RAM usage: `free -h`
