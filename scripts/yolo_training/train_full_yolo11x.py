#!/usr/bin/env python3
"""
Full YOLOv11x Training - Optimized for 125K images (87K train, 25K val, 13K test)
Target: 300 epochs over ~8-10 days
"""

import torch
from ultralytics import YOLO
from datetime import datetime
import os

def main():
    print("="*80)
    print("YOLOv11x FULL TRAINING - 2641 Card Classes")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("="*80)
    
    # Configuration
    data_yaml = '/workspace/FaBCode/data/synthetic/data.yaml'
    model_path = '/workspace/FaBCode/yolo11x.pt'
    cache_dir = '/workspace/FaBCode/cache/dataset_cache'
    project_dir = '/workspace/FaBCode/runs/full_training'
    run_name = f'yolo11x_2641classes_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    
    print(f"\nConfiguration:")
    print(f"  Data: {data_yaml}")
    print(f"  Model: {model_path}")
    print(f"  Cache: {cache_dir}")
    print(f"  Output: {project_dir}/{run_name}")
    print(f"  Epochs: 300")
    print(f"  Batch Size: 24")
    print(f"  Image Size: 1280")
    print(f"  Workers: 12")
    print("="*80 + "\n")
    
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    
    # Initialize model
    print("Loading YOLOv11x model...")
    model = YOLO(model_path)
    print("Model loaded successfully!\n")
    
    # Start training
    print("Starting training...")
    print("NOTE: First epoch will be SLOW (disk caching ~2-3 hours)")
    print("Subsequent epochs will be much faster (~35-45 min)")
    print("="*80 + "\n")
    
    try:
        results = model.train(
            # Data
            data=data_yaml,
            
            # Training duration
            epochs=300,
            patience=50,
            
            # Batch and image settings
            batch=24,  # Optimized for 32GB VRAM
            imgsz=1280,  # Full resolution
            
            # Hardware
            device=0,
            workers=12,  # Increased for better data loading
            
            # Caching - CRITICAL for performance
            cache='disk',  # Cache to disk, not RAM (109GB pod limit)
            
            # Checkpointing and validation
            save=True,
            save_period=10,  # Save checkpoint every 10 epochs
            val=True,  # Run validation
            plots=True,  # Generate plots for monitoring
            
            # Output
            project=project_dir,
            name=run_name,
            exist_ok=False,
            
            # Optimization
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            
            # Augmentation
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.0,
            
            # Advanced
            amp=True,  # Automatic Mixed Precision
            fraction=1.0,  # Use 100% of data
            seed=0,
            deterministic=True,
            close_mosaic=10,  # Disable mosaic in last 10 epochs
            verbose=True,
        )
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE!")
        print("="*80)
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Best weights: {project_dir}/{run_name}/weights/best.pt")
        print(f"Last weights: {project_dir}/{run_name}/weights/last.pt")
        print(f"Results: {project_dir}/{run_name}")
        print("="*80 + "\n")
        
        return results
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print(f"Checkpoint saved to: {project_dir}/{run_name}/weights/last.pt")
        print("Resume training with --resume flag")
        return None
        
    except Exception as e:
        print(f"\n\nERROR during training: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
