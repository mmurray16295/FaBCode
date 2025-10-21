"""
YOLO11x Production Training Script
Full training on all 2641 FaB card classes
Optimized for maximum accuracy
Runs for ~48 hours on RTX 5090 (~$35)
"""

import os
import sys
import time
from pathlib import Path
from ultralytics import YOLO

def train_yolo11x(
    data_yaml="../../data/synthetic/data.yaml",
    epochs=400,
    imgsz=640,
    batch=16,
    device=0,
    resume=False,
    resume_checkpoint=None
):
    """
    Train YOLO11x on all 2641 FaB card classes
    
    Args:
        data_yaml: Path to data.yaml configuration
        epochs: Training epochs (400 recommended for accuracy)
        imgsz: Image size (640 standard)
        batch: Batch size (16 for RTX 5090 24GB)
        device: GPU device ID
        resume: Resume from checkpoint
        resume_checkpoint: Path to checkpoint to resume from
    """
    
    print("=" * 80)
    print("YOLO11x Production Training - All 2641 Classes")
    print("=" * 80)
    print(f"Model: YOLO11x (56.9M parameters)")
    print(f"Classes: 2641 (all FaB cards)")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch}")
    print(f"Device: cuda:{device}")
    print(f"Resume: {resume}")
    print("=" * 80)
    
    # Estimate time and cost
    time_hours = 48.0  # Estimated for 2641 classes, 400 epochs
    cost_per_hour = 0.69  # RTX 5090
    total_cost = time_hours * cost_per_hour
    
    print(f"\nEstimated time: {time_hours:.0f} hours ({time_hours/24:.1f} days)")
    print(f"Estimated cost: ${total_cost:.2f} (RTX 5090 @ ${cost_per_hour}/hr)")
    print(f"Expected completion: {time.strftime('%Y-%m-%d %I:%M %p', time.localtime(time.time() + time_hours * 3600))}")
    
    # Hardware check
    print("\n" + "=" * 80)
    print("Hardware Requirements:")
    print("=" * 80)
    print("✓ GPU: RTX 5090 (24GB VRAM) or RTX 4090")
    print("✓ RAM: 32GB+ recommended")
    print("✓ Storage: 50GB+ free space")
    print("✓ CUDA: 11.8+ or 12.0+")
    
    # Confirm
    print("\n" + "=" * 80)
    response = input("Proceed with full training? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Training cancelled.")
        return
    
    print("\n" + "=" * 80)
    print("Starting YOLO11x production training...")
    print("=" * 80)
    
    start_time = time.time()
    
    # Load model
    if resume and resume_checkpoint:
        print(f"\n[1/2] Resuming from checkpoint: {resume_checkpoint}")
        model = YOLO(resume_checkpoint)
    else:
        print("\n[1/2] Loading YOLO11x pretrained model...")
        model = YOLO("yolo11x.pt")
    
    # Train
    print("\n[2/2] Starting training...")
    print("=" * 80)
    print("Training Configuration:")
    print("  - Optimizer: AdamW (best for fine details)")
    print("  - Learning rate: 0.001 → 0.00001 (cosine decay)")
    print("  - Weight decay: 0.0005 (regularization)")
    print("  - Warmup: 5 epochs (gradual lr increase)")
    print("  - Close mosaic: 10 epochs (clean images at end)")
    print("  - Patience: 50 epochs (early stopping)")
    print("  - Checkpoints: Every 25 epochs")
    print("=" * 80)
    
    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            
            # Accuracy optimization
            optimizer='AdamW',      # Better than SGD for complex tasks
            lr0=0.001,              # Initial learning rate
            lrf=0.01,               # Final learning rate (0.001 * 0.01 = 0.00001)
            momentum=0.937,         # SGD momentum (for AdamW, less critical)
            weight_decay=0.0005,    # L2 regularization
            warmup_epochs=5,        # Gradual warmup
            warmup_momentum=0.8,    # Warmup momentum
            warmup_bias_lr=0.1,     # Warmup bias lr
            
            # Data augmentation
            hsv_h=0.015,            # Hue augmentation
            hsv_s=0.7,              # Saturation augmentation
            hsv_v=0.4,              # Value augmentation
            degrees=10.0,           # Rotation (±10°)
            translate=0.1,          # Translation (10%)
            scale=0.5,              # Scale (0.5-1.5×)
            shear=0.0,              # No shear (cards are flat)
            perspective=0.0,        # No perspective (already in synthetic)
            flipud=0.0,             # No vertical flip
            fliplr=0.5,             # 50% horizontal flip
            mosaic=1.0,             # Mosaic augmentation
            mixup=0.0,              # No mixup (too many classes)
            copy_paste=0.0,         # No copy-paste (cards don't overlap much)
            close_mosaic=10,        # Disable mosaic last 10 epochs
            
            # Training behavior
            patience=50,            # Early stopping patience
            save=True,              # Save checkpoints
            save_period=25,         # Save every 25 epochs
            cache=False,            # Don't cache (too much memory for 25k images)
            workers=8,              # Data loading workers
            project='runs/detect',
            name='fab_2641_yolo11x',
            exist_ok=True,
            pretrained=True,
            verbose=True,
            seed=42,
            deterministic=False,    # Faster (slightly less reproducible)
            single_cls=False,       # Multi-class detection
            rect=False,             # No rectangular training
            cos_lr=True,            # Cosine learning rate decay
            label_smoothing=0.0,    # No label smoothing
            nbs=64,                 # Nominal batch size
            overlap_mask=True,      # Allow mask overlap
            mask_ratio=4,           # Mask downsample ratio
            dropout=0.0,            # No dropout (already enough regularization)
            val=True,               # Run validation
            plots=True,             # Generate plots
            
            # Performance
            amp=True,               # Automatic Mixed Precision (faster)
            fraction=1.0,           # Use 100% of data
        )
        
        # Summary
        total_time = time.time() - start_time
        actual_cost = (total_time / 3600) * cost_per_hour
        
        print("\n" + "=" * 80)
        print("Training Complete!")
        print("=" * 80)
        print(f"Total time: {total_time/3600:.2f} hours ({total_time/86400:.2f} days)")
        print(f"Actual cost: ${actual_cost:.2f}")
        print(f"\nResults directory: {results.save_dir}")
        print(f"Best weights: {results.save_dir}/weights/best.pt")
        print(f"Last weights: {results.save_dir}/weights/last.pt")
        
        # Check results
        print("\n" + "=" * 80)
        print("Training Metrics:")
        print("=" * 80)
        
        # Read results.csv if available
        results_csv = Path(results.save_dir) / "results.csv"
        if results_csv.exists():
            import pandas as pd
            df = pd.read_csv(results_csv)
            
            # Get final and best metrics
            final_epoch = df.iloc[-1]
            best_map50 = df['metrics/mAP50(B)'].max()
            best_map5095 = df['metrics/mAP50-95(B)'].max()
            
            print(f"Final mAP@0.5: {final_epoch.get('metrics/mAP50(B)', 'N/A'):.3f}")
            print(f"Final mAP@0.5:0.95: {final_epoch.get('metrics/mAP50-95(B)', 'N/A'):.3f}")
            print(f"Final precision: {final_epoch.get('metrics/precision(B)', 'N/A'):.3f}")
            print(f"Final recall: {final_epoch.get('metrics/recall(B)', 'N/A'):.3f}")
            print(f"\nBest mAP@0.5: {best_map50:.3f}")
            print(f"Best mAP@0.5:0.95: {best_map5095:.3f}")
        
        print("\n" + "=" * 80)
        print("Next Steps:")
        print("=" * 80)
        print("1. Test on real cards:")
        print("   python scripts/test_on_real_cards.py")
        print("\n2. Export for deployment:")
        print("   yolo export model=runs/detect/fab_2641_yolo11x/weights/best.pt format=onnx")
        print("\n3. View training plots:")
        print(f"   Open: {results.save_dir}/results.png")
        print("\n4. Monitor with TensorBoard:")
        print(f"   tensorboard --logdir runs/detect")
        print("=" * 80)
        
        # Performance assessment
        if results_csv.exists():
            final_map = final_epoch.get('metrics/mAP50(B)', 0)
            print("\n" + "=" * 80)
            print("Performance Assessment:")
            print("=" * 80)
            if final_map >= 0.95:
                print("🎉 EXCELLENT! (mAP@0.5 ≥ 95%)")
                print("   → Ready for competitive play")
            elif final_map >= 0.90:
                print("✅ VERY GOOD (mAP@0.5 ≥ 90%)")
                print("   → Should work well in most scenarios")
            elif final_map >= 0.85:
                print("⚠️  ACCEPTABLE (mAP@0.5 ≥ 85%)")
                print("   → May need more training data or fine-tuning")
            else:
                print("❌ NEEDS IMPROVEMENT (mAP@0.5 < 85%)")
                print("   → Consider: more data, longer training, or check data quality")
            print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print(f"Checkpoint saved at: runs/detect/fab_2641_yolo11x/weights/last.pt")
        print(f"\nTo resume training, run:")
        print(f"  python scripts/yolo_training/train_yolo11x.py --resume --checkpoint runs/detect/fab_2641_yolo11x/weights/last.pt")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("\nTroubleshooting:")
        print("  - Check CUDA availability: nvidia-smi")
        print("  - Verify data.yaml paths are correct")
        print("  - Ensure sufficient GPU memory (reduce batch size if needed)")
        print("  - Check disk space for checkpoints")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train YOLO11x on all 2641 FaB card classes")
    parser.add_argument("--data", type=str, default="../../data/synthetic/data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=400, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    train_yolo11x(
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        resume=args.resume,
        resume_checkpoint=args.checkpoint
    )
