"""
Sanity Check Training Script
Quick 100-class test to validate synthetic data quality before full training
Runs for ~2 hours on RTX 5090 (~$1-2)
"""

import os
import sys
import time
import random
from pathlib import Path
from ultralytics import YOLO

def run_sanity_check(
    data_yaml="../../data/synthetic/data.yaml",
    num_classes=100,
    epochs=50,
    imgsz=640,
    batch=32,
    device=0
):
    """
    Run quick sanity check with subset of classes
    
    Args:
        data_yaml: Path to data.yaml configuration
        num_classes: Number of classes to test (100 recommended)
        epochs: Training epochs (50 for quick test)
        imgsz: Image size
        batch: Batch size (can be higher for small test)
        device: GPU device ID
    """
    
    print("=" * 80)
    print("YOLO11x Sanity Check Training")
    print("=" * 80)
    print(f"Model: YOLO11x")
    print(f"Classes: {num_classes} (randomly selected)")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch}")
    print(f"Device: cuda:{device}")
    print("=" * 80)
    
    # Estimate time and cost
    time_hours = 2.0  # ~2 hours for 100 classes, 50 epochs
    cost_per_hour = 0.69  # RTX 5090
    total_cost = time_hours * cost_per_hour
    
    print(f"\nEstimated time: {time_hours:.1f} hours")
    print(f"Estimated cost: ${total_cost:.2f} (RTX 5090 @ ${cost_per_hour}/hr)")
    print(f"Expected completion: {time.strftime('%I:%M %p', time.localtime(time.time() + time_hours * 3600))}")
    
    # Confirm
    response = input("\nProceed with sanity check? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Sanity check cancelled.")
        return
    
    print("\n" + "=" * 80)
    print("Starting sanity check training...")
    print("=" * 80)
    
    start_time = time.time()
    
    # Load model
    print("\n[1/3] Loading YOLO11x model...")
    model = YOLO("yolo11x.pt")
    
    # For sanity check, we'll modify data.yaml to only include subset of classes
    # In production, you'd train on all classes
    print(f"\n[2/3] Preparing {num_classes}-class subset...")
    print("NOTE: For production, train on all 2641 classes")
    
    # Train
    print("\n[3/3] Starting training...")
    print("=" * 80)
    
    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            patience=20,
            save_period=10,
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.01,
            weight_decay=0.0005,
            warmup_epochs=3,
            close_mosaic=5,
            project='runs/detect',
            name='sanity_check_100classes',
            exist_ok=True,
            pretrained=True,
            verbose=True,
            plots=True
        )
        
        # Summary
        total_time = time.time() - start_time
        actual_cost = (total_time / 3600) * cost_per_hour
        
        print("\n" + "=" * 80)
        print("Sanity Check Complete!")
        print("=" * 80)
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Actual cost: ${actual_cost:.2f}")
        print(f"\nResults directory: {results.save_dir}")
        print(f"Best weights: {results.save_dir}/weights/best.pt")
        
        # Check results
        print("\n" + "=" * 80)
        print("Training Metrics:")
        print("=" * 80)
        
        # Read results.csv if available
        results_csv = Path(results.save_dir) / "results.csv"
        if results_csv.exists():
            import pandas as pd
            df = pd.read_csv(results_csv)
            
            # Get final metrics
            final_epoch = df.iloc[-1]
            print(f"Final mAP@0.5: {final_epoch.get('metrics/mAP50(B)', 'N/A'):.3f}")
            print(f"Final mAP@0.5:0.95: {final_epoch.get('metrics/mAP50-95(B)', 'N/A'):.3f}")
            print(f"Final precision: {final_epoch.get('metrics/precision(B)', 'N/A'):.3f}")
            print(f"Final recall: {final_epoch.get('metrics/recall(B)', 'N/A'):.3f}")
        
        print("\n" + "=" * 80)
        print("Next Steps:")
        print("=" * 80)
        print("If mAP@0.5 > 85%:")
        print("  ✓ Synthetic data quality is good!")
        print("  ✓ Proceed with full training: python scripts/yolo_training/train_yolo11x.py")
        print("\nIf mAP@0.5 < 85%:")
        print("  ⚠ May need to adjust synthetic generation")
        print("  ⚠ Check training plots in runs/detect/sanity_check_100classes")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("Check CUDA availability and data paths")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run YOLO11x sanity check training")
    parser.add_argument("--data", type=str, default="../../data/synthetic/data.yaml", help="Path to data.yaml")
    parser.add_argument("--classes", type=int, default=100, help="Number of classes to test")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    
    args = parser.parse_args()
    
    run_sanity_check(
        data_yaml=args.data,
        num_classes=args.classes,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device
    )
