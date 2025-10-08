#!/usr/bin/env python3
"""
Train YOLO model for FaB card detection
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description='Train YOLO model for FaB card detection')
    parser.add_argument('--data', type=str, required=True, 
                        help='Path to data.yaml file')
    parser.add_argument('--model', type=str, default='yolo11n.pt',
                        help='Base model to use (default: yolo11n.pt)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size (default: 640)')
    parser.add_argument('--device', type=str, default='0',
                        help='Device to train on (default: 0 for GPU, cpu for CPU)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of workers for data loading (default: 8)')
    parser.add_argument('--project', type=str, default='runs/detect',
                        help='Project directory (default: runs/detect)')
    parser.add_argument('--name', type=str, default='train',
                        help='Experiment name (default: train)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from last checkpoint')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience (default: 50)')
    parser.add_argument('--cache', action='store_true',
                        help='Cache images for faster training')
    
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 60)
    print("FaB Card Detection Training")
    print("=" * 60)
    print(f"Data:       {args.data}")
    print(f"Model:      {args.model}")
    print(f"Epochs:     {args.epochs}")
    print(f"Batch:      {args.batch}")
    print(f"Image size: {args.imgsz}")
    print(f"Device:     {args.device}")
    print(f"Workers:    {args.workers}")
    print(f"Project:    {args.project}")
    print(f"Name:       {args.name}")
    print(f"Resume:     {args.resume}")
    print(f"Patience:   {args.patience}")
    print(f"Cache:      {args.cache}")
    print("=" * 60)
    print()
    
    # Load model
    model = YOLO(args.model)
    
    # Train model
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        resume=args.resume,
        patience=args.patience,
        cache=args.cache,
        plots=True,
        save=True,
        verbose=True,
    )
    
    print()
    print("=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Results saved to: {results.save_dir}")
    print(f"Best weights: {results.save_dir}/weights/best.pt")
    print(f"Last weights: {results.save_dir}/weights/last.pt")
    print("=" * 60)

if __name__ == '__main__':
    main()
