#!/bin/bash

echo "============================================"
echo "PHASE 2 TRAINING (500 CLASSES)"
echo "============================================"
echo "Dataset: 33,300 images (30k synthetic + 3.3k random)"
echo "Classes: 500 (ranks 1-500)"
echo "Transfer learning: Phase 1 weights (100 classes)"
echo "Max epochs: 300"
echo "Patience: 30 (early stopping)"
echo "============================================"
echo ""

python3 /root/FaBCode/scripts/train_yolo.py \
    --data /root/FaBCode/data/phase2_500classes/data.yaml \
    --model /root/FaBCode/runs/train/phase1_100classes/weights/best.pt \
    --project /root/FaBCode/runs/train \
    --name phase2_500classes \
    --epochs 300 \
    --batch 16 \
    --imgsz 1280 \
    --patience 30 \
    --device 0 \
    --workers 8 \
    --cache ram

echo ""
echo "============================================"
echo "Training complete!"
echo "Best weights: runs/train/phase2_500classes/weights/best.pt"
echo "============================================"
