#!/bin/bash

echo "========================================"
echo "YOLOv11x Training Monitor"
echo "========================================"
echo ""

# Training directory
TRAIN_DIR="/workspace/FaBCode/runs/full_training/yolo11x_2641classes_20251023_212202"

# Check if process is running
echo "=== Process Status ==="
ps -o pid,stat,etime,cputime,%cpu,%mem,cmd -p 3890074 2>/dev/null || echo "❌ Training process not found!"
echo ""

# Check for results
echo "=== Training Progress ==="
if [ -f "$TRAIN_DIR/results.csv" ]; then
    echo "✅ results.csv exists - training is progressing!"
    echo ""
    echo "Latest epochs:"
    tail -5 "$TRAIN_DIR/results.csv"
    echo ""
    EPOCHS=$(tail -n +2 "$TRAIN_DIR/results.csv" | wc -l)
    echo "Completed epochs: $EPOCHS / 100"
else
    echo "⏳ Epoch 1 still in progress (results.csv not created yet)"
    echo "   This is normal - first epoch can take 45-90 minutes"
fi
echo ""

# Check weights
echo "=== Checkpoints ==="
ls -lh "$TRAIN_DIR/weights/" 2>/dev/null | tail -5
echo ""

# GPU check
echo "=== GPU Utilization ==="
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits
echo ""

echo "========================================"
echo "Training started at: ~21:22"
echo "Current time: $(date +%H:%M)"
echo "========================================"
