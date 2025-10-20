#!/bin/bash
# Quick training monitor script

echo "=== YOLOv11x Training Monitor ==="
echo "Time: $(date)"
echo ""

# Check if process is running
PID=$(ps aux | grep "train_full_yolo11x.py" | grep -v grep | awk '{print $2}' | head -1)
if [ -z "$PID" ]; then
    echo "❌ Training process NOT running"
else
    echo "✅ Training process running (PID: $PID)"
    
    # Process stats
    CPU=$(ps -p $PID -o %cpu= | tr -d ' ')
    MEM=$(ps -p $PID -o %mem= | tr -d ' ')
    TIME=$(ps -p $PID -o time= | tr -d ' ')
    echo "   CPU: ${CPU}% | Memory: ${MEM}% | Runtime: ${TIME}"
fi

echo ""
echo "=== GPU Status ==="
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits | \
    awk -F', ' '{printf "   GPU: %s%% | VRAM: %s%% (%s/%s MB) | Temp: %s°C | Power: %sW\n", $1, $2, $3, $4, $5, $6}'

echo ""
echo "=== Latest Training Output (last 30 lines) ==="
tail -30 /workspace/FaBCode/train_full_yolo11x.log

echo ""
echo "=== Log Stats ==="
echo "   Total lines: $(wc -l < /workspace/FaBCode/train_full_yolo11x.log)"
echo "   Log size: $(du -h /workspace/FaBCode/train_full_yolo11x.log | awk '{print $1}')"

echo ""
echo "=== Saved Checkpoints ==="
if [ -d "/workspace/FaBCode/runs/full_training" ]; then
    find /workspace/FaBCode/runs/full_training -name "*.pt" -type f -exec ls -lh {} \; | awk '{print "   "$9" ("$5")"}'
else
    echo "   No checkpoints yet"
fi

echo ""
echo "=== Quick Commands ==="
echo "   Watch log: tail -f /workspace/FaBCode/train_full_yolo11x.log"
echo "   Watch GPU: watch -n 2 nvidia-smi"
echo "   Kill training: pkill -f train_full_yolo11x.py"
echo "   Resume: cd /workspace/FaBCode && nohup python3 train_full_yolo11x.py > train_full_yolo11x.log 2>&1 &"
