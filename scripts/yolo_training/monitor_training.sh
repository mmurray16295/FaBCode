#!/bin/bash
# Monitor training progress

echo "=== FaB Card Detection Training Monitor ==="
echo ""

# Find the latest training directory
LATEST_TRAIN=$(ls -td runs/detect/train* 2>/dev/null | head -1)

if [ -z "$LATEST_TRAIN" ]; then
    echo "No training runs found in runs/detect/"
    exit 1
fi

echo "Monitoring: $LATEST_TRAIN"
echo ""

# Watch for results file
while true; do
    clear
    echo "=== FaB Card Detection Training Monitor ==="
    echo "Training directory: $LATEST_TRAIN"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Check if results.csv exists
    if [ -f "$LATEST_TRAIN/results.csv" ]; then
        echo "=== Latest Training Metrics ==="
        tail -1 "$LATEST_TRAIN/results.csv" | awk -F',' '{
            printf "Epoch: %s\n", $1
            printf "Box Loss: %s\n", $5
            printf "Class Loss: %s\n", $6
            printf "DFL Loss: %s\n", $7
            printf "mAP50: %s\n", $11
            printf "mAP50-95: %s\n", $12
        }'
        echo ""
    fi
    
    # Check GPU usage
    if command -v nvidia-smi &> /dev/null; then
        echo "=== GPU Usage ==="
        nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | awk -F',' '{
            printf "GPU Util: %s%%\n", $1
            printf "Mem Util: %s%%\n", $2
            printf "Mem Used: %s MB / %s MB\n", $3, $4
            printf "Temp: %s°C\n", $5
        }'
        echo ""
    fi
    
    # Check if training is complete
    if [ -f "$LATEST_TRAIN/weights/best.pt" ]; then
        echo "✅ Training complete! Best weights saved."
        break
    fi
    
    sleep 10
done
