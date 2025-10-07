#!/bin/bash
# Parallel synthetic data generation wrapper
# Runs multiple generation processes in parallel

TOTAL_IMAGES=7000
NUM_WORKERS=16  # Using 16 parallel workers
IMAGES_PER_WORKER=$((TOTAL_IMAGES / NUM_WORKERS))

SCRIPT_DIR="/workspace/fab/src/scripts"
CARD_DIRS="/workspace/fab/src/data/images/WTR /workspace/fab/src/data/images/SEA /workspace/fab/src/data/images/HVY /workspace/fab/src/data/images/CRU /workspace/fab/src/data/images/ARC"

echo "Starting parallel generation with $NUM_WORKERS workers"
echo "Each worker will generate $IMAGES_PER_WORKER images"
echo "Total target: $TOTAL_IMAGES images"
echo ""

# Launch workers in parallel
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    SEED=$((42 + i))  # Different seed for each worker
    LOG_FILE="/tmp/worker_${i}.log"
    
    echo "Starting worker $i (seed: $SEED, log: $LOG_FILE)"
    
    nohup python "$SCRIPT_DIR/generate_synthetic_playmat_screenshots.py" \
        --num-images $IMAGES_PER_WORKER \
        --card-dirs $CARD_DIRS \
        --coverage-guided \
        --seed $SEED \
        > "$LOG_FILE" 2>&1 &
    
    # Store PID
    echo $! >> /tmp/worker_pids.txt
done

echo ""
echo "All workers launched! PIDs stored in /tmp/worker_pids.txt"
echo "Monitor progress with: watch -n 5 'find /workspace/fab/src/data/synthetic_2 -name \"*.png\" | wc -l'"
echo "View worker logs: tail -f /tmp/worker_*.log"
