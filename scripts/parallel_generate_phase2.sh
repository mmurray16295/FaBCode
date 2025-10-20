#!/bin/bash
# Phase 2: Generate 30,000 images for ranks 1-500 (all mission-critical cards)
# Using 25 parallel processes for maximum CPU utilization

TOTAL_IMAGES=30000
NUM_PROCESSES=25
IMAGES_PER_PROCESS=$((TOTAL_IMAGES / NUM_PROCESSES))
OUTPUT_DIR="/root/FaBCode/data/phase2_500classes"

echo "============================================"
echo "PHASE 2 DATA GENERATION (500 CLASSES)"
echo "============================================"
echo "Target classes: 500 (ranks 1-500)"
echo "Total images: $TOTAL_IMAGES"
echo "Parallel processes: $NUM_PROCESSES"
echo "Images per process: $IMAGES_PER_PROCESS"
echo "Output directory: $OUTPUT_DIR"
echo "============================================"
echo ""

START_TIME=$(date +%s)

mkdir -p /root/FaBCode/logs/phase2_generation
mkdir -p "$OUTPUT_DIR"

# Start all processes in background
for i in $(seq 0 $((NUM_PROCESSES - 1))); do
    PROCESS_OUTPUT="/root/FaBCode/data/phase2_p${i}"
    LOG_FILE="/root/FaBCode/logs/phase2_generation/process_${i}.log"
    echo "Starting process $i -> $PROCESS_OUTPUT (log: $LOG_FILE)"
    
    python3 /root/FaBCode/scripts/generate_synthetic_playmat_screenshots.py \
        --num-images $IMAGES_PER_PROCESS \
        --popularity-min 1 \
        --popularity-max 500 \
        --use-popularity-weights \
        --coverage-guided \
        --card-dirs /root/FaBCode/data/images/*/ \
        --output-dir "$PROCESS_OUTPUT" \
        > "$LOG_FILE" 2>&1 &
    
    # Save PID
    echo $! >> /root/FaBCode/logs/phase2_generation/pids.txt
done

echo ""
echo "All $NUM_PROCESSES processes started!"
echo "PIDs saved to: logs/phase2_generation/pids.txt"
echo ""
echo "Monitor progress:"
echo "  watch -n 5 'find $OUTPUT_DIR -name \"*.jpg\" | wc -l'"
echo ""
echo "Check individual logs:"
echo "  tail -f logs/phase2_generation/process_0.log"
echo ""
echo "Stop all processes:"
echo "  kill \$(cat logs/phase2_generation/pids.txt)"
echo ""
echo "Estimated time: ~2-3 hours"
echo "============================================"
echo ""

# Monitor progress
while true; do
    sleep 60
    COUNT=$(find "$OUTPUT_DIR" -name "*.jpg" 2>/dev/null | wc -l)
    PERCENT=$((COUNT * 100 / TOTAL_IMAGES))
    echo "[$(date '+%H:%M:%S')] Progress: $COUNT/$TOTAL_IMAGES images ($PERCENT%)"
    
    # Check if all processes finished
    RUNNING=0
    for pid in $(cat /root/FaBCode/logs/phase2_generation/pids.txt 2>/dev/null); do
        if kill -0 $pid 2>/dev/null; then
            RUNNING=$((RUNNING + 1))
        fi
    done
    
    if [ "$RUNNING" -eq 0 ]; then
        echo ""
        echo "All processes completed!"
        break
    fi
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))
RATE=$(echo "scale=2; $TOTAL_IMAGES / $ELAPSED" | bc)

echo ""
echo "============================================"
echo "PHASE 2 GENERATION COMPLETE"
echo "============================================"
echo "Total time: ${MINUTES}m ${SECONDS}s"
echo "Images generated: $TOTAL_IMAGES"
echo "Rate: ${RATE} img/s"
echo "============================================"
echo ""

# Merge all outputs into single directory
FINAL_OUTPUT="/root/FaBCode/data/phase2_500classes"
mkdir -p "$FINAL_OUTPUT/train/images" "$FINAL_OUTPUT/train/labels"
mkdir -p "$FINAL_OUTPUT/valid/images" "$FINAL_OUTPUT/valid/labels"
mkdir -p "$FINAL_OUTPUT/test/images" "$FINAL_OUTPUT/test/labels"

echo "Merging outputs to $FINAL_OUTPUT..."
for i in $(seq 0 $((NUM_PROCESSES - 1))); do
    SOURCE="/root/FaBCode/data/phase2_p${i}"
    if [ -d "$SOURCE" ]; then
        echo "  Merging process $i..."
        cp -r "$SOURCE"/train/images/* "$FINAL_OUTPUT/train/images/" 2>/dev/null || true
        cp -r "$SOURCE"/train/labels/* "$FINAL_OUTPUT/train/labels/" 2>/dev/null || true
        cp -r "$SOURCE"/valid/images/* "$FINAL_OUTPUT/valid/images/" 2>/dev/null || true
        cp -r "$SOURCE"/valid/labels/* "$FINAL_OUTPUT/valid/labels/" 2>/dev/null || true
        cp -r "$SOURCE"/test/images/* "$FINAL_OUTPUT/test/images/" 2>/dev/null || true
        cp -r "$SOURCE"/test/labels/* "$FINAL_OUTPUT/test/labels/" 2>/dev/null || true
    fi
done

# Copy metadata files
cp /root/FaBCode/data/phase2_p0/classes.yaml "$FINAL_OUTPUT/" 2>/dev/null || true
cp /root/FaBCode/data/phase2_p0/data.yaml "$FINAL_OUTPUT/" 2>/dev/null || true

echo ""
echo "Merged! Checking final counts..."
TRAIN_COUNT=$(find "$FINAL_OUTPUT/train/images" -name "*.png" 2>/dev/null | wc -l)
VALID_COUNT=$(find "$FINAL_OUTPUT/valid/images" -name "*.png" 2>/dev/null | wc -l)
TEST_COUNT=$(find "$FINAL_OUTPUT/test/images" -name "*.png" 2>/dev/null | wc -l)
TOTAL_COUNT=$((TRAIN_COUNT + VALID_COUNT + TEST_COUNT))

echo "  Train: $TRAIN_COUNT"
echo "  Valid: $VALID_COUNT"
echo "  Test: $TEST_COUNT"
echo "  Total: $TOTAL_COUNT"
echo ""

# Check class count
if [ -f "$FINAL_OUTPUT/classes.yaml" ]; then
    CLASS_COUNT=$(grep -c "^- " "$FINAL_OUTPUT/classes.yaml" 2>/dev/null || echo "unknown")
    echo "Classes: $CLASS_COUNT"
fi

echo ""
echo "Phase 2 data ready for training!"
echo "Location: $FINAL_OUTPUT"
echo ""
