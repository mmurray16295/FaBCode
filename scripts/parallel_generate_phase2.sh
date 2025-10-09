#!/bin/bash
# Phase 2: Generate 20,000 images for ranks 1-500 (all mission-critical cards)
# Using parallelized card processing for speed

TOTAL_IMAGES=20000
NUM_PROCESSES=25
IMAGES_PER_PROCESS=$((TOTAL_IMAGES / NUM_PROCESSES))

echo "============================================"
echo "PHASE 2 DATA GENERATION"
echo "============================================"
echo "Target classes: 500 (ranks 1-500)"
echo "Total images: $TOTAL_IMAGES"
echo "Processes: $NUM_PROCESSES"
echo "Images per process: $IMAGES_PER_PROCESS"
echo "Card processing: 8 threads per process"
echo "============================================"
echo ""

START_TIME=$(date +%s)

mkdir -p /root/FaBCode/logs

# Start all processes in background
for i in $(seq 0 $((NUM_PROCESSES - 1))); do
    OUTPUT_DIR="/root/FaBCode/data/phase2_p${i}"
    echo "Starting process $i..."
    
    python3 /root/FaBCode/scripts/generate_synthetic_playmat_screenshots.py \
        --num-images $IMAGES_PER_PROCESS \
        --popularity-min 1 \
        --popularity-max 500 \
        --use-popularity-weights \
        --card-dirs /root/FaBCode/data/images/*/ \
        --output-dir "$OUTPUT_DIR" \
        > "/root/FaBCode/logs/phase2_p${i}.log" 2>&1 &
done

echo "All $NUM_PROCESSES processes started. Waiting for completion..."
echo "This will take approximately 30-40 minutes with parallelized card processing"
echo ""

# Wait for all background processes to complete
wait

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
