#!/bin/bash
# Parallel synthetic data generation script
# Generates 5238 images using 25 processes (209 images each)

TOTAL_IMAGES=5238
NUM_PROCESSES=25
IMAGES_PER_PROCESS=$((TOTAL_IMAGES / NUM_PROCESSES))

echo "Starting parallel generation: $NUM_PROCESSES processes, $IMAGES_PER_PROCESS images each"
mkdir -p /root/FaBCode/logs

# Start all processes in background
for i in $(seq 0 $((NUM_PROCESSES - 1))); do
    OUTPUT_DIR="/root/FaBCode/data/synthetic_p${i}"
    echo "Starting process $i -> $OUTPUT_DIR"
    
    python3 /root/FaBCode/scripts/generate_synthetic_playmat_screenshots.py \
        --num-images $IMAGES_PER_PROCESS \
        --popularity-min 1 \
        --popularity-max 100 \
        --use-popularity-weights \
        --card-dirs /root/FaBCode/data/images/*/ \
        --output-dir "$OUTPUT_DIR" \
        > "/root/FaBCode/logs/gen_synthetic_p${i}.log" 2>&1 &
done

echo "All $NUM_PROCESSES processes started. Waiting for completion..."

# Wait for all background processes to complete
wait

echo "All processes complete! Merging outputs..."

# Merge all outputs into single directory
FINAL_OUTPUT="/root/FaBCode/data/synthetic"
mkdir -p "$FINAL_OUTPUT/train/images" "$FINAL_OUTPUT/train/labels"
mkdir -p "$FINAL_OUTPUT/valid/images" "$FINAL_OUTPUT/valid/labels"
mkdir -p "$FINAL_OUTPUT/test/images" "$FINAL_OUTPUT/test/labels"

for i in $(seq 0 $((NUM_PROCESSES - 1))); do
    SOURCE="/root/FaBCode/data/synthetic_p${i}"
    if [ -d "$SOURCE" ]; then
        echo "Merging process $i output..."
        cp -r "$SOURCE"/train/images/* "$FINAL_OUTPUT/train/images/" 2>/dev/null || true
        cp -r "$SOURCE"/train/labels/* "$FINAL_OUTPUT/train/labels/" 2>/dev/null || true
        cp -r "$SOURCE"/valid/images/* "$FINAL_OUTPUT/valid/images/" 2>/dev/null || true
        cp -r "$SOURCE"/valid/labels/* "$FINAL_OUTPUT/valid/labels/" 2>/dev/null || true
        cp -r "$SOURCE"/test/images/* "$FINAL_OUTPUT/test/images/" 2>/dev/null || true
        cp -r "$SOURCE"/test/labels/* "$FINAL_OUTPUT/test/labels/" 2>/dev/null || true
        
        # Copy the classes.yaml from first process that has it
        if [ ! -f "$FINAL_OUTPUT/classes.yaml" ] && [ -f "$SOURCE/classes.yaml" ]; then
            cp "$SOURCE/classes.yaml" "$FINAL_OUTPUT/classes.yaml"
        fi
        # Copy data.yaml from first process that has it  
        if [ ! -f "$FINAL_OUTPUT/data.yaml" ] && [ -f "$SOURCE/data.yaml" ]; then
            cp "$SOURCE/data.yaml" "$FINAL_OUTPUT/data.yaml"
        fi
    fi
done

echo "Merge complete! Cleaning up temporary directories..."
rm -rf /root/FaBCode/data/synthetic_p*

echo "Done! Final output in $FINAL_OUTPUT"
echo "Image counts per split:"
find "$FINAL_OUTPUT/train/images" -type f | wc -l | xargs echo "  Train:"
find "$FINAL_OUTPUT/valid/images" -type f | wc -l | xargs echo "  Valid:"
find "$FINAL_OUTPUT/test/images" -type f | wc -l | xargs echo "  Test:"
