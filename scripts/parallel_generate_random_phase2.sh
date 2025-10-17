#!/bin/bash

# ============================================
# PHASE 2 RANDOM PLAYMAT DATA GENERATION (500 CLASSES)
# ============================================
# Generates 3,300 random playmat images with top 500 cards
# Uses 25 parallel processes for faster generation
# ============================================

NUM_PROCESSES=25
TOTAL_IMAGES=3300
IMAGES_PER_PROCESS=132  # 25 x 132 = 3,300

echo "============================================"
echo "PHASE 2 RANDOM PLAYMAT GENERATION (500 CLASSES)"
echo "============================================"
echo "Target classes: 500 (ranks 1-500)"
echo "Total images: $TOTAL_IMAGES"
echo "Parallel processes: $NUM_PROCESSES"
echo "Images per process: $IMAGES_PER_PROCESS"
echo "Output directory: /root/FaBCode/data/phase2_random"
echo "============================================"
echo ""

# Create log directory
mkdir -p /root/FaBCode/logs/phase2_random_generation
rm -f /root/FaBCode/logs/phase2_random_generation/pids.txt

# Start all processes in background
for i in $(seq 0 $((NUM_PROCESSES - 1))); do
    PROCESS_OUTPUT="/root/FaBCode/data/phase2_random_p${i}"
    LOG_FILE="/root/FaBCode/logs/phase2_random_generation/process_${i}.log"
    echo "Starting process $i -> $PROCESS_OUTPUT (log: $LOG_FILE)"
    
    python3 /root/FaBCode/scripts/generate_random_playmat.py \
        --num-images $IMAGES_PER_PROCESS \
        --popularity-min 1 \
        --popularity-max 500 \
        --use-popularity-weights \
        --card-dirs /root/FaBCode/data/images/*/ \
        --output-dir "$PROCESS_OUTPUT" \
        > "$LOG_FILE" 2>&1 &
    
    # Save PID
    echo $! >> /root/FaBCode/logs/phase2_random_generation/pids.txt
done

echo ""
echo "All $NUM_PROCESSES processes started!"
echo "PIDs saved to: logs/phase2_random_generation/pids.txt"
echo ""
echo "Monitor progress:"
echo "  watch -n 5 'find /root/FaBCode/data/phase2_random_p* -name \"*.png\" | wc -l'"
echo ""
echo "Check individual logs:"
echo "  tail -f logs/phase2_random_generation/process_0.log"
echo ""
echo "Stop all processes:"
echo "  kill \$(cat logs/phase2_random_generation/pids.txt)"
echo ""
echo "Estimated time: ~15-20 minutes"
echo "============================================"

# Wait for all processes to complete
echo ""
echo "Waiting for all processes to complete..."
wait

echo ""
echo "============================================"
echo "MERGING OUTPUTS"
echo "============================================"

# Create merged directory
MERGED_DIR="/root/FaBCode/data/phase2_random"
mkdir -p "$MERGED_DIR"

# Merge all process outputs
echo "Merging $NUM_PROCESSES process directories..."
for split in train valid test; do
    echo "  Merging $split split..."
    mkdir -p "$MERGED_DIR/$split/images"
    mkdir -p "$MERGED_DIR/$split/labels"
    
    for i in $(seq 0 $((NUM_PROCESSES - 1))); do
        PROCESS_DIR="/root/FaBCode/data/phase2_random_p${i}"
        if [ -d "$PROCESS_DIR/$split/images" ]; then
            cp "$PROCESS_DIR/$split/images/"* "$MERGED_DIR/$split/images/" 2>/dev/null || true
            cp "$PROCESS_DIR/$split/labels/"* "$MERGED_DIR/$split/labels/" 2>/dev/null || true
        fi
    done
done

# Copy classes.yaml and create data.yaml from first process
if [ -f "/root/FaBCode/data/phase2_random_p0/classes.yaml" ]; then
    cp /root/FaBCode/data/phase2_random_p0/classes.yaml "$MERGED_DIR/"
    echo "Copied classes.yaml"
fi

# Create data.yaml with correct paths
cat > "$MERGED_DIR/data.yaml" << EOF
train: $MERGED_DIR/train/images
val: $MERGED_DIR/valid/images
test: $MERGED_DIR/test/images
nc: 500
EOF

# Get first 500 class names from phase2_500classes (they should be the same)
if [ -f "/root/FaBCode/data/phase2_500classes/data.yaml" ]; then
    # Extract names array from existing data.yaml
    python3 -c "
import yaml
with open('/root/FaBCode/data/phase2_500classes/data.yaml', 'r') as f:
    data = yaml.safe_load(f)
with open('$MERGED_DIR/data.yaml', 'r') as f:
    random_data = yaml.safe_load(f)
random_data['names'] = data['names']
with open('$MERGED_DIR/data.yaml', 'w') as f:
    yaml.dump(random_data, f, default_flow_style=False)
print('Updated data.yaml with class names')
"
fi

echo ""
echo "============================================"
echo "GENERATION COMPLETE!"
echo "============================================"
echo "Total images generated: $(find $MERGED_DIR -name '*.png' | wc -l)"
echo "Train: $(find $MERGED_DIR/train -name '*.png' | wc -l)"
echo "Valid: $(find $MERGED_DIR/valid -name '*.png' | wc -l)"
echo "Test: $(find $MERGED_DIR/test -name '*.png' | wc -l)"
echo ""
echo "Output directory: $MERGED_DIR"
echo ""
echo "Cleaning up temporary process directories..."
rm -rf /root/FaBCode/data/phase2_random_p*
echo "Done!"
echo "============================================"
