#!/bin/bash
# Benchmark parallel synthetic data generation with parallelized card processing
# Generates 2500 images using 25 processes (100 images each)
# Using ranks 1-200 to ensure complete coverage of top 100 + next 100

TOTAL_IMAGES=2500
NUM_PROCESSES=25
IMAGES_PER_PROCESS=$((TOTAL_IMAGES / NUM_PROCESSES))

echo "============================================"
echo "PARALLEL GENERATION BENCHMARK"
echo "============================================"
echo "Processes: $NUM_PROCESSES"
echo "Images per process: $IMAGES_PER_PROCESS"
echo "Total images: $TOTAL_IMAGES"
echo "Rank range: 1-200 (ensures all top 100 + next 100)"
echo "Card processing: 8 threads per process (NEW)"
echo "============================================"
echo ""

START_TIME=$(date +%s)

mkdir -p /root/FaBCode/logs

# Start all processes in background
for i in $(seq 0 $((NUM_PROCESSES - 1))); do
    OUTPUT_DIR="/root/FaBCode/data/benchmark_synth_p${i}"
    
    python3 /root/FaBCode/scripts/generate_synthetic_playmat_screenshots.py \
        --num-images $IMAGES_PER_PROCESS \
        --popularity-min 1 \
        --popularity-max 200 \
        --use-popularity-weights \
        --card-dirs /root/FaBCode/data/images/*/ \
        --output-dir "$OUTPUT_DIR" \
        > "/root/FaBCode/logs/benchmark_synth_p${i}.log" 2>&1 &
done

echo "All $NUM_PROCESSES processes started. Waiting for completion..."
echo ""

# Wait for all background processes to complete
wait

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
RATE=$(echo "scale=2; $TOTAL_IMAGES / $ELAPSED" | bc)

echo ""
echo "============================================"
echo "BENCHMARK RESULTS"
echo "============================================"
echo "Total time: ${ELAPSED}s"
echo "Images generated: $TOTAL_IMAGES"
echo "Rate: ${RATE} img/s"
echo "============================================"
echo ""

# Merge outputs
FINAL_OUTPUT="/root/FaBCode/data/benchmark_synthetic"
mkdir -p "$FINAL_OUTPUT/train/images" "$FINAL_OUTPUT/train/labels"
mkdir -p "$FINAL_OUTPUT/valid/images" "$FINAL_OUTPUT/valid/labels"
mkdir -p "$FINAL_OUTPUT/test/images" "$FINAL_OUTPUT/test/labels"

echo "Merging outputs..."
for i in $(seq 0 $((NUM_PROCESSES - 1))); do
    SOURCE="/root/FaBCode/data/benchmark_synth_p${i}"
    if [ -d "$SOURCE" ]; then
        cp -r "$SOURCE"/train/images/* "$FINAL_OUTPUT/train/images/" 2>/dev/null || true
        cp -r "$SOURCE"/train/labels/* "$FINAL_OUTPUT/train/labels/" 2>/dev/null || true
        cp -r "$SOURCE"/valid/images/* "$FINAL_OUTPUT/valid/images/" 2>/dev/null || true
        cp -r "$SOURCE"/valid/labels/* "$FINAL_OUTPUT/valid/labels/" 2>/dev/null || true
        cp -r "$SOURCE"/test/images/* "$FINAL_OUTPUT/test/images/" 2>/dev/null || true
        cp -r "$SOURCE"/test/labels/* "$FINAL_OUTPUT/test/labels/" 2>/dev/null || true
    fi
done

# Copy metadata
cp /root/FaBCode/data/benchmark_synth_p0/classes.yaml "$FINAL_OUTPUT/" 2>/dev/null || true
cp /root/FaBCode/data/benchmark_synth_p0/data.yaml "$FINAL_OUTPUT/" 2>/dev/null || true

echo "Merged to $FINAL_OUTPUT"
echo ""
echo "Previous baseline (sequential card processing): 9.55 img/s"
echo "New rate (parallel card processing): ${RATE} img/s"

SPEEDUP=$(echo "scale=2; $RATE / 9.55" | bc)
echo "Speedup: ${SPEEDUP}x"
echo ""
