#!/bin/bash
# Benchmark different process counts to find optimal parallelization

echo "=========================================="
echo "Parallel Generation Benchmark"
echo "=========================================="
echo ""

# Test configurations: [num_processes, test_duration_seconds]
CONFIGS=(
    "10:20"
    "15:20"
    "20:20"
    "25:20"
)

RESULTS_FILE="/root/FaBCode/benchmark_results.txt"
echo "Benchmark Results - $(date)" > "$RESULTS_FILE"
echo "========================================" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

for config in "${CONFIGS[@]}"; do
    IFS=':' read -r NUM_PROCESSES TEST_DURATION <<< "$config"
    IMAGES_PER_PROCESS=100  # Small test batch
    
    echo ""
    echo "=========================================="
    echo "Testing with $NUM_PROCESSES processes"
    echo "=========================================="
    
    # Clean up previous test
    rm -rf /root/FaBCode/data/synthetic_p*
    
    # Start processes
    for i in $(seq 0 $((NUM_PROCESSES - 1))); do
        OUTPUT_DIR="/root/FaBCode/data/synthetic_p${i}"
        
        python3 /root/FaBCode/scripts/generate_synthetic_playmat_screenshots.py \
            --num-images $IMAGES_PER_PROCESS \
            --popularity-min 1 \
            --popularity-max 100 \
            --use-popularity-weights \
            --card-dirs /root/FaBCode/data/images/*/ \
            --output-dir "$OUTPUT_DIR" \
            > "/root/FaBCode/logs/benchmark_p${i}.log" 2>&1 &
    done
    
    echo "Started $NUM_PROCESSES processes, monitoring for ${TEST_DURATION}s..."
    sleep 3  # Give processes time to start
    
    # Measure generation rate
    python3 <<EOF
import glob
import time

measurements = []
for i in range(5):  # Monitor for ${TEST_DURATION}s
    train_count = len(glob.glob('/root/FaBCode/data/synthetic_p*/train/images/*.png'))
    valid_count = len(glob.glob('/root/FaBCode/data/synthetic_p*/valid/images/*.png'))
    test_count = len(glob.glob('/root/FaBCode/data/synthetic_p*/test/images/*.png'))
    total = train_count + valid_count + test_count
    
    current_time = time.time()
    measurements.append((current_time, total))
    
    if i == 0:
        print(f"  Starting count: {total} images")
    else:
        elapsed = current_time - measurements[0][0]
        images_generated = total - measurements[0][1]
        rate = images_generated / elapsed if elapsed > 0 else 0
        print(f"  After {elapsed:.1f}s: {total} images (+{images_generated}, {rate:.2f} img/s total, {rate/$NUM_PROCESSES:.3f} img/s per process)")
    
    if i < 4:
        time.sleep(4)

# Calculate final metrics
if len(measurements) >= 2:
    total_elapsed = measurements[-1][0] - measurements[0][0]
    total_images = measurements[-1][1] - measurements[0][1]
    avg_rate = total_images / total_elapsed if total_elapsed > 0 else 0
    per_process_rate = avg_rate / $NUM_PROCESSES
    
    print("")
    print(f"RESULT: {$NUM_PROCESSES} processes = {avg_rate:.2f} img/s total, {per_process_rate:.3f} img/s per process")
    
    # Save to results file
    with open('/root/FaBCode/benchmark_results.txt', 'a') as f:
        f.write(f"Processes: {$NUM_PROCESSES}\n")
        f.write(f"  Total rate: {avg_rate:.2f} img/s\n")
        f.write(f"  Per-process: {per_process_rate:.3f} img/s\n")
        f.write(f"  Efficiency: {(per_process_rate/1.5)*100:.1f}% of 1.5 img/s baseline\n")
        f.write("\n")
EOF
    
    # Kill processes
    pkill -f generate_synthetic_playmat_screenshots
    sleep 2
done

echo ""
echo "=========================================="
echo "Benchmark Complete!"
echo "=========================================="
echo ""
cat "$RESULTS_FILE"
