#!/bin/bash
# Parallel random playmat generation script
# Generates remaining random images using 25 processes

TOTAL_IMAGES=654
NUM_PROCESSES=25
IMAGES_PER_PROCESS=$((TOTAL_IMAGES / NUM_PROCESSES))

echo "Starting parallel random generation: $NUM_PROCESSES processes, $IMAGES_PER_PROCESS images each"
mkdir -p /root/FaBCode/logs

# Start all processes in background
for i in $(seq 0 $((NUM_PROCESSES - 1))); do
    echo "Starting random process $i (${IMAGES_PER_PROCESS} images)"
    
    python3 /root/FaBCode/scripts/generate_random_playmat.py \
        --num-images $IMAGES_PER_PROCESS \
        --popularity-min 1 \
        --popularity-max 100 \
        --use-popularity-weights \
        --card-dirs /root/FaBCode/data/images/*/ \
        > "/root/FaBCode/logs/gen_random_p${i}.log" 2>&1 &
done

echo "All $NUM_PROCESSES processes started for random generation!"
echo "Monitor progress with: tail -f /root/FaBCode/logs/gen_random_p0.log"

# Wait for all background processes to complete
wait

echo "Random generation complete!"
