#!/bin/bash
# Monitor dataset generation progress

echo "================================================================================"
echo "DATASET GENERATION MONITOR"
echo "================================================================================"

# Check if generation is running
if pgrep -f "parallel_generate_dataset.py" > /dev/null; then
    echo "✓ Generation process is RUNNING"
    echo ""
else:
    echo "✗ No generation process found"
    echo ""
fi

# Show recent log output
echo "Recent log output:"
echo "--------------------------------------------------------------------------------"
tail -30 /root/FaBCode/generation_log.txt
echo "--------------------------------------------------------------------------------"

echo ""

# Count generated images
echo "Current image counts:"
for split in train valid test; do
    count=$(find /root/FaBCode/data/synthetic/$split/images -name "*.jpg" 2>/dev/null | wc -l)
    printf "  %-8s: %6d images\n" "$split" "$count"
done

total=$(find /root/FaBCode/data/synthetic/*/images -name "*.jpg" 2>/dev/null | wc -l)
echo "  ----------------------------------------"
printf "  %-8s: %6d images\n" "TOTAL" "$total"

echo ""
echo "Progress: $total / 125,000 ($(awk "BEGIN {printf \"%.1f\", ($total/125000)*100}")%)"
echo ""
echo "================================================================================"
echo "Commands:"
echo "  Watch live: tail -f /root/FaBCode/generation_log.txt"
echo "  Stop: pkill -f parallel_generate_dataset.py"
echo "================================================================================"
