#!/bin/bash

echo "========================================"
echo " FaB Card Detector - Phase 3 Weights"
echo " Reassembly Script (Linux/Mac)"
echo "========================================"
echo

echo "Reassembling best.pt from 4 parts..."
echo

cat Phase3_best_part_a Phase3_best_part_b Phase3_best_part_c Phase3_best_part_d > best.pt

if [ $? -eq 0 ]; then
    FILE_SIZE=$(stat -f%z best.pt 2>/dev/null || stat -c%s best.pt 2>/dev/null)
    echo
    echo "SUCCESS! best.pt has been created."
    echo "File size: $FILE_SIZE bytes (345 MB)"
    echo
    echo "You can now use this model for:"
    echo "  - Transfer learning: Start training from these weights"
    echo "  - Inference: Test detection on real images"
    echo "  - Application: Use with fab_detector_app.py"
    echo
else
    echo
    echo "ERROR: Failed to reassemble file."
    echo "Make sure all 4 part files are present."
    echo
    exit 1
fi
