@echo off
echo ========================================
echo  FaB Card Detector - Phase 3 Weights
echo  Reassembly Script (Windows)
echo ========================================
echo.

echo Reassembling best.pt from 4 parts...
echo.

copy /b Phase3_best_part_a+Phase3_best_part_b+Phase3_best_part_c+Phase3_best_part_d best.pt

if %errorlevel% == 0 (
    echo.
    echo SUCCESS! best.pt has been created.
    echo File size: 360,824,151 bytes (345 MB)
    echo.
    echo You can now use this model for:
    echo  - Transfer learning: Start training from these weights
    echo  - Inference: Test detection on real images
    echo  - Application: Use with fab_detector_app.py
    echo.
) else (
    echo.
    echo ERROR: Failed to reassemble file.
    echo Make sure all 4 part files are present.
    echo.
)

pause
