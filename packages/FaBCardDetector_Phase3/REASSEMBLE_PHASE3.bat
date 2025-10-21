@echo off
echo ========================================
echo Reassembling FaB Detector Phase 3
echo ========================================
echo.
echo This will combine the split files...
echo.

copy /b Phase3_part_aa + Phase3_part_ab + Phase3_part_ac FaBCardDetector_Phase3_20251020.zip

echo.
echo ========================================
echo Done! Extract FaBCardDetector_Phase3_20251020.zip
echo ========================================
pause
