@echo off
echo ========================================
echo Reassembling FaB Detector Phase 4
echo ========================================
echo.
echo This will combine the split files...
echo.

copy /b Phase4_part_aa + Phase4_part_ab + Phase4_part_ac FaBCardDetector_Phase4_20251024.zip

echo.
echo ========================================
echo Done! Extract FaBCardDetector_Phase4_20251024.zip
echo ========================================
pause
