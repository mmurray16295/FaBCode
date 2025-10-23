#!/bin/bash
# Auto-backup training checkpoints to S3
# Monitors training run and backs up weights every N minutes

TRAINING_DIR="/workspace/FaBCode/runs/full_training"
S3_BUCKET="s3://fabcode-backup-1760719829/training_backups"
BACKUP_INTERVAL=1800  # 30 minutes in seconds
LOG_FILE="/workspace/FaBCode/scripts/yolo_training/backup.log"

echo "=== Auto-Backup Started: $(date) ===" | tee -a "$LOG_FILE"
echo "Backup interval: $BACKUP_INTERVAL seconds ($(($BACKUP_INTERVAL / 60)) minutes)" | tee -a "$LOG_FILE"
echo "S3 Bucket: $S3_BUCKET" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

while true; do
    # Find the most recent training run directory
    LATEST_RUN=$(ls -td "$TRAINING_DIR"/yolo11x_2641classes_* 2>/dev/null | head -1)
    
    if [ -z "$LATEST_RUN" ]; then
        echo "[$(date)] No training run found yet, waiting..." | tee -a "$LOG_FILE"
        sleep 60
        continue
    fi
    
    RUN_NAME=$(basename "$LATEST_RUN")
    WEIGHTS_DIR="$LATEST_RUN/weights"
    
    if [ -d "$WEIGHTS_DIR" ]; then
        echo "[$(date)] Backing up $RUN_NAME..." | tee -a "$LOG_FILE"
        
        # Backup all weight files
        aws s3 sync "$WEIGHTS_DIR" "$S3_BUCKET/$RUN_NAME/weights/" \
            --exclude "*" \
            --include "*.pt" \
            --no-progress 2>&1 | tee -a "$LOG_FILE"
        
        # Also backup the latest results and plots
        if [ -f "$LATEST_RUN/results.csv" ]; then
            aws s3 cp "$LATEST_RUN/results.csv" "$S3_BUCKET/$RUN_NAME/results.csv" --no-progress 2>&1 | tee -a "$LOG_FILE"
        fi
        
        if [ -d "$LATEST_RUN/plots" ]; then
            aws s3 sync "$LATEST_RUN/plots" "$S3_BUCKET/$RUN_NAME/plots/" --no-progress 2>&1 | tee -a "$LOG_FILE"
        fi
        
        echo "[$(date)] Backup complete. Next backup in $(($BACKUP_INTERVAL / 60)) minutes." | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
    else
        echo "[$(date)] Weights directory not found yet in $RUN_NAME, waiting..." | tee -a "$LOG_FILE"
    fi
    
    sleep "$BACKUP_INTERVAL"
done
