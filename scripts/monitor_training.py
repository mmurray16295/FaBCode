"""
Training Monitoring Script
Real-time monitoring of YOLO11x training progress with alerts
"""

import os
import sys
import time
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

def monitor_training(
    results_dir="runs/detect/fab_2641_yolo11x",
    check_interval=60,
    alert_threshold_minutes=30,
    email_alerts=False,
    email_address=None
):
    """
    Monitor training progress and alert if issues detected
    
    Args:
        results_dir: Directory containing training results
        check_interval: Seconds between checks
        alert_threshold_minutes: Alert if no progress for this many minutes
        email_alerts: Send email alerts
        email_address: Email address for alerts
    """
    
    print("=" * 80)
    print("YOLO11x Training Monitor")
    print("=" * 80)
    print(f"Monitoring: {results_dir}")
    print(f"Check interval: {check_interval}s")
    print(f"Alert threshold: {alert_threshold_minutes} minutes")
    print("=" * 80)
    
    results_csv = Path(results_dir) / "results.csv"
    
    last_epoch = -1
    last_update_time = time.time()
    start_time = time.time()
    
    print("\nWaiting for training to start...")
    
    # Wait for results.csv to be created
    while not results_csv.exists():
        time.sleep(check_interval)
        elapsed = time.time() - start_time
        print(f"  Waiting... ({elapsed/60:.1f} minutes)")
        
        if elapsed > 600:  # 10 minutes
            print("\n⚠️  WARNING: Training hasn't started after 10 minutes")
            print("   Check if training script is running correctly")
    
    print("\n✓ Training started! Beginning monitoring...\n")
    
    try:
        while True:
            time.sleep(check_interval)
            
            # Read results
            if not results_csv.exists():
                print("⚠️  WARNING: results.csv disappeared!")
                continue
            
            try:
                df = pd.read_csv(results_csv)
            except Exception as e:
                # File might be being written to
                continue
            
            if len(df) == 0:
                continue
            
            current_epoch = len(df) - 1
            
            # Check if new epoch completed
            if current_epoch > last_epoch:
                last_epoch = current_epoch
                last_update_time = time.time()
                
                # Get latest metrics
                latest = df.iloc[-1]
                
                # Calculate stats
                elapsed = time.time() - start_time
                epochs_per_hour = (current_epoch + 1) / (elapsed / 3600)
                
                # Estimate remaining time (assuming 400 epochs)
                total_epochs = 400
                remaining_epochs = total_epochs - (current_epoch + 1)
                remaining_hours = remaining_epochs / epochs_per_hour if epochs_per_hour > 0 else 0
                eta = datetime.now() + timedelta(hours=remaining_hours)
                
                # Display update
                print(f"\n{'='*80}")
                print(f"Epoch {current_epoch + 1}/{total_epochs} Complete")
                print(f"{'='*80}")
                print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Elapsed: {elapsed/3600:.1f}h | Remaining: {remaining_hours:.1f}h | ETA: {eta.strftime('%Y-%m-%d %H:%M')}")
                print(f"Speed: {epochs_per_hour:.2f} epochs/hour")
                print(f"\nMetrics:")
                print(f"  mAP@0.5:     {latest.get('metrics/mAP50(B)', 0):.3f}")
                print(f"  mAP@0.5:0.95: {latest.get('metrics/mAP50-95(B)', 0):.3f}")
                print(f"  Precision:    {latest.get('metrics/precision(B)', 0):.3f}")
                print(f"  Recall:       {latest.get('metrics/recall(B)', 0):.3f}")
                print(f"  Box Loss:     {latest.get('train/box_loss', 0):.4f}")
                print(f"  Cls Loss:     {latest.get('train/cls_loss', 0):.4f}")
                print(f"  DFL Loss:     {latest.get('train/dfl_loss', 0):.4f}")
                
                # Check for issues
                if current_epoch > 50:
                    map50 = latest.get('metrics/mAP50(B)', 0)
                    if map50 < 0.5:
                        print(f"\n⚠️  WARNING: Low mAP@0.5 ({map50:.3f}) at epoch {current_epoch}")
                        print("   Training may not be converging properly")
                    
                    # Check if losses are increasing
                    if len(df) > 10:
                        recent_box_loss = df['train/box_loss'].iloc[-10:].mean()
                        earlier_box_loss = df['train/box_loss'].iloc[-20:-10].mean()
                        if recent_box_loss > earlier_box_loss * 1.1:
                            print(f"\n⚠️  WARNING: Box loss increasing")
                            print(f"   Recent: {recent_box_loss:.4f} vs Earlier: {earlier_box_loss:.4f}")
                
                print(f"{'='*80}\n")
            
            else:
                # Check if training has stalled
                time_since_update = time.time() - last_update_time
                if time_since_update > alert_threshold_minutes * 60:
                    print(f"\n⚠️  ALERT: No progress for {time_since_update/60:.1f} minutes!")
                    print(f"   Last epoch: {last_epoch}")
                    print(f"   Training may have crashed or stalled")
                    
                    if email_alerts and email_address:
                        send_email_alert(email_address, "Training Stalled", 
                                       f"No progress for {time_since_update/60:.1f} minutes")
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")
        print(f"Final epoch: {last_epoch}")
        print(f"Total time: {(time.time() - start_time)/3600:.2f} hours")
    
    except Exception as e:
        print(f"\n❌ Monitoring error: {e}")
        raise


def send_email_alert(email, subject, message):
    """Send email alert (requires SMTP configuration)"""
    # TODO: Implement email alerts if needed
    print(f"\n📧 Email alert would be sent to {email}")
    print(f"   Subject: {subject}")
    print(f"   Message: {message}")


def quick_check(results_dir="runs/detect/fab_2641_yolo11x"):
    """Quick check of current training status"""
    
    results_csv = Path(results_dir) / "results.csv"
    
    if not results_csv.exists():
        print("❌ No training results found")
        print(f"   Looking for: {results_csv}")
        return
    
    df = pd.read_csv(results_csv)
    
    if len(df) == 0:
        print("⚠️  Training started but no epochs completed yet")
        return
    
    current_epoch = len(df) - 1
    latest = df.iloc[-1]
    
    # Calculate best metrics
    best_map50 = df['metrics/mAP50(B)'].max()
    best_epoch = df['metrics/mAP50(B)'].idxmax()
    
    print("=" * 80)
    print("Training Status")
    print("=" * 80)
    print(f"Current Epoch: {current_epoch + 1}")
    print(f"Best Epoch: {best_epoch + 1} (mAP@0.5: {best_map50:.3f})")
    print(f"\nCurrent Metrics:")
    print(f"  mAP@0.5:     {latest.get('metrics/mAP50(B)', 0):.3f}")
    print(f"  mAP@0.5:0.95: {latest.get('metrics/mAP50-95(B)', 0):.3f}")
    print(f"  Precision:    {latest.get('metrics/precision(B)', 0):.3f}")
    print(f"  Recall:       {latest.get('metrics/recall(B)', 0):.3f}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor YOLO11x training progress")
    parser.add_argument("--results-dir", type=str, default="runs/detect/fab_2641_yolo11x", 
                       help="Training results directory")
    parser.add_argument("--interval", type=int, default=60, 
                       help="Check interval in seconds")
    parser.add_argument("--alert-threshold", type=int, default=30, 
                       help="Alert if no progress for this many minutes")
    parser.add_argument("--quick-check", action="store_true", 
                       help="Just check current status and exit")
    
    args = parser.parse_args()
    
    if args.quick_check:
        quick_check(args.results_dir)
    else:
        monitor_training(
            results_dir=args.results_dir,
            check_interval=args.interval,
            alert_threshold_minutes=args.alert_threshold
        )
