#!/usr/bin/env python3
"""
Smart AWS S3 Downloader with Dynamic Concurrency Control
Automatically adjusts the number of concurrent downloads based on CPU usage
to target 95% CPU utilization for optimal performance.
"""

import boto3
import psutil
import threading
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from pathlib import Path
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('aws_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SmartS3Downloader:
    def __init__(self, bucket_name, prefix='', local_dir='./downloads', 
                 target_cpu=95.0, min_workers=1, max_workers=2000):
        """
        Initialize the Smart S3 Downloader
        
        Args:
            bucket_name: Name of the S3 bucket
            prefix: S3 prefix/folder to download from
            local_dir: Local directory to download files to
            target_cpu: Target CPU utilization percentage (default 95%)
            min_workers: Minimum number of concurrent downloads
            max_workers: Maximum number of concurrent downloads (default 2000 for handling many small files)
        """
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.local_dir = Path(local_dir)
        self.target_cpu = target_cpu
        self.min_workers = min_workers
        self.max_workers = max_workers
        
        # Dynamic concurrency control
        self.current_workers = min_workers
        self.worker_lock = threading.Lock()
        
        # Statistics
        self.downloaded_count = 0
        self.failed_count = 0
        self.total_bytes = 0
        self.start_time = None
        
        # CPU monitoring
        self.cpu_samples = []
        self.adjustment_interval = 2  # Adjust workers every 2 seconds for faster scaling
        self.last_adjustment = time.time()
        
        # Initialize S3 client with increased connection pool
        from botocore.config import Config
        config = Config(
            max_pool_connections=max_workers,  # Match max workers to avoid connection pool bottleneck
            retries={'max_attempts': 3, 'mode': 'adaptive'}
        )
        self.s3_client = boto3.client('s3', config=config)
        
        # Create local directory
        self.local_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized downloader for bucket: {bucket_name}")
        logger.info(f"Target CPU: {target_cpu}%, Workers range: {min_workers}-{max_workers}")
    
    def get_cpu_usage(self):
        """Get current CPU usage percentage"""
        return psutil.cpu_percent(interval=0.1)
    
    def adjust_workers(self):
        """Dynamically adjust the number of workers based on CPU usage"""
        current_time = time.time()
        
        # Only adjust every N seconds
        if current_time - self.last_adjustment < self.adjustment_interval:
            return
        
        # Get average CPU usage from recent samples
        if not self.cpu_samples:
            return
        
        avg_cpu = sum(self.cpu_samples) / len(self.cpu_samples)
        self.cpu_samples.clear()
        
        with self.worker_lock:
            old_workers = self.current_workers
            
            # Calculate adjustment - be much more aggressive when CPU is very low
            cpu_diff = self.target_cpu - avg_cpu
            
            # Much more aggressive scaling when far below target
            if cpu_diff > 50:  # Way below target
                # Double the workers when CPU is very low
                self.current_workers = min(self.max_workers, self.current_workers * 2)
            elif cpu_diff > 30:
                # Increase by 50%
                self.current_workers = min(self.max_workers, int(self.current_workers * 1.5))
            elif cpu_diff > 15:
                # Increase by 30%
                self.current_workers = min(self.max_workers, int(self.current_workers * 1.3))
            elif cpu_diff > 5:
                # Increase by 15%
                self.current_workers = min(self.max_workers, int(self.current_workers * 1.15))
            elif cpu_diff < -5:  # CPU usage too high, decrease workers
                self.current_workers = max(self.min_workers, int(self.current_workers * 0.9))
            
            # Ensure at least 1 worker change when adjustment is needed
            if cpu_diff > 5 and old_workers == self.current_workers:
                self.current_workers = min(self.max_workers, old_workers + 1)
            
            if old_workers != self.current_workers:
                logger.info(f"CPU: {avg_cpu:.1f}% -> Adjusted workers: {old_workers} -> {self.current_workers}")
        
        self.last_adjustment = current_time
    
    def monitor_cpu(self, stop_event):
        """Background thread to monitor CPU usage"""
        while not stop_event.is_set():
            cpu = self.get_cpu_usage()
            self.cpu_samples.append(cpu)
            time.sleep(1)
    
    def list_s3_objects(self):
        """List all objects in the S3 bucket with the given prefix"""
        logger.info(f"Listing objects from s3://{self.bucket_name}/{self.prefix}")
        
        objects = []
        paginator = self.s3_client.get_paginator('list_objects_v2')
        
        try:
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=self.prefix):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        # Skip directories
                        if not obj['Key'].endswith('/'):
                            objects.append(obj)
        except Exception as e:
            logger.error(f"Error listing objects: {e}")
            raise
        
        logger.info(f"Found {len(objects)} objects to download")
        return objects
    
    def download_file(self, s3_key, size):
        """Download a single file from S3"""
        try:
            # Create local file path
            relative_path = s3_key
            if self.prefix and s3_key.startswith(self.prefix):
                relative_path = s3_key[len(self.prefix):].lstrip('/')
            
            local_path = self.local_dir / relative_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Skip if file already exists and has the same size
            if local_path.exists() and local_path.stat().st_size == size:
                logger.debug(f"Skipping (already exists): {s3_key}")
                return True, size, s3_key
            
            # Download the file
            self.s3_client.download_file(self.bucket_name, s3_key, str(local_path))
            logger.debug(f"Downloaded: {s3_key} ({size / 1024 / 1024:.2f} MB)")
            
            return True, size, s3_key
        
        except Exception as e:
            logger.error(f"Failed to download {s3_key}: {e}")
            return False, 0, s3_key
    
    def download_all(self):
        """Download all files with dynamic concurrency control"""
        self.start_time = time.time()
        
        # Get list of objects to download
        objects = self.list_s3_objects()
        
        if not objects:
            logger.warning("No objects found to download")
            return
        
        total_size = sum(obj['Size'] for obj in objects)
        logger.info(f"Total size to download: {total_size / 1024 / 1024 / 1024:.2f} GB")
        
        # Start CPU monitoring thread
        stop_event = threading.Event()
        cpu_monitor = threading.Thread(target=self.monitor_cpu, args=(stop_event,))
        cpu_monitor.daemon = True
        cpu_monitor.start()
        
        # Create a queue of tasks
        task_queue = Queue()
        for obj in objects:
            task_queue.put((obj['Key'], obj['Size']))
        
        # Process downloads with dynamic worker pool
        completed = 0
        total_tasks = len(objects)
        
        while not task_queue.empty() or self.current_workers > 0:
            # Adjust workers based on CPU
            self.adjust_workers()
            
            # Create executor with current worker count
            with self.worker_lock:
                workers = self.current_workers
            
            # Get batch of tasks
            batch_size = min(workers * 2, task_queue.qsize())
            if batch_size == 0:
                break
            
            batch = []
            for _ in range(batch_size):
                if not task_queue.empty():
                    batch.append(task_queue.get())
            
            # Execute batch
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(self.download_file, key, size): (key, size)
                    for key, size in batch
                }
                
                for future in as_completed(futures):
                    success, bytes_downloaded, key = future.result()
                    completed += 1
                    
                    if success:
                        self.downloaded_count += 1
                        self.total_bytes += bytes_downloaded
                    else:
                        self.failed_count += 1
                    
                    # Progress report
                    if completed % 10 == 0 or completed == total_tasks:
                        elapsed = time.time() - self.start_time
                        speed = self.total_bytes / elapsed / 1024 / 1024 if elapsed > 0 else 0
                        progress = (completed / total_tasks) * 100
                        cpu = self.get_cpu_usage()
                        
                        logger.info(
                            f"Progress: {completed}/{total_tasks} ({progress:.1f}%) | "
                            f"Speed: {speed:.2f} MB/s | CPU: {cpu:.1f}% | Workers: {workers}"
                        )
        
        # Stop CPU monitoring
        stop_event.set()
        cpu_monitor.join(timeout=2)
        
        # Final report
        self.print_summary()
    
    def print_summary(self):
        """Print download summary"""
        elapsed = time.time() - self.start_time
        avg_speed = self.total_bytes / elapsed / 1024 / 1024 if elapsed > 0 else 0
        
        logger.info("\n" + "="*60)
        logger.info("DOWNLOAD SUMMARY")
        logger.info("="*60)
        logger.info(f"Total files: {self.downloaded_count + self.failed_count}")
        logger.info(f"Successfully downloaded: {self.downloaded_count}")
        logger.info(f"Failed: {self.failed_count}")
        logger.info(f"Total data: {self.total_bytes / 1024 / 1024 / 1024:.2f} GB")
        logger.info(f"Time elapsed: {elapsed / 60:.2f} minutes")
        logger.info(f"Average speed: {avg_speed:.2f} MB/s")
        logger.info(f"Files saved to: {self.local_dir.absolute()}")
        logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Smart AWS S3 Downloader with dynamic concurrency control'
    )
    parser.add_argument('bucket', help='S3 bucket name')
    parser.add_argument('--prefix', default='', help='S3 prefix/folder (optional)')
    parser.add_argument('--output', '-o', default='./downloads', 
                       help='Local output directory (default: ./downloads)')
    parser.add_argument('--target-cpu', type=float, default=95.0,
                       help='Target CPU utilization percentage (default: 95)')
    parser.add_argument('--min-workers', type=int, default=1,
                       help='Minimum number of concurrent downloads (default: 1)')
    parser.add_argument('--max-workers', type=int, default=2000,
                       help='Maximum number of concurrent downloads (default: 2000)')
    parser.add_argument('--region', default=None,
                       help='AWS region (optional, uses default if not specified)')
    
    args = parser.parse_args()
    
    # Configure AWS region if specified
    if args.region:
        boto3.setup_default_session(region_name=args.region)
    
    # Create and run downloader
    downloader = SmartS3Downloader(
        bucket_name=args.bucket,
        prefix=args.prefix,
        local_dir=args.output,
        target_cpu=args.target_cpu,
        min_workers=args.min_workers,
        max_workers=args.max_workers
    )
    
    try:
        downloader.download_all()
    except KeyboardInterrupt:
        logger.info("\nDownload interrupted by user")
        downloader.print_summary()
    except Exception as e:
        logger.error(f"Download failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
