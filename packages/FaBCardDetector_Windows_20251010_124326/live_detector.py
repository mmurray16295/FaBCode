#!/usr/bin/env python3
"""
FaB Card Live Detector
======================

Real-time Flesh and Blood card detection application.
Supports webcam, screen capture, and image file input.

Usage:
    # Webcam detection
    python live_detector.py --source webcam
    
    # Screen capture detection
    python live_detector.py --source screen
    
    # Test on image
    python live_detector.py --source image.jpg
    
    # Use custom model
    python live_detector.py --source webcam --model path/to/best.pt
"""

import cv2
import argparse
import numpy as np
from pathlib import Path
import time
import json
from ultralytics import YOLO
import mss
import mss.tools

class FaBCardDetector:
    """Real-time Flesh and Blood card detector."""
    
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45):
        """
        Initialize the detector.
        
        Args:
            model_path: Path to YOLO model weights (.pt file)
            conf_threshold: Confidence threshold for detections (0-1)
            iou_threshold: IOU threshold for NMS
        """
        print(f"[init] Loading model from {model_path}...")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Get class names from model
        self.class_names = self.model.names
        print(f"[init] Model loaded with {len(self.class_names)} classes")
        
        # Load card data if available
        self.card_data = self._load_card_data()
        
        # Performance tracking
        self.fps_history = []
        self.max_fps_history = 30
        
    def _load_card_data(self):
        """Load card metadata from card.json if available."""
        card_json_path = Path(__file__).parent / 'data' / 'card.json'
        if card_json_path.exists():
            try:
                with open(card_json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"[init] Loaded metadata for {len(data)} cards")
                return data
            except Exception as e:
                print(f"[warning] Could not load card.json: {e}")
        return {}
    
    def get_card_info(self, class_name):
        """Get card metadata from loaded data."""
        if not self.card_data:
            return None
        
        # Try to find card by name
        for card in self.card_data:
            if card.get('name', '').lower().replace(' ', '_').replace('-', '_') == class_name.lower():
                return card
        return None
    
    def detect_webcam(self, camera_id=0, display_size=(1280, 720)):
        """
        Run detection on webcam feed.
        
        Args:
            camera_id: Camera device ID (usually 0 for default webcam)
            display_size: (width, height) for display window
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"[error] Could not open camera {camera_id}")
            return
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_size[1])
        
        print("[info] Starting webcam detection. Press 'q' to quit, 'c' to toggle confidence display.")
        show_conf = True
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[error] Failed to grab frame")
                break
            
            # Run detection
            annotated_frame, detections = self.detect_frame(frame, show_conf=show_conf)
            
            # Display FPS
            fps = self._get_average_fps()
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display detection count
            cv2.putText(annotated_frame, f"Cards: {len(detections)}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display instructions
            cv2.putText(annotated_frame, "Press 'q' to quit, 'c' to toggle confidence", 
                       (10, annotated_frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow('FaB Card Detector - Webcam', annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                show_conf = not show_conf
        
        cap.release()
        cv2.destroyAllWindows()
    
    def detect_screen(self, monitor_number=1, display_scale=0.5):
        """
        Run detection on screen capture.
        
        Args:
            monitor_number: Monitor to capture (1 = primary)
            display_scale: Scale factor for display window (0.5 = 50% size)
        """
        with mss.mss() as sct:
            # Get monitor info
            if monitor_number > len(sct.monitors) - 1:
                print(f"[error] Monitor {monitor_number} not found. Available: {len(sct.monitors) - 1}")
                return
            
            monitor = sct.monitors[monitor_number]
            print(f"[info] Capturing monitor {monitor_number}: {monitor['width']}x{monitor['height']}")
            print("[info] Starting screen capture detection. Press 'q' to quit, 'c' to toggle confidence display.")
            show_conf = True
            
            while True:
                start_time = time.time()
                
                # Capture screen
                screenshot = sct.grab(monitor)
                frame = np.array(screenshot)
                
                # Convert BGRA to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
                # Run detection
                annotated_frame, detections = self.detect_frame(frame, show_conf=show_conf)
                
                # Scale for display
                display_width = int(frame.shape[1] * display_scale)
                display_height = int(frame.shape[0] * display_scale)
                annotated_frame = cv2.resize(annotated_frame, (display_width, display_height))
                
                # Display FPS
                fps = self._get_average_fps()
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display detection count
                cv2.putText(annotated_frame, f"Cards: {len(detections)}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display instructions
                cv2.putText(annotated_frame, "Press 'q' to quit, 'c' to toggle confidence", 
                           (10, annotated_frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                cv2.imshow('FaB Card Detector - Screen Capture', annotated_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    show_conf = not show_conf
        
        cv2.destroyAllWindows()
    
    def detect_image(self, image_path, save_path=None):
        """
        Run detection on a single image.
        
        Args:
            image_path: Path to input image
            save_path: Optional path to save annotated image
        """
        print(f"[info] Processing image: {image_path}")
        
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"[error] Could not load image: {image_path}")
            return
        
        annotated_frame, detections = self.detect_frame(frame, show_conf=True)
        
        print(f"[result] Detected {len(detections)} cards:")
        for det in detections:
            print(f"  - {det['class_name']} (confidence: {det['confidence']:.2%})")
        
        # Save if requested
        if save_path:
            cv2.imwrite(str(save_path), annotated_frame)
            print(f"[info] Saved annotated image to: {save_path}")
        
        # Display
        cv2.imshow('FaB Card Detector - Image', annotated_frame)
        print("[info] Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def detect_frame(self, frame, show_conf=True):
        """
        Run detection on a single frame.
        
        Args:
            frame: Input image (BGR format)
            show_conf: Whether to show confidence scores
            
        Returns:
            annotated_frame: Frame with bounding boxes drawn
            detections: List of detection dictionaries
        """
        start_time = time.time()
        
        # Run inference
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        # Calculate FPS
        inference_time = time.time() - start_time
        fps = 1.0 / inference_time if inference_time > 0 else 0
        self.fps_history.append(fps)
        if len(self.fps_history) > self.max_fps_history:
            self.fps_history.pop(0)
        
        # Process results
        annotated_frame = frame.copy()
        detections = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for box in boxes:
                # Extract box data
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.class_names[class_id]
                
                # Store detection
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name
                })
                
                # Draw bounding box
                color = self._get_class_color(class_id)
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # Prepare label
                if show_conf:
                    label = f"{class_name} {confidence:.2%}"
                else:
                    label = class_name
                
                # Draw label background
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    annotated_frame,
                    (int(x1), int(y1) - label_height - 10),
                    (int(x1) + label_width, int(y1)),
                    color,
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    annotated_frame,
                    label,
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
        
        return annotated_frame, detections
    
    def _get_class_color(self, class_id):
        """Get consistent color for each class."""
        np.random.seed(class_id)
        color = tuple(map(int, np.random.randint(0, 255, 3)))
        return color
    
    def _get_average_fps(self):
        """Calculate average FPS from recent history."""
        if not self.fps_history:
            return 0.0
        return sum(self.fps_history) / len(self.fps_history)


def main():
    parser = argparse.ArgumentParser(description='FaB Card Live Detector')
    parser.add_argument('--source', type=str, default='webcam',
                       help='Detection source: webcam, screen, or image path')
    parser.add_argument('--model', type=str, default='runs/train/phase1_100classes/weights/best.pt',
                       help='Path to YOLO model weights')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (0-1)')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IOU threshold for NMS')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID for webcam mode')
    parser.add_argument('--monitor', type=int, default=1,
                       help='Monitor number for screen capture mode')
    parser.add_argument('--scale', type=float, default=0.5,
                       help='Display scale for screen capture mode')
    parser.add_argument('--save', type=str, default=None,
                       help='Save annotated image to this path (image mode only)')
    
    args = parser.parse_args()
    
    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[error] Model not found: {model_path}")
        print("[info] Please specify a valid model path with --model")
        return
    
    # Initialize detector
    detector = FaBCardDetector(
        model_path=str(model_path),
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # Run detection based on source
    source = args.source.lower()
    
    if source == 'webcam':
        detector.detect_webcam(camera_id=args.camera)
    
    elif source == 'screen':
        detector.detect_screen(monitor_number=args.monitor, display_scale=args.scale)
    
    else:
        # Assume it's an image path
        image_path = Path(args.source)
        if not image_path.exists():
            print(f"[error] Image not found: {image_path}")
            return
        
        detector.detect_image(image_path, save_path=args.save)


if __name__ == '__main__':
    main()
