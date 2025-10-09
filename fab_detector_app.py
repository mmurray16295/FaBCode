#!/usr/bin/env python3
"""
FaB Card Detector - Live Detection Application
===============================================

Real-time Flesh and Blood card detection with two modes:
1. Windowed Mode: Shows captured screen with detection boxes
2. Transparent Overlay Mode: Invisible window that shows card preview on hover

Features:
- Multi-monitor support
- Configurable confidence thresholds
- Card image preview on hover
- Performance metrics
- GUI launcher for easy configuration
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import json
import os
import sys
import requests
from PIL import Image
from io import BytesIO
import argparse
import time
import ctypes
from ctypes import wintypes
from pathlib import Path
import threading

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import mss
    import mss.tools
except ImportError:
    mss = None


class DetectorGUI:
    """GUI launcher for the card detector."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("FaB Card Detector")
        self.root.geometry("650x800")
        self.root.resizable(True, True)  # Allow window resizing
        self.root.minsize(600, 700)  # Minimum size to prevent too small
        
        # Detection thread
        self.detector_thread = None
        self.detector_running = False
        
        self._create_widgets()
        self._load_defaults()
        
    def _create_widgets(self):
        """Create GUI widgets."""
        
        # Create a canvas with scrollbar for scrolling
        canvas = tk.Canvas(self.root, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        # Bind mousewheel for scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Title
        title = tk.Label(scrollable_frame, text="FaB Card Live Detector", 
                        font=("Arial", 20, "bold"))
        title.pack(pady=20)
        
        # Model selection
        model_frame = ttk.LabelFrame(scrollable_frame, text="Model Configuration", padding=10)
        model_frame.pack(fill="x", padx=20, pady=10)
        
        ttk.Label(model_frame, text="Model Weights:").grid(row=0, column=0, sticky="w", pady=5)
        # Default to models/best.pt for packaged version, fallback to training path
        default_model = "models/best.pt" if os.path.exists("models/best.pt") else "runs/train/phase1_100classes/weights/best.pt"
        self.model_path = tk.StringVar(value=default_model)
        model_entry = ttk.Entry(model_frame, textvariable=self.model_path, width=40)
        model_entry.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(model_frame, text="Browse", command=self._browse_model).grid(row=0, column=2, padx=5)
        
        # Detection mode
        mode_frame = ttk.LabelFrame(scrollable_frame, text="Detection Mode", padding=10)
        mode_frame.pack(fill="x", padx=20, pady=10)
        
        self.mode = tk.StringVar(value="windowed")
        ttk.Radiobutton(mode_frame, text="Windowed Mode (Shows detection boxes)", 
                       variable=self.mode, value="windowed").pack(anchor="w", pady=5)
        ttk.Radiobutton(mode_frame, text="Transparent Overlay Mode (Invisible until hover)", 
                       variable=self.mode, value="overlay").pack(anchor="w", pady=5)
        
        # Monitor selection
        monitor_frame = ttk.LabelFrame(scrollable_frame, text="Monitor Configuration", padding=10)
        monitor_frame.pack(fill="x", padx=20, pady=10)
        
        ttk.Label(monitor_frame, text="Capture Monitor:").grid(row=0, column=0, sticky="w", pady=5)
        self.capture_monitor = tk.IntVar(value=1)
        ttk.Spinbox(monitor_frame, from_=1, to=4, textvariable=self.capture_monitor, 
                   width=10).grid(row=0, column=1, sticky="w", padx=5)
        
        ttk.Label(monitor_frame, text="Display Monitor:").grid(row=1, column=0, sticky="w", pady=5)
        self.display_monitor = tk.IntVar(value=2)
        ttk.Spinbox(monitor_frame, from_=1, to=4, textvariable=self.display_monitor, 
                   width=10).grid(row=1, column=1, sticky="w", padx=5)
        
        ttk.Label(monitor_frame, text="(Monitor 1 = Primary, 2 = Secondary, etc.)", 
                 font=("Arial", 8, "italic")).grid(row=2, column=0, columnspan=2, sticky="w", pady=2)
        
        # Detection settings
        settings_frame = ttk.LabelFrame(scrollable_frame, text="Detection Settings", padding=10)
        settings_frame.pack(fill="x", padx=20, pady=10)
        
        ttk.Label(settings_frame, text="Confidence Threshold:").grid(row=0, column=0, sticky="w", pady=5)
        self.conf_threshold = tk.DoubleVar(value=0.69)
        conf_scale = ttk.Scale(settings_frame, from_=0.1, to=0.95, variable=self.conf_threshold, 
                              orient="horizontal", length=200)
        conf_scale.grid(row=0, column=1, padx=5)
        self.conf_label = ttk.Label(settings_frame, text="0.69")
        self.conf_label.grid(row=0, column=2, padx=5)
        conf_scale.configure(command=lambda v: self.conf_label.config(text=f"{float(v):.2f}"))
        
        ttk.Label(settings_frame, text="IOU Threshold:").grid(row=1, column=0, sticky="w", pady=5)
        self.iou_threshold = tk.DoubleVar(value=0.50)
        iou_scale = ttk.Scale(settings_frame, from_=0.1, to=0.95, variable=self.iou_threshold, 
                             orient="horizontal", length=200)
        iou_scale.grid(row=1, column=1, padx=5)
        self.iou_label = ttk.Label(settings_frame, text="0.50")
        self.iou_label.grid(row=1, column=2, padx=5)
        iou_scale.configure(command=lambda v: self.iou_label.config(text=f"{float(v):.2f}"))
        
        # Overlay settings
        overlay_frame = ttk.LabelFrame(scrollable_frame, text="Overlay Settings", padding=10)
        overlay_frame.pack(fill="x", padx=20, pady=10)
        
        self.topmost = tk.BooleanVar(value=True)
        ttk.Checkbutton(overlay_frame, text="Keep window always on top", 
                       variable=self.topmost).pack(anchor="w", pady=2)
        
        self.transparent = tk.BooleanVar(value=True)
        ttk.Checkbutton(overlay_frame, text="Enable transparency (Windows only)", 
                       variable=self.transparent).pack(anchor="w", pady=2)
        
        self.click_through = tk.BooleanVar(value=False)
        ttk.Checkbutton(overlay_frame, text="Click-through mode (Windows only)", 
                       variable=self.click_through).pack(anchor="w", pady=2)
        
        ttk.Label(overlay_frame, text="Card Preview Size:").pack(anchor="w", pady=5)
        size_frame = ttk.Frame(overlay_frame)
        size_frame.pack(anchor="w", padx=20)
        self.card_width = tk.IntVar(value=300)
        self.card_height = tk.IntVar(value=420)
        ttk.Label(size_frame, text="Width:").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(size_frame, from_=100, to=800, textvariable=self.card_width, 
                   width=10).grid(row=0, column=1, padx=5)
        ttk.Label(size_frame, text="Height:").grid(row=0, column=2, sticky="w", padx=(20, 0))
        ttk.Spinbox(size_frame, from_=100, to=1000, textvariable=self.card_height, 
                   width=10).grid(row=0, column=3, padx=5)
        
        # Control buttons
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(pady=20)
        
        self.start_button = ttk.Button(button_frame, text="Start Detection", 
                                       command=self._start_detection, width=20)
        self.start_button.grid(row=0, column=0, padx=10)
        
        self.stop_button = ttk.Button(button_frame, text="Stop Detection", 
                                      command=self._stop_detection, width=20, state="disabled")
        self.stop_button.grid(row=0, column=1, padx=10)
        
        # Status
        self.status_label = ttk.Label(scrollable_frame, text="Ready", 
                                     font=("Arial", 10, "bold"), foreground="green")
        self.status_label.pack(pady=10)
        
    def _load_defaults(self):
        """Load default settings from config if exists."""
        config_path = Path("detector_config.json")
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                self.model_path.set(config.get('model_path', self.model_path.get()))
                self.conf_threshold.set(config.get('conf_threshold', 0.69))
                self.iou_threshold.set(config.get('iou_threshold', 0.50))
            except Exception as e:
                print(f"Could not load config: {e}")
    
    def _save_config(self):
        """Save current settings to config."""
        config = {
            'model_path': self.model_path.get(),
            'conf_threshold': self.conf_threshold.get(),
            'iou_threshold': self.iou_threshold.get(),
        }
        try:
            with open("detector_config.json", 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Could not save config: {e}")
    
    def _browse_model(self):
        """Browse for model file."""
        filename = filedialog.askopenfilename(
            title="Select Model Weights",
            filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")]
        )
        if filename:
            self.model_path.set(filename)
    
    def _start_detection(self):
        """Start detection in a separate thread."""
        # Validate model path
        if not Path(self.model_path.get()).exists():
            messagebox.showerror("Error", f"Model not found: {self.model_path.get()}")
            return
        
        # Check dependencies
        if YOLO is None:
            messagebox.showerror("Error", "ultralytics not installed. Run: pip install ultralytics")
            return
        
        if mss is None:
            messagebox.showerror("Error", "mss not installed. Run: pip install mss")
            return
        
        # Save config
        self._save_config()
        
        # Update UI
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.status_label.config(text="Starting detection...", foreground="orange")
        
        # Start detector thread
        self.detector_running = True
        self.detector_thread = threading.Thread(target=self._run_detector, daemon=True)
        self.detector_thread.start()
    
    def _stop_detection(self):
        """Stop detection thread."""
        self.detector_running = False
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status_label.config(text="Stopping...", foreground="orange")
        
        # Wait for thread to finish
        if self.detector_thread:
            self.detector_thread.join(timeout=2.0)
        
        self.status_label.config(text="Stopped", foreground="red")
    
    def _run_detector(self):
        """Run the detector (in separate thread)."""
        try:
            # Update status
            self.root.after(0, lambda: self.status_label.config(
                text="Running...", foreground="green"))
            
            # Build arguments
            args = argparse.Namespace(
                weights=self.model_path.get(),
                conf=self.conf_threshold.get(),
                iou=self.iou_threshold.get(),
                imgsz=640,
                capture_monitor=self.capture_monitor.get(),
                display_monitor=self.display_monitor.get() if self.mode.get() == "overlay" else None,
                overlay_only=self.mode.get() == "overlay",
                mask_overlay=True,
                card_size=(self.card_width.get(), self.card_height.get()),
                topmost=self.topmost.get(),
                transparent=self.transparent.get() and self.mode.get() == "overlay",
                chroma=(255, 0, 255),
                click_through=self.click_through.get() and self.mode.get() == "overlay",
                video=None
            )
            
            # Run detector
            detector = CardDetector(args, stop_callback=lambda: not self.detector_running)
            detector.run()
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Detection Error", str(e)))
        finally:
            self.root.after(0, self._stop_detection)


class CardDetector:
    """Core detection engine."""
    
    def __init__(self, args, stop_callback=None):
        self.args = args
        self.stop_callback = stop_callback
        
        # Load model
        print(f"[init] Loading model from {args.weights}...")
        self.model = YOLO(args.weights)
        print(f"[init] Model loaded with {len(self.model.names)} classes")
        
        # Load card data
        self.card_data = self._load_card_data()
        self.card_image_cache = {}
        
        # Tracking
        self.last_overlay_rect = None
        self.mouse_pos = (0, 0)
        self.fps_history = []
        
    def _load_card_data(self):
        """Load card metadata."""
        card_json_path = Path('data/card.json')
        if card_json_path.exists():
            try:
                with open(card_json_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[warning] Could not load card.json: {e}")
        return []
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for window."""
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_pos = (x, y)
    
    def get_image_url_by_name(self, card_name):
        """Get card image URL by name."""
        search_name = card_name.replace('_', ' ').lower().strip()
        
        for card in self.card_data:
            card_display_name = card.get('name', '').lower().strip()
            if card_display_name == search_name:
                printings = card.get('printings', [])
                if printings:
                    return printings[0].get('image_url')
            
            # Fuzzy match without punctuation
            import re
            card_no_punct = re.sub(r'[^\w\s]', '', card_display_name)
            search_no_punct = re.sub(r'[^\w\s]', '', search_name)
            if card_no_punct == search_no_punct:
                printings = card.get('printings', [])
                if printings:
                    return printings[0].get('image_url')
        
        return None
    
    def get_card_image(self, url):
        """Load and cache card image."""
        if url in self.card_image_cache:
            return self.card_image_cache[url]
        
        try:
            response = requests.get(url, timeout=5)
            img = Image.open(BytesIO(response.content)).convert('RGBA')
            self.card_image_cache[url] = img
            return img
        except Exception as e:
            print(f"[warning] Could not load card image: {e}")
            return None
    
    def run(self):
        """Main detection loop."""
        window_name = 'FaB Card Detector'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self._mouse_callback)
        
        # Setup screen capture
        with mss.mss() as sct:
            monitors = sct.monitors
            cap_mon = max(1, min(self.args.capture_monitor, len(monitors) - 1))
            monitor = monitors[cap_mon]
            
            # Display monitor
            if self.args.display_monitor is None:
                disp_mon = cap_mon
            else:
                disp_mon = max(1, min(self.args.display_monitor, len(monitors) - 1))
            display_info = monitors[disp_mon]
            
            base_h = monitor['height']
            base_w = monitor['width']
            
            # Position window
            window_offset = (100, 100)
            self._move_window_to_monitor(window_name, display_info, window_offset)
            
            try:
                cv2.resizeWindow(window_name, int(base_w), int(base_h))
            except Exception:
                pass
            
            # Window styling
            if self.args.topmost:
                self._set_window_topmost(window_name, True)
            
            if self.args.overlay_only and self.args.transparent:
                self._enable_chromakey_transparency(
                    window_name, 
                    rgb=tuple(self.args.chroma), 
                    click_through=self.args.click_through
                )
            
            card_w, card_h = self.args.card_size
            
            # Main loop
            while True:
                # Check stop callback
                if self.stop_callback and self.stop_callback():
                    break
                
                start_time = time.time()
                
                # Capture screen
                img = np.array(sct.grab(monitor))
                frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                
                # Run detection
                boxes, clss, names = self._detect(frame)
                
                # Filter boxes
                boxes, clss = self._filter_boxes(boxes, clss, frame.shape)
                
                # Find hovered box
                mouse_over_box = self._find_hovered_box(boxes, clss, names, monitor)
                
                # Create display frame
                display_frame = self._create_display_frame(
                    frame, base_h, base_w, boxes, clss, names, mouse_over_box, card_w, card_h
                )
                
                # Calculate FPS
                fps = 1.0 / (time.time() - start_time)
                self.fps_history.append(fps)
                if len(self.fps_history) > 30:
                    self.fps_history.pop(0)
                avg_fps = sum(self.fps_history) / len(self.fps_history)
                
                # Draw FPS
                cv2.putText(display_frame, f"FPS: {avg_fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Cards: {len(boxes)}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow(window_name, display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
        
        cv2.destroyAllWindows()
    
    def _detect(self, frame):
        """Run YOLO detection on frame."""
        try:
            results = self.model.predict(
                source=frame,
                imgsz=self.args.imgsz,
                conf=self.args.conf,
                iou=self.args.iou,
                verbose=False
            )
            
            if results and len(results) > 0:
                r0 = results[0]
                if hasattr(r0, 'boxes') and r0.boxes is not None:
                    boxes = r0.boxes.xyxy.detach().cpu().numpy()
                    clss = r0.boxes.cls.detach().cpu().numpy()
                    names = r0.names if hasattr(r0, 'names') else self.model.names
                    return boxes, clss, names
        except Exception as e:
            print(f"[error] Detection failed: {e}")
        
        return np.array([]), np.array([]), self.model.names
    
    def _filter_boxes(self, boxes, clss, frame_shape):
        """Filter boxes by aspect ratio and area."""
        H, W = frame_shape[:2]
        keep = []
        
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            
            ar = h / w  # Cards are tall
            area = (w * h) / (W * H)
            
            # Reasonable card aspect ratios and sizes
            if 0.6 <= ar <= 2.2 and 0.004 <= area <= 0.15:
                keep.append(i)
        
        return boxes[keep], clss[keep]
    
    def _find_hovered_box(self, boxes, clss, names, monitor):
        """Find which box the mouse is hovering over."""
        if self.args.transparent:
            # Use global mouse position
            gx, gy = self._get_global_mouse_pos()
            mx = gx - monitor['left']
            my = gy - monitor['top']
        else:
            # Use window-relative mouse position
            mx, my = self.mouse_pos
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(v) for v in box]
            class_id = int(clss[i]) if i < len(clss) else -1
            
            if isinstance(names, dict):
                label = names.get(class_id, str(class_id))
            else:
                label = str(class_id)
            
            if x1 <= mx <= x2 and y1 <= my <= y2:
                return (x1, y1, x2, y2, label)
        
        return None
    
    def _create_display_frame(self, frame, base_h, base_w, boxes, clss, names, mouse_over_box, card_w, card_h):
        """Create the display frame with boxes and card overlay."""
        if self.args.overlay_only:
            # Transparent overlay mode
            if self.args.transparent:
                r, g, b = self.args.chroma
                display_frame = np.full((base_h, base_w, 3), (b, g, r), dtype=np.uint8)
            else:
                display_frame = np.zeros((base_h, base_w, 3), dtype=np.uint8)
        else:
            # Windowed mode
            display_frame = frame.copy()
            
            # Draw all boxes
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = [int(v) for v in box]
                class_id = int(clss[i]) if i < len(clss) else -1
                
                if isinstance(names, dict):
                    label = names.get(class_id, str(class_id))
                else:
                    label = str(class_id)
                
                # Draw box
                color = (0, 255, 0) if mouse_over_box and (x1, y1, x2, y2, label) == mouse_over_box else (255, 0, 0)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(display_frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
                cv2.putText(display_frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw card preview if hovering
        if mouse_over_box:
            bx1, by1, bx2, by2, class_name = mouse_over_box
            image_url = self.get_image_url_by_name(class_name)
            
            if image_url:
                card_img = self.get_card_image(image_url)
                if card_img:
                    card_img = card_img.resize((card_w, card_h))
                    card_np = np.array(card_img)
                    card_np = cv2.cvtColor(card_np, cv2.COLOR_RGBA2BGRA)
                    
                    # Position card below detection box
                    overlay_x = max(0, min(bx1, display_frame.shape[1] - card_np.shape[1]))
                    overlay_y = max(0, min(by2 + 10, display_frame.shape[0] - card_np.shape[0]))
                    
                    # Blend card with alpha
                    roi = display_frame[overlay_y:overlay_y + card_np.shape[0], 
                                       overlay_x:overlay_x + card_np.shape[1]]
                    
                    if roi.shape[:2] == card_np.shape[:2]:
                        alpha = card_np[:, :, 3:] / 255.0
                        roi[:] = (alpha * card_np[:, :, :3] + (1 - alpha) * roi).astype(np.uint8)
        
        return display_frame
    
    def _move_window_to_monitor(self, window_name, monitor_rect, offset_xy=(100, 100)):
        """Move window to specific monitor."""
        x = int(monitor_rect.get("left", 0) + offset_xy[0])
        y = int(monitor_rect.get("top", 0) + offset_xy[1])
        cv2.moveWindow(window_name, x, y)
    
    def _set_window_topmost(self, window_name, topmost=True):
        """Set window always-on-top (Windows)."""
        try:
            user32 = ctypes.windll.user32
            hwnd = user32.FindWindowW(None, window_name)
            if hwnd:
                HWND_TOPMOST = -1
                HWND_NOTOPMOST = -2
                SWP_NOSIZE = 0x0001
                SWP_NOMOVE = 0x0002
                SWP_NOACTIVATE = 0x0010
                user32.SetWindowPos(hwnd, HWND_TOPMOST if topmost else HWND_NOTOPMOST,
                                   0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE)
        except Exception:
            pass
    
    def _enable_chromakey_transparency(self, window_name, rgb=(255, 0, 255), click_through=False):
        """Enable chroma-key transparency (Windows)."""
        try:
            user32 = ctypes.windll.user32
            hwnd = user32.FindWindowW(None, window_name)
            if not hwnd:
                return
            
            GWL_EXSTYLE = -20
            WS_EX_LAYERED = 0x00080000
            WS_EX_TRANSPARENT = 0x00000020
            LWA_COLORKEY = 0x00000001
            
            current = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            new_style = current | WS_EX_LAYERED
            if click_through:
                new_style |= WS_EX_TRANSPARENT
            
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_style)
            
            r, g, b = rgb
            colorref = (r & 0xFF) | ((g & 0xFF) << 8) | ((b & 0xFF) << 16)
            user32.SetLayeredWindowAttributes(hwnd, colorref, 0, LWA_COLORKEY)
        except Exception:
            pass
    
    def _get_global_mouse_pos(self):
        """Get global mouse position (Windows)."""
        try:
            pt = wintypes.POINT()
            ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
            return (pt.x, pt.y)
        except Exception:
            return (0, 0)


def main():
    """Main entry point."""
    # Check if launched from command line with arguments
    if len(sys.argv) > 1:
        # Command-line mode (backward compatible)
        parser = argparse.ArgumentParser(description='FaB Card Live Detector')
        parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
        parser.add_argument('--conf', type=float, default=0.69, help='Confidence threshold')
        parser.add_argument('--iou', type=float, default=0.50, help='IOU threshold')
        parser.add_argument('--mode', type=str, choices=['windowed', 'overlay'], default='windowed')
        # Add more args as needed...
        
        args = parser.parse_args()
        detector = CardDetector(args)
        detector.run()
    else:
        # GUI mode
        root = tk.Tk()
        app = DetectorGUI(root)
        root.mainloop()


if __name__ == '__main__':
    main()
