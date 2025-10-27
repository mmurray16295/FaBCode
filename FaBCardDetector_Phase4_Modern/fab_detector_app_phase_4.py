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
from PIL import Image, ImageTk
from io import BytesIO
import argparse
import time
import ctypes
from ctypes import wintypes
from pathlib import Path
import threading
import queue

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import mss
    import mss.tools
except ImportError:
    mss = None

try:
    import keyboard
except ImportError:
    keyboard = None

try:
    from confidence_booster import ConfidenceBooster
except ImportError:
    ConfidenceBooster = None
    print("[warning] ConfidenceBooster module not found - confidence boosting disabled")

try:
    from confidence_booster import ConfidenceBooster
except ImportError:
    ConfidenceBooster = None
    print("[warning] ConfidenceBooster module not found - confidence boosting disabled")


class DetectorGUI:
    """GUI launcher for the card detector."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("FaB Card Detector - Phase 4 (Pre-Filtering + Confidence Boosting)")
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
        title = tk.Label(scrollable_frame, text="FaB Card Live Detector - Phase 4", 
                        font=("Arial", 20, "bold"))
        title.pack(pady=20)
        
        # Detection mode
        mode_frame = ttk.LabelFrame(scrollable_frame, text="Detection Mode", padding=10)
        mode_frame.pack(fill="x", padx=20, pady=10)
        
        self.mode = tk.StringVar(value="overlay")
        ttk.Radiobutton(mode_frame, text="Windowed Mode (Shows detection boxes)", 
                       variable=self.mode, value="windowed").pack(anchor="w", pady=5)
        ttk.Radiobutton(mode_frame, text="Transparent Overlay Mode (Invisible until hover)", 
                       variable=self.mode, value="overlay").pack(anchor="w", pady=5)
        
        # Control buttons at top with colors
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(pady=20, padx=20)
        
        self.start_button = tk.Button(button_frame, text="Start Detection", 
                                       command=self._start_detection, width=20,
                                       bg="#28a745", fg="white", font=("Arial", 10, "bold"),
                                       activebackground="#218838", relief="raised", bd=2)
        self.start_button.grid(row=0, column=0, padx=10)
        
        self.stop_button = tk.Button(button_frame, text="Stop Detection", 
                                      command=self._stop_detection, width=20, state="disabled",
                                      bg="#dc3545", fg="white", font=("Arial", 10, "bold"),
                                      activebackground="#c82333", activeforeground="white",
                                      disabledforeground="#ffcccc", relief="raised", bd=2)
        self.stop_button.grid(row=0, column=1, padx=10)
        
        # Hero selection (right after buttons)
        hero_frame = ttk.LabelFrame(scrollable_frame, text="Hero Override (Optional - Auto-detects if left as 'Auto Detect')", padding=10)
        hero_frame.pack(fill="x", padx=20, pady=10)
        
        # Load hero list
        self.hero_names = self._load_hero_names()
        hero_options = ["Auto Detect"] + self.hero_names
        
        ttk.Label(hero_frame, text="Player 1 Hero:").grid(row=0, column=0, sticky="w", pady=5, padx=5)
        self.hero1_var = tk.StringVar(value="Auto Detect")
        hero1_combo = ttk.Combobox(hero_frame, textvariable=self.hero1_var, 
                                   values=hero_options, state="readonly", width=35)
        hero1_combo.grid(row=0, column=1, padx=5, pady=5)
        hero1_combo.bind("<<ComboboxSelected>>", self._on_hero_changed)
        
        self.hero1_status_var = tk.StringVar(value="")
        self.hero1_status_label = ttk.Label(hero_frame, textvariable=self.hero1_status_var, 
                                            foreground="green", font=("Arial", 8, "italic"))
        self.hero1_status_label.grid(row=0, column=2, sticky="w", padx=10, pady=5)
        
        ttk.Label(hero_frame, text="Player 2 Hero:").grid(row=1, column=0, sticky="w", pady=5, padx=5)
        self.hero2_var = tk.StringVar(value="Auto Detect")
        hero2_combo = ttk.Combobox(hero_frame, textvariable=self.hero2_var, 
                                   values=hero_options, state="readonly", width=35)
        hero2_combo.grid(row=1, column=1, padx=5, pady=5)
        hero2_combo.bind("<<ComboboxSelected>>", self._on_hero_changed)
        
        self.hero2_status_var = tk.StringVar(value="")
        self.hero2_status_label = ttk.Label(hero_frame, textvariable=self.hero2_status_var, 
                                            foreground="green", font=("Arial", 8, "italic"))
        self.hero2_status_label.grid(row=1, column=2, sticky="w", padx=10, pady=5)
        
        # Reset button for hero auto-detection
        reset_hero_btn = ttk.Button(hero_frame, text="Reset Auto-Detection (F9)", command=self._reset_hero_detection)
        reset_hero_btn.grid(row=2, column=1, pady=10)
        
        # Monitor selection
        monitor_frame = ttk.LabelFrame(scrollable_frame, text="Monitor Configuration", padding=10)
        monitor_frame.pack(fill="x", padx=20, pady=10)
        
        ttk.Label(monitor_frame, text="Capture Monitor:").grid(row=0, column=0, sticky="w", pady=5)
        self.capture_monitor = tk.IntVar(value=1)
        ttk.Spinbox(monitor_frame, from_=1, to=4, textvariable=self.capture_monitor, 
                   width=10).grid(row=0, column=1, sticky="w", padx=5)
        
        ttk.Label(monitor_frame, text="Display Monitor:").grid(row=1, column=0, sticky="w", pady=5)
        self.display_monitor = tk.IntVar(value=1)
        ttk.Spinbox(monitor_frame, from_=1, to=4, textvariable=self.display_monitor, 
                   width=10).grid(row=1, column=1, sticky="w", padx=5)
        
        ttk.Label(monitor_frame, text="(Monitor 1 = Primary, 2 = Secondary, etc.)", 
                 font=("Arial", 8, "italic")).grid(row=2, column=0, columnspan=2, sticky="w", pady=2)
        
        # Detection settings
        settings_frame = ttk.LabelFrame(scrollable_frame, text="Detection Settings (Adjustable During Detection)", padding=10)
        settings_frame.pack(fill="x", padx=20, pady=10)
        
        ttk.Label(settings_frame, text="Confidence Threshold:").grid(row=0, column=0, sticky="w", pady=5)
        self.conf_threshold = tk.DoubleVar(value=0.69)
        self.conf_scale = tk.Scale(settings_frame, from_=0.0, to=1.0, resolution=0.01,
                              variable=self.conf_threshold, 
                              orient="horizontal", length=200, command=self._update_conf_threshold,
                              bg='#f0f0f0', troughcolor='#d0d0d0', highlightthickness=0)
        self.conf_scale.grid(row=0, column=1, padx=5)
        self.conf_label = ttk.Label(settings_frame, text="0.69")
        self.conf_label.grid(row=0, column=2, padx=5)
        
        ttk.Label(settings_frame, text="IOU Threshold (removes duplicate boxes):").grid(row=1, column=0, sticky="w", pady=5)
        self.iou_threshold = tk.DoubleVar(value=0.50)
        self.iou_scale = tk.Scale(settings_frame, from_=0.0, to=1.0, resolution=0.01,
                             variable=self.iou_threshold, 
                             orient="horizontal", length=200, command=self._update_iou_threshold,
                             bg='#f0f0f0', troughcolor='#d0d0d0', highlightthickness=0)
        self.iou_scale.grid(row=1, column=1, padx=5)
        self.iou_label = ttk.Label(settings_frame, text="0.50")
        self.iou_label.grid(row=1, column=2, padx=5)
        
        # Overlay settings
        overlay_frame = ttk.LabelFrame(scrollable_frame, text="Display Settings", padding=10)
        overlay_frame.pack(fill="x", padx=20, pady=10)
        
        self.show_bboxes = tk.BooleanVar(value=False)
        ttk.Checkbutton(overlay_frame, text="Display bounding boxes (both modes)", 
                       variable=self.show_bboxes, command=self._toggle_bboxes).pack(anchor="w", pady=2)
        
        self.click_through = tk.BooleanVar(value=False)
        self.click_through_cb = ttk.Checkbutton(overlay_frame, 
                       text="Click-through mode (Overlay Mode: disable to move window, enable to click through)", 
                       variable=self.click_through,
                       command=self._toggle_click_through)
        self.click_through_cb.pack(anchor="w", pady=2)
        
        self.show_card_preview = tk.BooleanVar(value=True)
        ttk.Checkbutton(overlay_frame, text="Show card preview on hover", 
                       variable=self.show_card_preview).pack(anchor="w", pady=2)
        
        self.enable_weight_adjustment = tk.BooleanVar(value=True)
        ttk.Checkbutton(overlay_frame, text="Active Hero Weight Adjustment (boost hero-specific cards)", 
                       variable=self.enable_weight_adjustment).pack(anchor="w", pady=2)
        
        ttk.Label(overlay_frame, text="Manual Card Preview Position:").pack(anchor="w", pady=(5, 0))
        preview_pos_frame = ttk.Frame(overlay_frame)
        preview_pos_frame.pack(anchor="w", padx=20)
        self.manual_preview_side = tk.StringVar(value="left")
        ttk.Radiobutton(preview_pos_frame, text="Left", variable=self.manual_preview_side, 
                       value="left").pack(side="left", padx=5)
        ttk.Radiobutton(preview_pos_frame, text="Right", variable=self.manual_preview_side, 
                       value="right").pack(side="left", padx=5)
        
        ttk.Label(overlay_frame, text="Card Preview Size:").pack(anchor="w", pady=5)
        size_frame = ttk.Frame(overlay_frame)
        size_frame.pack(anchor="w", padx=20)
        self.card_width = tk.IntVar(value=294)
        self.card_height = tk.IntVar(value=408)
        ttk.Label(size_frame, text="Width:").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(size_frame, from_=100, to=800, textvariable=self.card_width, 
                   width=10).grid(row=0, column=1, padx=5)
        ttk.Label(size_frame, text="Height:").grid(row=0, column=2, sticky="w", padx=(20, 0))
        ttk.Spinbox(size_frame, from_=100, to=1000, textvariable=self.card_height, 
                   width=10).grid(row=0, column=3, padx=5)
        
        # Detection History Log
        history_frame = ttk.LabelFrame(scrollable_frame, text="Detection History (Last 2 Minutes)", padding=10)
        history_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # History listbox with scrollbar
        history_scroll = ttk.Scrollbar(history_frame)
        history_scroll.pack(side="right", fill="y")
        
        self.history_listbox = tk.Listbox(history_frame, yscrollcommand=history_scroll.set, 
                                          height=10, font=("Courier", 9))
        self.history_listbox.pack(side="left", fill="both", expand=True)
        history_scroll.config(command=self.history_listbox.yview)
        
        # Bind click event and hover event
        self.history_listbox.bind('<<ListboxSelect>>', self._on_history_select)
        self.history_listbox.bind('<Motion>', self._on_history_hover)
        self.history_listbox.bind('<Leave>', self._on_history_leave)
        
        # Enable mouse wheel scrolling for the listbox
        self.history_listbox.bind('<MouseWheel>', self._on_listbox_scroll)
        self.history_listbox.bind('<Button-4>', self._on_listbox_scroll)  # Linux scroll up
        self.history_listbox.bind('<Button-5>', self._on_listbox_scroll)  # Linux scroll down
        
        # Card Preview Panel
        preview_frame = ttk.LabelFrame(scrollable_frame, text="Selected Card Preview", padding=10)
        preview_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        self.preview_label = ttk.Label(preview_frame, text="Click a card in history to preview", 
                                      font=("Arial", 10, "italic"))
        self.preview_label.pack(pady=20)
        
        # Model Configuration (at bottom)
        model_frame = ttk.LabelFrame(scrollable_frame, text="Model Configuration", padding=10)
        model_frame.pack(fill="x", padx=20, pady=10)
        
        ttk.Label(model_frame, text="Model Weights:").grid(row=0, column=0, sticky="w", pady=5)
        # Default to models/best.pt for packaged version, fallback to training path
        default_model = "models/best.pt" if os.path.exists("models/best.pt") else "runs/train/phase1_100classes/weights/best.pt"
        self.model_path = tk.StringVar(value=default_model)
        model_entry = ttk.Entry(model_frame, textvariable=self.model_path, width=40)
        model_entry.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(model_frame, text="Browse", command=self._browse_model).grid(row=0, column=2, padx=5)
        
        # Status
        self.status_label = ttk.Label(scrollable_frame, text="Ready", 
                                     font=("Arial", 10, "bold"), foreground="green")
        self.status_label.pack(pady=10)
        
        # Initialize history tracking
        self.detection_history = []  # List of (first_timestamp, card_name, total_seconds, avg_confidence) tuples
        self.card_accumulation = {}  # Dict of card_name -> list of (timestamp, confidence) tuples
        self.detection_queue = queue.Queue()  # Thread-safe queue for (card_name, confidence) tuples
        self.locked_preview = None  # Card name that's locked (clicked)
        self.active_window_name = None  # Track the active CV window name for runtime toggles
        self.topmost_timer_active = False  # Track if topmost timer is running
        
        # Start processing detection queue
        self._process_detection_queue()
    
    def _toggle_click_through(self):
        """Toggle click-through mode while detection is running."""
        if self.detector_running and self.mode.get() == "overlay":
            # Update the detector's click-through setting in real-time
            if hasattr(self, 'detector_instance') and self.detector_instance:
                # Sync the window name if available
                if hasattr(self.detector_instance, 'window_name'):
                    self.active_window_name = self.detector_instance.window_name
                
                # Update args
                self.detector_instance.args.click_through = self.click_through.get()
                
                # Reapply the transparency with new click-through setting
                if self.active_window_name:
                    try:
                        # Get chroma color
                        chroma_rgb = tuple(self.detector_instance.args.chroma) if hasattr(self.detector_instance.args, 'chroma') else (255, 0, 255)
                        
                        # Reapply transparency
                        self.detector_instance._enable_chromakey_transparency(
                            self.active_window_name,
                            rgb=chroma_rgb,
                            click_through=self.click_through.get()
                        )
                        print(f"Click-through mode {'enabled' if self.click_through.get() else 'disabled'}")
                    except Exception as e:
                        print(f"Failed to toggle click-through: {e}")
    
    def _update_conf_threshold(self, value):
        """Update confidence threshold in real-time."""
        self.conf_label.config(text=f"{float(value):.2f}")
        if self.detector_running and hasattr(self, 'detector_instance') and self.detector_instance:
            self.detector_instance.args.conf = float(value)
            print(f"Confidence threshold updated to {float(value):.2f}")
    
    def _update_iou_threshold(self, value):
        """Update IOU threshold in real-time."""
        self.iou_label.config(text=f"{float(value):.2f}")
        if self.detector_running and hasattr(self, 'detector_instance') and self.detector_instance:
            self.detector_instance.args.iou = float(value)
            print(f"IOU threshold updated to {float(value):.2f}")
    
    def _toggle_bboxes(self):
        """Toggle bounding box visibility in real-time."""
        if self.detector_running and hasattr(self, 'detector_instance') and self.detector_instance:
            self.detector_instance.args.show_bboxes = self.show_bboxes.get()
            print(f"Bounding boxes {'enabled' if self.show_bboxes.get() else 'disabled'}")
    
    def _scale_click(self, event, scale_widget, var, min_val, max_val):
        """Handle slider click to snap to mouse position instead of jumping."""
        # Get the widget width
        widget_width = scale_widget.winfo_width()
        if widget_width <= 1:
            return
        
        # Calculate the value based on click position
        click_pos = event.x
        value_range = max_val - min_val
        value = min_val + (click_pos / widget_width) * value_range
        
        # Clamp to min/max
        value = max(min_val, min(max_val, value))
        
        # Set the variable
        var.set(value)
    
    def _load_hero_names(self):
        """Load list of all hero names from heroes_card.json."""
        hero_json_path = CardDetector._find_data_file('heroes_card.json', ['data'])
        
        if hero_json_path:
            try:
                with open(hero_json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    adult_heroes = [h['name'] for h in data.get('adult_heroes', [])]
                    young_heroes = [h['name'] for h in data.get('young_heroes', [])]
                    all_heroes = sorted(set(adult_heroes + young_heroes))
                    print(f"[init] Loaded {len(all_heroes)} heroes from {hero_json_path}")
                    return all_heroes
            except Exception as e:
                print(f"[warning] Could not load heroes from {hero_json_path}: {e}")
        
        print("[warning] heroes_card.json not found - hero override will not work")
        return []
    
    def _on_hero_changed(self, event=None):
        """Called when hero dropdown selection changes."""
        if hasattr(self, 'detector_instance') and self.detector_instance:
            hero1 = None if self.hero1_var.get() == "Auto Detect" else self.hero1_var.get()
            hero2 = None if self.hero2_var.get() == "Auto Detect" else self.hero2_var.get()
            self.detector_instance.set_hero_override(hero1, hero2)
    
    def _reset_hero_detection(self):
        """Reset hero auto-detection to start fresh."""
        if hasattr(self, 'detector_instance') and self.detector_instance:
            self.detector_instance.reset_hero_detection()
            # Only update GUI if it exists
            if hasattr(self, 'hero1_var'):
                self.hero1_var.set("Auto Detect")
                self.hero2_var.set("Auto Detect")
                self.hero1_status_var.set("")
                self.hero2_status_var.set("")
            print("[hero] Reset auto-detection")
        
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
                # Update labels to match loaded values
                self.conf_label.config(text=f"{self.conf_threshold.get():.2f}")
                self.iou_label.config(text=f"{self.iou_threshold.get():.2f}")
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
        # Get current model path directory as initial directory
        current_path = self.model_path.get()
        initial_dir = os.path.dirname(current_path) if current_path and os.path.exists(current_path) else os.getcwd()
        
        filename = filedialog.askopenfilename(
            title="Select Model Weights",
            initialdir=initial_dir,
            filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")]
        )
        if filename:
            self.model_path.set(filename)
    
    def _process_detection_queue(self):
        """Process detection updates from the detector thread."""
        try:
            current_time = time.time()
            
            # Process all pending detections
            while not self.detection_queue.empty():
                card_name, confidence = self.detection_queue.get_nowait()
                
                # Track accumulation time and confidence for each card
                if card_name not in self.card_accumulation:
                    self.card_accumulation[card_name] = []
                self.card_accumulation[card_name].append((current_time, confidence))
            
            # Clean old timestamps (older than 2 minutes)
            cutoff_time = current_time - 120  # 120 seconds = 2 minutes
            for card_name in list(self.card_accumulation.keys()):
                # Remove old timestamps
                self.card_accumulation[card_name] = [
                    (t, conf) for t, conf in self.card_accumulation[card_name] if t > cutoff_time
                ]
                
                # Remove card if no recent detections
                if not self.card_accumulation[card_name]:
                    del self.card_accumulation[card_name]
            
            # Calculate total active time and average confidence for each card
            cards_meeting_threshold = []
            for card_name, detections in self.card_accumulation.items():
                # Each detection represents approximately 0.1 seconds of visibility
                total_seconds = len(detections) * 0.1
                
                # Only include cards with at least 3 seconds of active time
                if total_seconds >= 3.0:
                    # Find first timestamp and calculate average confidence
                    first_timestamp = min(t for t, conf in detections)
                    avg_confidence = sum(conf for t, conf in detections) / len(detections)
                    cards_meeting_threshold.append((first_timestamp, card_name, total_seconds, avg_confidence))
            
            # Sort by card name alphabetically (case-insensitive)
            cards_meeting_threshold.sort(key=lambda x: x[1].lower())
            
            # Check if we need to update the display
            if cards_meeting_threshold != self.detection_history:
                self.detection_history = cards_meeting_threshold
                self._rebuild_history_display()
            
            # Update hero status display
            self._update_hero_status()
                
        except Exception as e:
            print(f"Error processing detection queue: {e}")
        
        # Schedule next check (every 100ms)
        self.root.after(100, self._process_detection_queue)
    
    def _update_hero_status(self):
        """Update the hero status labels with currently detected heroes."""
        if hasattr(self, 'detector_instance') and self.detector_instance:
            # Update Hero 1 status
            if self.detector_instance.detected_hero1:
                if self.detector_instance.hero1_override:
                    self.hero1_status_var.set("(Manual Override)")
                else:
                    self.hero1_status_var.set(f"Currently Detecting: {self.detector_instance.detected_hero1}")
            else:
                self.hero1_status_var.set("")
            
            # Update Hero 2 status
            if self.detector_instance.detected_hero2:
                if self.detector_instance.hero2_override:
                    self.hero2_status_var.set("(Manual Override)")
                else:
                    self.hero2_status_var.set(f"Currently Detecting: {self.detector_instance.detected_hero2}")
            else:
                self.hero2_status_var.set("")
    
    def _rebuild_history_display(self):
        """Rebuild the entire history listbox without auto-scrolling."""
        # Save current scroll position
        first_visible = self.history_listbox.yview()[0]
        
        # Clear and rebuild
        self.history_listbox.delete(0, tk.END)
        
        # Add all entries with confidence scores
        for first_timestamp, card_name, total_seconds, avg_confidence in self.detection_history:
            display_text = f"{card_name} ({avg_confidence*100:.1f}%)"
            self.history_listbox.insert(tk.END, display_text)
        
        # Restore scroll position (don't auto-scroll to bottom)
        self.history_listbox.yview_moveto(first_visible)
    
    def _on_history_select(self, event):
        """Handle clicking on a history entry to lock the card preview."""
        selection = self.history_listbox.curselection()
        if not selection:
            return
        
        # Get selected text - format is "Card Name (XX.X%)"
        selected_text = self.history_listbox.get(selection[0])
        # Extract card name before the confidence percentage
        card_name = selected_text.rsplit(" (", 1)[0] if " (" in selected_text else selected_text
        
        print(f"Locked card: '{card_name}'")
        
        # Lock this card
        self.locked_preview = card_name
        
        # Load and display card
        self._show_card_preview(card_name)
    
    def _on_history_hover(self, event):
        """Handle hovering over a history entry to preview card (if not locked)."""
        # If a card is locked, don't change preview on hover
        if self.locked_preview:
            return
        
        # Get the index of the item under the mouse
        index = self.history_listbox.nearest(event.y)
        if index >= 0 and index < self.history_listbox.size():
            # Get text and extract card name
            selected_text = self.history_listbox.get(index)
            card_name = selected_text.rsplit(" (", 1)[0] if " (" in selected_text else selected_text
            
            # Load and display card
            self._show_card_preview(card_name)
    
    def _on_history_leave(self, event):
        """Handle mouse leaving the listbox - clear preview if not locked."""
        if not self.locked_preview:
            self.preview_label.config(text="Hover over a card to preview", image="")
            self.preview_label.image = None
    
    def _on_listbox_scroll(self, event):
        """Handle mouse wheel scrolling in the listbox."""
        if event.num == 5 or event.delta < 0:
            # Scroll down
            self.history_listbox.yview_scroll(1, "units")
        elif event.num == 4 or event.delta > 0:
            # Scroll up
            self.history_listbox.yview_scroll(-1, "units")
        # Return "break" to prevent event from propagating to parent
        return "break"
    
    def _show_card_preview(self, card_name):
        """Load and display the selected card in the preview panel."""
        try:
            # Use the same method as the hover feature
            image_url = None
            if hasattr(self, 'detector_thread') and hasattr(self, 'detector_instance'):
                # If detector is running, use its method
                image_url = self.detector_instance.get_image_url_by_name(card_name)
            else:
                # Detector not running, load card data directly
                card_json_path = CardDetector._find_data_file('card.json', ['data'])
                if not card_json_path:
                    self.preview_label.config(text=f"Card database not found", image="")
                    return
                
                with open(card_json_path, 'r', encoding='utf-8') as f:
                    card_data = json.load(f)
                
                # Use same logic as get_image_url_by_name
                search_name = card_name.replace('_', ' ').lower().strip()
                
                for card in card_data:
                    card_display_name = card.get('name', '').lower().strip()
                    if card_display_name == search_name:
                        printings = card.get('printings', [])
                        if printings:
                            image_url = printings[0].get('image_url')
                            break
            
            if not image_url:
                self.preview_label.config(text=f"No image URL for {card_name}", image="")
                print(f"No image URL found for: {card_name}")
                return
            
            # Print URL to console for debugging
            print(f"Loading image from: {image_url}")
            
            # Download and display image
            response = requests.get(image_url, timeout=5)
            if response.status_code == 200:
                # Load image with PIL
                image_data = BytesIO(response.content)
                pil_image = Image.open(image_data)
                
                # Resize to fit preview panel (max 300x450)
                max_width = 300
                max_height = 450
                pil_image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage for Tkinter
                photo = ImageTk.PhotoImage(pil_image)
                
                # Update label
                self.preview_label.config(image=photo, text="")
                self.preview_label.image = photo  # Keep reference to prevent garbage collection
                
                # Print URL to terminal as well
                print(f"Successfully loaded: {card_name}")
                print(f"URL: {image_url}")
            else:
                self.preview_label.config(text=f"Failed to load image for {card_name}", image="")
                print(f"Failed to download image (status {response.status_code})")
                
        except Exception as e:
            print(f"Error showing card preview: {e}")
            import traceback
            traceback.print_exc()
            self.preview_label.config(text=f"Error: {str(e)}", image="")
    
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
        
        # Clear history data and display
        self.detection_history = []
        self.card_accumulation = {}
        self.locked_preview = None
        self.history_listbox.delete(0, tk.END)
        self.preview_label.config(text="Click a card in history to preview", image="")
        self.preview_label.image = None
        
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
                transparent=self.mode.get() == "overlay",  # Always transparent in overlay mode
                chroma=(255, 0, 255),
                click_through=self.click_through.get() and self.mode.get() == "overlay",
                show_card_preview=self.show_card_preview.get(),
                manual_preview_side=self.manual_preview_side.get(),
                show_bboxes=self.show_bboxes.get(),
                hero1_override=None if self.hero1_var.get() == "Auto Detect" else self.hero1_var.get(),
                hero2_override=None if self.hero2_var.get() == "Auto Detect" else self.hero2_var.get(),
                enable_weight_adjustment=self.enable_weight_adjustment.get(),  # Phase 4
                video=None
            )
            
            # Run detector
            detector = CardDetector(args, stop_callback=lambda: not self.detector_running, 
                                   detection_queue=self.detection_queue,
                                   minimize_gui_callback=lambda: self.root.iconify())
            self.detector_instance = detector  # Store reference for GUI to access
            
            # Start detection
            detector.run()
            
        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda msg=error_msg: messagebox.showerror("Detection Error", msg))
        finally:
            self.detector_instance = None  # Clear reference
            self.root.after(0, self._stop_detection)


class CardDetector:
    """Core detection engine."""
    
    def __init__(self, args, stop_callback=None, detection_queue=None, minimize_gui_callback=None):
        self.args = args
        self.stop_callback = stop_callback
        self.detection_queue = detection_queue  # Queue for sending detections to GUI
        self.minimize_gui_callback = minimize_gui_callback  # Callback to minimize GUI window
        
        # Debug: show preview setting
        print(f"[init] show_card_preview setting: {args.show_card_preview}")
        print(f"[init] show_bboxes setting: {args.show_bboxes}")
        
        # Load model
        print(f"[init] Loading model from {args.weights}...")
        self.model = YOLO(args.weights)
        print(f"[init] Model loaded with {len(self.model.names)} classes")
        
        # Load card data
        self.card_data = self._load_card_data()
        self.card_image_cache = {}
        
        # Initialize ConfidenceBooster (Phase 4 feature)
        self.confidence_booster = None
        self.enable_weight_adjustment = getattr(args, 'enable_weight_adjustment', False)  # Store boolean from args
        if ConfidenceBooster:
            try:
                weights_path = self._find_data_file('card_weights_all_printings.json', ['data'])
                if weights_path:
                    self.confidence_booster = ConfidenceBooster(str(weights_path), enabled=True)
                    print(f"[phase4] ConfidenceBooster initialized from {weights_path}")
                else:
                    print(f"[phase4] Card weights file not found")
            except Exception as e:
                print(f"[phase4] Failed to initialize ConfidenceBooster: {e}")
        elif not ConfidenceBooster:
            print("[phase4] ConfidenceBooster module not available")
        
        # Hero detection and legal pool filtering (support 2 heroes)
        self.detected_hero1 = None
        self.detected_hero2 = None
        self.hero1_confidence_history = []  # Track hero 1 detections over time
        self.hero2_confidence_history = []  # Track hero 2 detections over time
        self.legal_card_names = None  # Set of legal card names for both heroes
        
        # Dynamic hero detection threshold - starts at 0.65, lowers by 0.1 every 30 frames until 0.07
        self.hero_detection_threshold = 0.65  # Starting threshold
        self.hero_threshold_min = 0.07  # Minimum threshold
        self.hero_threshold_step = 0.10  # Amount to lower per interval
        self.hero_threshold_frame_interval = 30  # Lower every 30 frames (works at any FPS)
        self.hero_threshold_frame_counter = 0  # Frame counter for threshold lowering
        
        self.hero_min_detections = 3  # Minimum detections required (works even at <1 FPS)
        self.hero_max_history_size = 50  # Maximum history entries to keep
        
        # Manual hero overrides
        self.hero1_override = getattr(args, 'hero1_override', None)
        self.hero2_override = getattr(args, 'hero2_override', None)
        if self.hero1_override:
            self.detected_hero1 = self.hero1_override
            print(f"[hero] Hero 1 override: {self.hero1_override}")
        if self.hero2_override:
            self.detected_hero2 = self.hero2_override
            print(f"[hero] Hero 2 override: {self.hero2_override}")
        
        # Build initial legal pool if both heroes are overridden
        if self.detected_hero1 and self.detected_hero2:
            self._rebuild_legal_pool()
        
        # Timer for topmost Tab key (once per second)
        self.last_topmost_time = 0
        
        # Tracking (no separate preview window)
        self.last_overlay_rect = None
        self.mouse_pos = (0, 0)
        self.fps_history = []
        
        # Stable preview tracking (prevents flicker)
        self.stable_card_name = None
        self.stable_card_image = None
        self.stable_preview_rect = None  # (px1, py1, px2, py2) of preview image
        self.stable_hover_zone = None  # (x1, y1, x2, y2) expanded bbox zone for tracking
        self.preview_last_active_time = 0  # Timestamp when preview was last actively hovered
        self.preview_linger_duration = 0.1  # Keep preview visible for 0.1s after hover ends
        self.hover_zone_margin = 18  # Pixels to expand bbox for stable hover detection
        
        # Manual detection mode (for missed cards)
        self.manual_mode = False  # Whether in manual bbox drawing mode
        self.manual_bbox_start = None  # (x, y) where user started drawing
        self.manual_bbox_current = None  # (x, y) current mouse position while dragging
        self.manual_bbox_complete = None  # (x1, y1, x2, y2) completed bbox
        self.manual_predictions = []  # List of (class_name, confidence) from manual detection
        self.manual_frame_snapshot = None  # Clean frame captured before entering manual mode
        self.drawing_bbox = False  # Whether currently dragging to draw bbox
        self.manual_hover_zone_expiry = 0  # Timestamp when manual hover zone expires (15 seconds after selection)
        
        # Separate manual preview window state (independent of normal detection)
        self.manual_preview_card_name = None
        self.manual_preview_card_image = None
        self.manual_preview_expiry = 0  # Timestamp when manual preview expires
        self.manual_preview_window_name = "Manual Card Selection"
        
        # Text input state for '6' key feature
        self.text_input_mode = False  # Whether in text input mode
        self.text_input_buffer = ""  # Current text being typed
        self.text_input_matches = []  # List of matching card names
        self.text_input_selection_mode = False  # Whether showing numbered list for selection
        self.text_input_suggestion_rects = []  # List of (x1, y1, x2, y2, card_name) for clickable suggestions
        self.text_input_selected_index = -1  # Currently highlighted suggestion index (-1 = none)
        
        # Cache management - limit sizes and periodic cleanup
        self.max_image_cache_size = 100  # Maximum number of cached card images
        self.cache_last_cleanup = time.time()
        self.cache_cleanup_interval = 300  # Clean cache every 5 minutes
        
        # Reset button bounds (x1, y1, x2, y2) - will be set when drawing
        self.reset_button_bounds = None
        
        # Reset button bounds (x1, y1, x2, y2) - will be set when drawing
        self.reset_button_bounds = None
        
        # Reset button bounds (x1, y1, x2, y2) - will be set when drawing
        self.reset_button_bounds = None
    
    @staticmethod
    def _find_data_file(filename, subdirs=['data']):
        """Helper to find data files with PyInstaller support.
        
        Args:
            filename: Name of the file to find (e.g., 'card.json')
            subdirs: List of subdirectory names to check (e.g., ['data'])
            
        Returns:
            Path object if found, None otherwise
        """
        possible_paths = []
        
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        if getattr(sys, 'frozen', False):
            base_path = Path(sys._MEIPASS)
            for subdir in subdirs:
                possible_paths.append(base_path / subdir / filename)
        
        # Development paths
        script_dir = Path(__file__).parent
        for subdir in subdirs:
            possible_paths.extend([
                script_dir / subdir / filename,
                Path(subdir) / filename,
                Path.cwd() / subdir / filename
            ])
        
        for path in possible_paths:
            if path.exists():
                return path
        
        return None
    
    def _on_preview_click(self, direction):
        """Handle click on preview window - cycle alternatives."""
        print(f"[preview] Click detected, direction={direction}")
        self._cycle_alternative_detection(direction)
        
    def _load_card_data(self):
        """Load card metadata."""
        card_json_path = self._find_data_file('card.json', ['data'])
        
        if card_json_path:
            try:
                with open(card_json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"[init] Loaded {len(data)} cards from {card_json_path}")
                    return data
            except Exception as e:
                print(f"[warning] Could not load {card_json_path}: {e}")
        
        print("[warning] card.json not found - card preview will not work")
        return []
    
    def _is_hero_card(self, card_name):
        """Check if a detected card is a hero."""
        search_name = card_name.replace('_', ' ').lower().strip()
        
        for card in self.card_data:
            card_display_name = card.get('name', '').lower().strip()
            if card_display_name == search_name:
                types = card.get('types', [])
                return 'Hero' in types or 'Young Hero' in types
        return False
    
    def _build_legal_card_pool(self, hero_name):
        """Build set of legal card names for a given hero."""
        # Find hero card
        search_name = hero_name.replace('_', ' ').lower().strip()
        hero_card = None
        
        for card in self.card_data:
            if card.get('name', '').lower().strip() == search_name:
                hero_card = card
                break
        
        if not hero_card:
            print(f"[hero] Could not find hero: {hero_name}")
            return set()
        
        # Define all classes and talents
        ALL_CLASSES = {'Warrior', 'Brute', 'Guardian', 'Ninja', 'Ranger', 'Assassin', 
                      'Wizard', 'Runeblade', 'Mechanologist', 'Merchant', 'Illusionist', 
                      'Shapeshifter', 'Bard', 'Mystic', 'Pirate', 'Necromancer', 'Thief', 
                      'Adjudicator'}
        ALL_TALENTS = {'Draconic', 'Elemental', 'Light', 'Shadow', 'Earth', 'Ice', 
                      'Lightning', 'Royal', 'Chi', 'Chaos', 'Arcane', 'Revered', 'Reviled'}
        
        # Extract hero's classes and talents from types
        hero_types = set(hero_card.get('types', []))
        hero_classes = hero_types & ALL_CLASSES
        hero_talents = hero_types & ALL_TALENTS
        
        # Check for "Essence of X" keywords that grant additional talent access
        card_keywords = hero_card.get('card_keywords', [])
        for keyword in card_keywords:
            if 'Essence of' in keyword:
                essence_part = keyword.replace('Essence of', '').strip()
                for separator in [', and ', ' and ', ',']:
                    essence_part = essence_part.replace(separator, '|')
                essence_talents = [t.strip() for t in essence_part.split('|')]
                for talent in essence_talents:
                    if talent in ALL_TALENTS:
                        hero_talents.add(talent)
        
        print(f"[hero] Building legal pool for {hero_card['name']}")
        print(f"[hero] Classes: {hero_classes}, Talents: {hero_talents}")
        
        # Build legal pool
        legal_cards = set()
        
        for card in self.card_data:
            card_name = card.get('name', '')
            card_types = set(card.get('types', []))
            card_classes = card_types & ALL_CLASSES
            card_talents = card_types & ALL_TALENTS
            
            # Skip heroes
            if 'Hero' in card_types or 'Young Hero' in card_types:
                continue
            
            # Generic cards and tokens are always legal
            if 'Generic' in card_types or 'Token' in card_types:
                legal_cards.add(card_name.lower())
                continue
            
            # If hero has no talents, exclude cards with ANY talent
            if not hero_talents and card_talents:
                continue
            
            # Check if card matches hero
            # If card has classes, at least ONE must match hero's classes (OR logic)
            # If card has talents, at least ONE must match hero's talents (OR logic)
            card_classes_match = not card_classes or bool(card_classes & hero_classes)
            card_talents_match = not card_talents or bool(card_talents & hero_talents)
            
            # Skip card if any class or talent doesn't match
            if not card_classes_match or not card_talents_match:
                continue
            
            # Card is legal!
            legal_cards.add(card_name.lower())
        
        print(f"[hero] Legal pool size for {hero_name}: {len(legal_cards)} cards")
        return legal_cards
    
    def _rebuild_legal_pool(self):
        """Rebuild the legal card pool based on detected heroes."""
        if not self.detected_hero1 and not self.detected_hero2:
            self.legal_card_names = None
            return
        
        # Build pools for each hero
        pool1 = self._build_legal_card_pool(self.detected_hero1) if self.detected_hero1 else set()
        pool2 = self._build_legal_card_pool(self.detected_hero2) if self.detected_hero2 else set()
        
        # Union of both pools (card is legal if legal for either hero)
        self.legal_card_names = pool1 | pool2
        print(f"[hero] Combined legal pool: {len(self.legal_card_names)} cards")
    
    def _get_legal_class_ids(self):
        """Get list of legal class IDs for pre-detection filtering (Phase 4 feature)."""
        # Only apply filtering if we have detected heroes AND have a legal card pool
        if not self.detected_hero1 and not self.detected_hero2:
            print(f"[phase4 debug] _get_legal_class_ids: No heroes detected (hero1={self.detected_hero1}, hero2={self.detected_hero2}), returning None")
            return None  # No heroes detected yet, allow all detections
        
        if not self.legal_card_names:
            print(f"[phase4 debug] _get_legal_class_ids: No legal pool (legal_card_names={self.legal_card_names}), returning None")
            return None  # No legal pool built yet
        
        print(f"[phase4 debug] _get_legal_class_ids: Building legal class IDs for heroes {self.detected_hero1}, {self.detected_hero2} from {len(self.legal_card_names)} legal cards")
        
        # Build set of legal class IDs from legal card names
        legal_class_ids = []
        for class_id, class_name in self.model.names.items():
            # Normalize class name to match card names
            normalized_name = class_name.lower().strip()
            if normalized_name in self.legal_card_names:
                legal_class_ids.append(class_id)
        
        if legal_class_ids:
            print(f"[phase4] Pre-filtering to {len(legal_class_ids)} legal class IDs for heroes: {self.detected_hero1}, {self.detected_hero2}")
        else:
            print(f"[phase4 debug] WARNING: No legal class IDs found even though we have heroes!")
        return legal_class_ids if legal_class_ids else None
    
    def set_hero_override(self, hero1, hero2):
        """Set hero overrides from GUI."""
        print(f"[hero] Override changed: Hero1={hero1}, Hero2={hero2}")
        self.hero1_override = hero1
        self.hero2_override = hero2
        
        # Clear auto-detected heroes if overridden
        if hero1:
            self.detected_hero1 = hero1
            self.hero1_confidence_history = []
        else:
            self.detected_hero1 = None
        
        if hero2:
            self.detected_hero2 = hero2
            self.hero2_confidence_history = []
        else:
            self.detected_hero2 = None
        
        # Rebuild legal pool
        self._rebuild_legal_pool()
    
    def reset_hero_detection(self):
        """Reset hero auto-detection to start fresh."""
        self.detected_hero1 = None
        self.detected_hero2 = None
        self.hero1_confidence_history = []
        self.hero2_confidence_history = []
        self.legal_card_names = None
        self.hero1_override = None
        self.hero2_override = None
        # Reset dynamic threshold
        self.hero_detection_threshold = 0.65
        self.hero_threshold_frame_counter = 0
        print("[hero] Detection reset - starting fresh with threshold 0.65")
    
    def _apply_hero_filtering(self, boxes, clss, confs, names, current_time):
        """Apply hero-based filtering and confidence boosting for two heroes."""
        if len(boxes) == 0:
            return boxes, clss, confs
        
        # Dynamic threshold adjustment: Lower threshold every N frames until both heroes detected or min reached
        if (self.detected_hero1 is None or self.detected_hero2 is None):
            self.hero_threshold_frame_counter += 1
            if self.hero_threshold_frame_counter >= self.hero_threshold_frame_interval:
                if self.hero_detection_threshold > self.hero_threshold_min:
                    self.hero_detection_threshold = max(
                        self.hero_threshold_min,
                        self.hero_detection_threshold - self.hero_threshold_step
                    )
                    print(f"[hero] Lowering detection threshold to {self.hero_detection_threshold:.2f} (frame {self.hero_threshold_frame_counter})")
                    self.hero_threshold_frame_counter = 0
        else:
            # Both heroes detected, reset counter
            self.hero_threshold_frame_counter = 0
        
        # Step 1: Check for hero detections (only if not overridden)
        for i, cls_idx in enumerate(clss):
            class_id = int(cls_idx)
            if class_id < len(names):
                card_name = names[class_id]
                confidence = float(confs[i])
                
                # Check if this is a hero card
                if self._is_hero_card(card_name):
                    if confidence >= self.hero_detection_threshold:
                        # Add to appropriate hero history
                        # Strategy: Hero1 gets priority - fill hero1 first, then hero2 with different hero
                        
                        if not self.hero1_override:
                            if not self.detected_hero1:
                                # Hero1 not set yet - add this hero to hero1 history ONLY
                                self.hero1_confidence_history.append((current_time, card_name, confidence))
                            elif self.detected_hero1 == card_name:
                                # Hero1 already set and this matches - keep confirming hero1
                                self.hero1_confidence_history.append((current_time, card_name, confidence))
                            elif not self.hero2_override and card_name != self.detected_hero1:
                                # Hero1 is set but this is a DIFFERENT hero - add to hero2 history
                                self.hero2_confidence_history.append((current_time, card_name, confidence))
                        
                        # If hero1 override is active, we can still detect hero2
                        elif not self.hero2_override:
                            if self.hero1_override and card_name != self.detected_hero1:
                                # Hero1 is manually set, this is different - add to hero2
                                self.hero2_confidence_history.append((current_time, card_name, confidence))
                    else:
                        # Debug: Hero detected but confidence too low
                        print(f"[hero] {card_name} detected but confidence {confidence:.2f} < threshold {self.hero_detection_threshold:.2f}")
        
        # Step 2: Process hero1 history
        if not self.hero1_override:
            # Limit history size (keep most recent entries)
            if len(self.hero1_confidence_history) > self.hero_max_history_size:
                self.hero1_confidence_history = self.hero1_confidence_history[-self.hero_max_history_size:]
            
            # Check if we have consistent hero detection (just need minimum count)
            if len(self.hero1_confidence_history) >= self.hero_min_detections and not self.detected_hero1:
                from collections import Counter
                hero_names = [name for _, name, _ in self.hero1_confidence_history]
                most_common_hero = Counter(hero_names).most_common(1)[0][0]
                self.detected_hero1 = most_common_hero
                print(f"[hero] Detected Hero 1: {most_common_hero}")
                self._rebuild_legal_pool()
        
        # Step 3: Process hero2 history
        if not self.hero2_override:
            # Limit history size (keep most recent entries)
            if len(self.hero2_confidence_history) > self.hero_max_history_size:
                self.hero2_confidence_history = self.hero2_confidence_history[-self.hero_max_history_size:]
            
            # Check if we have consistent hero detection (just need minimum count)
            if len(self.hero2_confidence_history) >= self.hero_min_detections and not self.detected_hero2:
                from collections import Counter
                hero_names = [name for _, name, _ in self.hero2_confidence_history]
                most_common_hero = Counter(hero_names).most_common(1)[0][0]
                self.detected_hero2 = most_common_hero
                print(f"[hero] Detected Hero 2: {most_common_hero}")
                self._rebuild_legal_pool()
        
        # Step 4: Apply confidence boosting/penalizing if at least one hero is detected
        if self.legal_card_names is not None:
            new_confs = []
            keep_indices = []
            
            for i, cls_idx in enumerate(clss):
                class_id = int(cls_idx)
                if class_id < len(names):
                    card_name = names[class_id]
                    confidence = float(confs[i])
                    
                    # Skip hero cards from filtering
                    if self._is_hero_card(card_name):
                        new_confs.append(confidence)
                        keep_indices.append(i)
                        continue
                    
                    # Check if card is legal
                    card_name_lower = card_name.replace('_', ' ').lower().strip()
                    
                    if card_name_lower in self.legal_card_names:
                        # Boost legal cards
                        boosted_conf = min(confidence * 1.2, 1.0)
                        new_confs.append(boosted_conf)
                        keep_indices.append(i)
                    # else: completely filter out illegal cards (no penalty, just block)
            
            # Apply filtering
            if len(keep_indices) < len(boxes):
                filtered_count = len(boxes) - len(keep_indices)
                boxes = boxes[keep_indices]
                clss = clss[keep_indices]
                confs = np.array(new_confs)
                if filtered_count > 0:
                    print(f"[hero] Filtered {filtered_count} illegal cards")
            else:
                confs = np.array(new_confs)
        
        return boxes, clss, confs
    
    def _draw_text_with_border(self, frame, text, pos, font, scale, color, thickness, border_color=(0, 0, 0), border_thickness=None):
        """Draw text with a black border for better readability.
        
        Args:
            frame: The frame to draw on
            text: The text string to draw
            pos: (x, y) position tuple
            font: cv2 font type
            scale: Font scale
            color: Text color (B, G, R)
            thickness: Text thickness
            border_color: Border color (default black)
            border_thickness: Border thickness (default: thickness + 1)
        """
        if border_thickness is None:
            border_thickness = thickness + 1
        
        # Draw black border first (thicker)
        cv2.putText(frame, text, pos, font, scale, border_color, border_thickness, cv2.LINE_AA)
        # Draw actual text on top
        cv2.putText(frame, text, pos, font, scale, color, thickness, cv2.LINE_AA)
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for CV2 window."""
        # Check for reset button click first (highest priority)
        if event == cv2.EVENT_LBUTTONDOWN and self.reset_button_bounds:
            bx1, by1, bx2, by2 = self.reset_button_bounds
            if bx1 <= x <= bx2 and by1 <= y <= by2:
                print("[button] Reset button clicked - resetting hero detection")
                self.reset_hero_detection()
                return
        
        # Text input mode - handle clicking on suggestions
        if self.text_input_mode and event == cv2.EVENT_LBUTTONDOWN:
            # Check if click is within any suggestion rectangle
            for rect in self.text_input_suggestion_rects:
                x1, y1, x2, y2, card_name = rect
                if x1 <= x <= x2 and y1 <= y <= y2:
                    print(f"[manual] Clicked suggestion: {card_name}")
                    # Get card dimensions from args
                    card_w, card_h = self.args.card_size
                    # Create a dummy display frame for the callback (won't be used)
                    display_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
                    self._apply_manual_card_selection(card_name, display_frame, card_w, card_h)
                    return
        
        # Manual detection mode - handle bbox drawing
        if self.manual_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                # Start drawing bbox
                self.manual_bbox_start = (x, y)
                self.drawing_bbox = True
                print(f"[manual] Started bbox at ({x}, {y})")
            
            elif event == cv2.EVENT_MOUSEMOVE and self.drawing_bbox:
                # Update current bbox position
                self.manual_bbox_current = (x, y)
            
            elif event == cv2.EVENT_LBUTTONUP and self.drawing_bbox:
                # Complete bbox drawing
                self.drawing_bbox = False
                if self.manual_bbox_start:
                    x1 = min(self.manual_bbox_start[0], x)
                    y1 = min(self.manual_bbox_start[1], y)
                    x2 = max(self.manual_bbox_start[0], x)
                    y2 = max(self.manual_bbox_start[1], y)
                    self.manual_bbox_complete = (x1, y1, x2, y2)
                    print(f"[manual] Completed bbox: ({x1}, {y1}, {x2}, {y2})")
            return
        
        # Normal mode - update mouse position on move
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_pos = (x, y)
        
        # Handle mouse wheel for confidence adjustment
        elif event == cv2.EVENT_MOUSEWHEEL:
            # flags > 0 = scroll up, flags < 0 = scroll down
            if flags > 0:
                # Scroll up - increase confidence (more strict)
                self.args.conf = min(0.95, self.args.conf + 0.01)
                print(f"[confidence] Increased to {self.args.conf:.2f}")
            else:
                # Scroll down - decrease confidence (less strict)
                self.args.conf = max(0.05, self.args.conf - 0.01)
                print(f"[confidence] Decreased to {self.args.conf:.2f}")
    
    def _run_manual_detection(self, frame, bbox):
        """Run detection on manually drawn bbox with low confidence threshold.
        
        Uses masking instead of cropping - blacks out everything except the bbox region.
        This preserves the full frame context that the model was trained on.
        
        Args:
            frame: The captured frame to analyze
            bbox: (x1, y1, x2, y2) bounding box coordinates
            
        Returns:
            List of (class_name, confidence) tuples for top predictions
        """
        x1, y1, x2, y2 = bbox
        
        # Ensure bbox is within frame bounds
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        if x2 <= x1 or y2 <= y1:
            print(f"[manual] Invalid bbox dimensions")
            return []
        
        # Create masked frame: black out everything except bbox region
        masked_frame = np.zeros_like(frame)
        masked_frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2].copy()
        
        # Save masked frame for debugging
        try:
            debug_path = "debug_manual_mask.png"
            cv2.imwrite(debug_path, masked_frame)
            print(f"[manual] Saved masked frame to {debug_path}: bbox=({x1},{y1},{x2},{y2})")
        except Exception as e:
            print(f"[manual] Could not save debug mask: {e}")
        
        print(f"[manual] Running detection on masked frame (bbox: {x1},{y1} to {x2},{y2}) with conf=0.001")
        
        # Run YOLO on masked frame with extremely low confidence
        try:
            results = self.model(masked_frame, conf=0.001, verbose=False)
            
            if not results or len(results) == 0:
                print(f"[manual] No results from model")
                return []
            
            result = results[0]
            
            if result.boxes is None or len(result.boxes) == 0:
                print(f"[manual] No detections in masked frame")
                return []
            
            # Extract predictions - only keep detections whose centers are inside the bbox
            predictions = []
            boxes_data = result.boxes
            
            for i in range(len(boxes_data)):
                cls_idx = int(boxes_data.cls[i])
                conf = float(boxes_data.conf[i])
                
                # Get detection box coordinates
                box = boxes_data.xyxy[i]
                det_x1, det_y1, det_x2, det_y2 = [float(v) for v in box]
                
                # Calculate center of detection
                center_x = (det_x1 + det_x2) / 2
                center_y = (det_y1 + det_y2) / 2
                
                # Only keep if center is inside the bbox region
                if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                    if cls_idx < len(result.names):
                        class_name = result.names[cls_idx]
                        predictions.append((class_name, conf))
                        print(f"[manual] Detection inside bbox: {class_name} ({conf:.3f}) at ({center_x:.0f},{center_y:.0f})")
            
            # Sort by confidence and return top 5
            predictions.sort(key=lambda x: x[1], reverse=True)
            top_predictions = predictions[:5]
            
            print(f"[manual] Found {len(predictions)} predictions inside bbox, top 5:")
            for i, (name, conf) in enumerate(top_predictions, 1):
                print(f"  {i}. {name} ({conf:.3f})")
            
            return top_predictions
            
        except Exception as e:
            print(f"[manual] Error during detection: {e}")
            return []
    
    def _apply_manual_card_selection(self, selected_card, display_frame, card_w, card_h):
        """Apply manual card selection and show preview on main display."""
        image_url = self.get_image_url_by_name(selected_card)
        if image_url:
            card_img = self.get_card_image(image_url)
            if card_img:
                # Store for manual preview (separate from normal hover preview)
                self.manual_preview_card_name = selected_card
                self.manual_preview_card_image = card_img
                self.manual_preview_expiry = time.time() + 15.0
                print(f"[manual] Created manual preview for 15 seconds: {selected_card}")
        
        # Exit manual mode and restore transparency
        self.manual_mode = False
        self.manual_bbox_start = None
        self.manual_bbox_current = None
        self.manual_bbox_complete = None
        self.manual_predictions = []
        self.manual_frame_snapshot = None
    
    def get_image_url_by_name(self, card_name):
        """Get card image URL by name."""
        # Model outputs names like: "Card_Name_SET001"
        # card.json has names like: "Card Name"
        
        # Try exact match first (replace underscores with spaces)
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
        
        # Try matching without set code (e.g., "Card Name SET001" -> "Card Name")
        # Remove common set code patterns like WTR001, ELE002, etc.
        name_without_set = re.sub(r'[A-Z]{3}\d{3}$', '', card_name).replace('_', ' ').strip().lower()
        if name_without_set != search_name:
            for card in self.card_data:
                card_display_name = card.get('name', '').lower().strip()
                if card_display_name == name_without_set:
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
            
            # Add to cache with size limit
            if len(self.card_image_cache) >= self.max_image_cache_size:
                # Remove oldest entries (simple FIFO, could use LRU)
                oldest_key = next(iter(self.card_image_cache))
                del self.card_image_cache[oldest_key]
                print(f"[cache] Removed old image from cache, size: {len(self.card_image_cache)}")
            
            self.card_image_cache[url] = img
            return img
        except Exception as e:
            print(f"[warning] Could not load card image: {e}")
            return None
    
    def _cleanup_caches(self):
        """Periodic cache cleanup to prevent memory bloat."""
        current_time = time.time()
        if current_time - self.cache_last_cleanup < self.cache_cleanup_interval:
            return
        
        # Clear half of the image cache if it's near capacity
        if len(self.card_image_cache) > self.max_image_cache_size * 0.8:
            items_to_remove = len(self.card_image_cache) // 2
            keys_to_remove = list(self.card_image_cache.keys())[:items_to_remove]
            for key in keys_to_remove:
                del self.card_image_cache[key]
            print(f"[cache] Cleaned {items_to_remove} images from cache")
        
        self.cache_last_cleanup = current_time
    
    def run(self):
        """Main detection loop."""
        window_name = 'FaB Card Detector'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback(window_name, self._mouse_callback)
        
        # Setup global F9 hotkey monitoring in background thread (works even without focus)
        self.f9_pressed = False
        if keyboard is not None:
            def f9_handler():
                print("[hotkey] F9 detected - will reset on next frame")
                self.f9_pressed = True
            
            try:
                keyboard.on_press_key('f9', lambda _: f9_handler())
                print("[hotkey] F9 monitoring active (works even in background)")
            except Exception as e:
                print(f"[hotkey] Could not register F9 monitor: {e}")
        
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
            
            # Position window at top-left corner (0,0)
            window_offset = (0, 0)
            self._move_window_to_monitor(window_name, display_info, window_offset)
            
            try:
                cv2.resizeWindow(window_name, int(base_w), int(base_h))
            except Exception:
                pass
            
            # Window styling
            if self.args.overlay_only and self.args.transparent:
                self._enable_chromakey_transparency(
                    window_name, 
                    rgb=tuple(self.args.chroma), 
                    click_through=self.args.click_through
                )
                # In transparent mode, bring window to foreground
                time.sleep(0.1)  # Small delay for window to be created
                self._set_window_foreground(window_name)
            
            # Minimize the GUI window
            if self.minimize_gui_callback:
                self.minimize_gui_callback()
            
            # Store window name for runtime access
            self.window_name = window_name
            
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
                
                # Check if manual bbox was just completed
                if self.manual_mode and self.manual_bbox_complete and not self.manual_predictions:
                    # Check if we already tried detection on this bbox
                    if not hasattr(self, '_last_bbox_tried') or self._last_bbox_tried != self.manual_bbox_complete:
                        # Run manual detection on the snapshot frame (before semi-transparent overlay)
                        if self.manual_frame_snapshot is not None:
                            print("[manual] Running detection on cropped region...")
                            self.manual_predictions = self._run_manual_detection(
                                self.manual_frame_snapshot, 
                                self.manual_bbox_complete
                            )
                            # Remember we tried this bbox to avoid repeated detection
                            self._last_bbox_tried = self.manual_bbox_complete
                            
                            # If no predictions found, keep bbox visible and wait for user action
                            if not self.manual_predictions:
                                print("[manual] No detections found - press ESC to cancel or draw another box")
                                # Don't reset bbox - keep it visible! User can press ESC or draw a new one
                
                # Run detection (skip if in manual mode to avoid interference)
                if not self.manual_mode:
                    boxes, clss, confs, names = self._detect(frame)
                    
                    # Filter boxes (including excluding overlay region)
                    boxes, clss, confs = self._filter_boxes(boxes, clss, confs, frame.shape)
                    
                    # Hero detection and legal pool filtering
                    boxes, clss, confs = self._apply_hero_filtering(boxes, clss, confs, names, start_time)
                else:
                    # In manual mode, don't run normal detection
                    boxes, clss, confs, names = [], [], [], {}
                
                # Periodic cache cleanup
                self._cleanup_caches()
                
                # Send all detections to history queue (if queue provided)
                if self.detection_queue is not None:
                    for i, cls_idx in enumerate(clss):
                        class_id = int(cls_idx)
                        if class_id < len(names):
                            card_name = names[class_id]
                            confidence = float(confs[i])
                            try:
                                self.detection_queue.put_nowait((card_name, confidence))
                            except:
                                pass  # Queue full, skip
                
                # Find hovered box
                mouse_over_box = self._find_hovered_box(boxes, clss, names, monitor)
                
                # Create display frame
                display_frame = self._create_display_frame(
                    frame, base_h, base_w, boxes, clss, confs, names, mouse_over_box, card_w, card_h, monitor
                )
                
                # Draw hero detection status
                hero1_text = self.detected_hero1 if self.detected_hero1 else "Detecting..."
                hero2_text = self.detected_hero2 if self.detected_hero2 else "Detecting..."
                
                cv2.putText(display_frame, f"Hero1: {hero1_text}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Hero2: {hero2_text}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Draw clickable reset button
                button_x, button_y = 10, 100
                button_w, button_h = 180, 40
                button_x2, button_y2 = button_x + button_w, button_y + button_h
                
                # Store button bounds for click detection
                self.reset_button_bounds = (button_x, button_y, button_x2, button_y2)
                
                # Check if mouse is hovering over button
                mouse_x, mouse_y = self.mouse_pos
                is_hovering = (button_x <= mouse_x <= button_x2 and button_y <= mouse_y <= button_y2)
                
                # Draw button background (brighter when hovering)
                button_color = (80, 80, 180) if is_hovering else (50, 50, 120)
                cv2.rectangle(display_frame, (button_x, button_y), (button_x2, button_y2), button_color, -1)
                cv2.rectangle(display_frame, (button_x, button_y), (button_x2, button_y2), (100, 100, 200), 2)
                
                # Draw button text
                cv2.putText(display_frame, "Reset Heroes", (button_x + 10, button_y + 27),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                print(f"[DEBUG] About to show window '{window_name}', frame shape: {display_frame.shape}")
                cv2.imshow(window_name, display_frame)
                print(f"[DEBUG] imshow called successfully")
                
                # Handle keyboard input
                key = cv2.waitKey(1)  # Don't mask with 0xFF to detect function keys
                
                # Check if F9 was pressed via global hotkey
                if self.f9_pressed:
                    print("[hotkey] F9 pressed - resetting hero detection")
                    self.reset_hero_detection()
                    self.f9_pressed = False  # Reset flag
                
                # Q key: quit
                if key == ord('q'):
                    break
                
                # B key: enter manual detection mode
                elif key == ord('b') and not self.manual_mode:
                    print("[manual] Entering manual detection mode - draw box around card")
                    self.manual_mode = True
                    
                    # In fullscreen mode, use the full frame (no crop needed)
                    self.manual_frame_snapshot = frame.copy()
                    print(f"[manual] Captured full screen: {self.manual_frame_snapshot.shape}")
                    
                    self.manual_bbox_start = None
                    self.manual_bbox_current = None
                    self.manual_bbox_complete = None
                    self.manual_predictions = []
                    self._last_bbox_tried = None
                
                # Number keys 1-5: select prediction in manual mode
                elif self.manual_mode and self.manual_predictions and key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
                    idx = key - ord('1')  # Convert to 0-4
                    if idx < len(self.manual_predictions):
                        selected_card, selected_conf = self.manual_predictions[idx]
                        print(f"[manual] Selected: {selected_card} (conf: {selected_conf:.3f})")
                        self._apply_manual_card_selection(selected_card, display_frame, card_w, card_h)
                
                # '6' key: manual card name input with autocomplete
                elif self.manual_mode and self.manual_predictions and key == ord('6'):
                    print("[manual] Entering text input mode")
                    self.text_input_mode = True
                    self.text_input_buffer = ""
                    self.text_input_matches = []
                    self.text_input_selection_mode = False
                
                # Handle text input mode (when '6' was pressed)
                elif self.text_input_mode:
                    if key == 27:  # ESC - cancel text input
                        print("[manual] Cancelled text input")
                        self.text_input_mode = False
                        self.text_input_buffer = ""
                        self.text_input_matches = []
                        self.text_input_selection_mode = False
                        self.text_input_selected_index = -1
                    elif key == 9:  # Tab - cycle through suggestions (loops back to top)
                        if self.text_input_matches and not self.text_input_selection_mode:
                            self.text_input_selected_index = (self.text_input_selected_index + 1) % len(self.text_input_matches[:5])
                            print(f"[manual] Selected suggestion {self.text_input_selected_index + 1}: {self.text_input_matches[self.text_input_selected_index]}")
                    elif key == 13:  # Enter - search or confirm selection
                        if self.text_input_selection_mode and self.text_input_matches:
                            # In selection mode, Enter without typing a number cancels
                            print("[manual] Selection cancelled")
                            self.text_input_mode = False
                            self.text_input_buffer = ""
                            self.text_input_matches = []
                            self.text_input_selection_mode = False
                            self.text_input_selected_index = -1
                        elif not self.text_input_selection_mode and self.text_input_selected_index >= 0:
                            # Enter with a highlighted suggestion - select it directly
                            selected_card = self.text_input_matches[self.text_input_selected_index]
                            print(f"[manual] Selected highlighted: {selected_card}")
                            self._apply_manual_card_selection(selected_card, display_frame, card_w, card_h)
                            self.text_input_mode = False
                            self.text_input_buffer = ""
                            self.text_input_matches = []
                            self.text_input_selected_index = -1
                        elif not self.text_input_selection_mode and self.text_input_buffer:
                            # Search for legal card matches only
                            search_lower = self.text_input_buffer.lower().strip()
                            matches = []
                            for card in self.card_data:
                                card_name = card.get('name', '')
                                card_name_lower = card_name.lower().strip()
                                # Only include if card name contains search string AND is legal
                                if search_lower in card_name_lower:
                                    # Check if card is legal (same logic as detection filtering)
                                    # If no heroes detected yet, allow all cards
                                    if self.legal_card_names is None or card_name_lower in self.legal_card_names or self._is_hero_card(card_name):
                                        matches.append(card_name)
                            
                            if not matches:
                                print(f"[manual] No legal cards found matching '{self.text_input_buffer}'")
                                self.text_input_mode = False
                                self.text_input_buffer = ""
                            elif len(matches) == 1:
                                # Single match - apply it
                                selected_card = matches[0]
                                print(f"[manual] Selected: {selected_card}")
                                self._apply_manual_card_selection(selected_card, display_frame, card_w, card_h)
                                self.text_input_mode = False
                                self.text_input_buffer = ""
                                self.text_input_matches = []
                            else:
                                # Multiple matches - show selection list
                                self.text_input_matches = matches[:10]  # Show max 10
                                self.text_input_selection_mode = True
                                print(f"[manual] Found {len(matches)} legal matches, showing top 10")
                    elif key == 8 or key == 127:  # Backspace
                        if not self.text_input_selection_mode:
                            self.text_input_buffer = self.text_input_buffer[:-1]
                            self.text_input_selected_index = -1  # Reset selection on edit
                            # Update autocomplete matches as user types
                            if self.text_input_buffer:
                                search_lower = self.text_input_buffer.lower().strip()
                                matches = []
                                for card in self.card_data:
                                    card_name = card.get('name', '')
                                    card_name_lower = card_name.lower().strip()
                                    if search_lower in card_name_lower:
                                        if self.legal_card_names is None or card_name_lower in self.legal_card_names or self._is_hero_card(card_name):
                                            matches.append(card_name)
                                self.text_input_matches = matches[:10]
                            else:
                                self.text_input_matches = []
                    elif self.text_input_selection_mode and 48 <= key <= 57:  # Numbers 0-9 in selection mode
                        choice_idx = key - 48 - 1  # Convert '1'-'9' to 0-8, '0' to -1 (cancel)
                        if choice_idx == -1:
                            print("[manual] Selection cancelled")
                        elif 0 <= choice_idx < len(self.text_input_matches):
                            selected_card = self.text_input_matches[choice_idx]
                            print(f"[manual] Selected: {selected_card}")
                            self._apply_manual_card_selection(selected_card, display_frame, card_w, card_h)
                        else:
                            print("[manual] Invalid selection")
                        self.text_input_mode = False
                        self.text_input_buffer = ""
                        self.text_input_matches = []
                        self.text_input_selection_mode = False
                    elif not self.text_input_selection_mode and 32 <= key <= 126:  # Printable characters
                        self.text_input_buffer += chr(key)
                        self.text_input_selected_index = -1  # Reset selection on edit
                        # Update autocomplete matches as user types
                        search_lower = self.text_input_buffer.lower().strip()
                        matches = []
                        for card in self.card_data:
                            card_name = card.get('name', '')
                            card_name_lower = card_name.lower().strip()
                            if search_lower in card_name_lower:
                                if self.legal_card_names is None or card_name_lower in self.legal_card_names or self._is_hero_card(card_name):
                                    matches.append(card_name)
                        self.text_input_matches = matches[:10]
                
                # ESC key: cancel manual mode
                elif key == 27 and self.manual_mode:
                    # If there's a bbox drawn, clear it first
                    if self.manual_bbox_complete or self.manual_bbox_start or self.drawing_bbox:
                        print("[manual] Cleared bounding box")
                        self.manual_bbox_start = None
                        self.manual_bbox_current = None
                        self.manual_bbox_complete = None
                        self.manual_predictions = []
                        self.drawing_bbox = False
                    else:
                        # No bbox exists, exit manual mode
                        print("[manual] Cancelled manual detection mode")
                        self.manual_mode = False
                        self.manual_frame_snapshot = None

                
                # Check if window closed
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
        
            # Cleanup
            if keyboard is not None:
                try:
                    keyboard.unhook_all()
                    print("[hotkey] Unregistered F9 monitor")
                except Exception:
                    pass
            cv2.destroyAllWindows()
    
    def _detect(self, frame):
        """Run YOLO detection on frame with Phase 4 pre-filtering and confidence boosting."""
        try:
            import traceback
            # Phase 4: Get legal class IDs for pre-filtering (only if weight adjustment enabled)
            legal_class_ids = None
            if self.enable_weight_adjustment:  # Boolean value from args
                legal_class_ids = self._get_legal_class_ids()
                print(f"[phase4 debug] Weight adjustment enabled, legal_class_ids: {legal_class_ids if legal_class_ids is None else f'{len(legal_class_ids)} classes'}")
            
            # Run YOLO with optional class filtering
            predict_args = {
                'source': frame,
                'imgsz': self.args.imgsz,
                'conf': self.args.conf,
                'iou': self.args.iou,
                'verbose': False
            }
            
            # Add classes parameter if we have legal class IDs
            if legal_class_ids is not None:
                predict_args['classes'] = legal_class_ids
                print(f"[phase4 debug] Pre-filtering to {len(legal_class_ids)} classes")
            
            results = self.model.predict(**predict_args)
            
            if results is not None and len(results) > 0:
                r0 = results[0]
                if hasattr(r0, 'boxes') and r0.boxes is not None:
                    boxes = r0.boxes.xyxy.detach().cpu().numpy()
                    clss = r0.boxes.cls.detach().cpu().numpy()
                    confs = r0.boxes.conf.detach().cpu().numpy()
                    names = r0.names if hasattr(r0, 'names') else self.model.names
                    
                    # Phase 4: Apply confidence boosting if enabled
                    if self.confidence_booster and self.enable_weight_adjustment:  # Boolean value
                        # Update booster with current heroes
                        if self.detected_hero1 or self.detected_hero2:
                            hero1 = self.detected_hero1 or "Generic"
                            hero2 = self.detected_hero2 or "Generic"
                            self.confidence_booster.set_active_hero(hero1, hero2)
                        
                        # Boost confidences
                        for i in range(len(confs)):
                            card_name = names.get(int(clss[i]), "Unknown")
                            original_conf = float(confs[i])
                            boosted_conf = self.confidence_booster.adjust_confidence(card_name, original_conf)
                            # Handle if boosted_conf is a tuple (return value, message)
                            if isinstance(boosted_conf, tuple):
                                boosted_conf = boosted_conf[0]
                            if abs(float(boosted_conf) - original_conf) > 0.0001:  # Use float comparison
                                confs[i] = boosted_conf
                    
                    return boxes, clss, confs, names
        except Exception as e:
            print(f"[error] Detection failed: {e}")
            traceback.print_exc()
        
        return np.array([]), np.array([]), np.array([]), self.model.names
    
    def _filter_boxes(self, boxes, clss, confs, frame_shape):
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
        
        return boxes[keep], clss[keep], confs[keep]
    
    def _exclude_overlay_region(self, boxes, clss, confs, overlay_rect):
        """Exclude detection boxes that overlap with the card preview overlay."""
        if not overlay_rect:
            return boxes, clss, confs
            
        ox1, oy1, ox2, oy2 = overlay_rect
        keep = []
        
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            # Calculate overlap with overlay region
            overlap_x1 = max(x1, ox1)
            overlap_y1 = max(y1, oy1)
            overlap_x2 = min(x2, ox2)
            overlap_y2 = min(y2, oy2)
            
            # If there's an overlap
            if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                box_area = (x2 - x1) * (y2 - y1)
                
                # Only exclude if more than 50% of box overlaps with overlay
                if overlap_area / box_area > 0.5:
                    continue
            
            keep.append(i)
        
        return boxes[keep], clss[keep], confs[keep]
    
    def _find_hovered_box(self, boxes, clss, names, monitor):
        """Find which box the mouse is hovering over (excluding stable hover zone)."""
        if self.args.transparent:
            # Use global mouse position
            gx, gy = self._get_global_mouse_pos()
            mx = gx - monitor['left']
            my = gy - monitor['top']
        else:
            # Use window-relative mouse position
            mx, my = self.mouse_pos
        
        # Find hovered box (but skip boxes that overlap with stable hover zone)
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(v) for v in box]
            
            # Skip this box if it overlaps with the stable hover zone
            if self.stable_hover_zone:
                zx1, zy1, zx2, zy2 = self.stable_hover_zone
                # Check for overlap using standard rectangle intersection test
                if not (x2 < zx1 or x1 > zx2 or y2 < zy1 or y1 > zy2):
                    # Boxes overlap - skip this detection
                    continue
            
            class_id = int(clss[i]) if i < len(clss) else -1
            
            if isinstance(names, dict):
                label = names.get(class_id, str(class_id))
            else:
                label = str(class_id)
            
            if x1 <= mx <= x2 and y1 <= my <= y2:
                return (x1, y1, x2, y2, label)
        
        return None
    
    def _create_display_frame(self, frame, base_h, base_w, boxes, clss, confs, names, mouse_over_box, card_w, card_h, monitor):
        """Create the display frame with boxes and card overlay."""
        # Manual mode: show the actual screen capture WITHOUT any tinting
        if self.manual_mode:
            # Use the snapshot directly (already resized to match display dimensions)
            if self.manual_frame_snapshot is not None:
                display_frame = self.manual_frame_snapshot.copy()
            else:
                # Fallback to chroma key if snapshot missing
                if self.args.transparent:
                    r, g, b = self.args.chroma
                    display_frame = np.full((base_h, base_w, 3), (b, g, r), dtype=np.uint8)
                else:
                    display_frame = np.zeros((base_h, base_w, 3), dtype=np.uint8)
            
            # Draw instruction text
            text = "Draw box around card (ESC to cancel)"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            thickness = 3
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = (base_w - text_w) // 2
            text_y = 50
            self._draw_text_with_border(display_frame, text, (text_x, text_y), font, font_scale, (0, 255, 255), thickness)
            
            # Draw bbox being drawn
            if self.drawing_bbox and self.manual_bbox_start and self.manual_bbox_current:
                x1, y1 = self.manual_bbox_start
                x2, y2 = self.manual_bbox_current
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Draw completed bbox
            elif self.manual_bbox_complete:
                x1, y1, x2, y2 = self.manual_bbox_complete
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # Show predictions if available
                if self.manual_predictions and not self.text_input_mode:
                    # Draw prediction list
                    pred_y = y2 + 40
                    self._draw_text_with_border(display_frame, "Select card (press 1-5, or 6 to type):", (x1, pred_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    
                    for i, (card_name, conf) in enumerate(self.manual_predictions, 1):
                        pred_y += 35
                        text = f"{i}. {card_name} ({conf:.2f})"
                        self._draw_text_with_border(display_frame, text, (x1, pred_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                elif self.text_input_mode:
                    # Clear suggestion rects at start of each frame
                    self.text_input_suggestion_rects = []
                    
                    # Draw text input overlay
                    pred_y = y2 + 40
                    if self.text_input_selection_mode:
                        # Show selection list
                        self._draw_text_with_border(display_frame, f"Found {len(self.text_input_matches)} matches - Select (1-{len(self.text_input_matches)}, ESC to cancel):", 
                                   (x1, pred_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                        for i, match in enumerate(self.text_input_matches, 1):
                            pred_y += 35
                            self._draw_text_with_border(display_frame, f"{i}. {match}", (x1, pred_y), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    else:
                        # Show text input prompt
                        self._draw_text_with_border(display_frame, "Type card name (Enter to search, ESC to cancel):", 
                                   (x1, pred_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                        pred_y += 40
                        # Show current input with cursor
                        input_text = f"> {self.text_input_buffer}_"
                        self._draw_text_with_border(display_frame, input_text, (x1, pred_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                        
                        # Show autocomplete suggestions while typing (clickable)
                        if self.text_input_matches and self.text_input_buffer:
                            pred_y += 40
                            self._draw_text_with_border(display_frame, f"Suggestions ({len(self.text_input_matches)} matches) - Click or Tab to cycle:", 
                                      (x1, pred_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
                            
                            # Draw clickable suggestions with hover boxes
                            for i, match in enumerate(self.text_input_matches[:5], 1):
                                pred_y += 30
                                text = f"  {match}"
                                
                                # Calculate text bounds for click detection
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                scale = 0.6
                                thick = 1
                                (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thick)
                                
                                # Store clickable rectangle (with some padding)
                                padding = 5
                                rect_x1 = x1 - padding
                                rect_y1 = pred_y - text_h - padding
                                rect_x2 = x1 + text_w + padding
                                rect_y2 = pred_y + baseline + padding
                                self.text_input_suggestion_rects.append((rect_x1, rect_y1, rect_x2, rect_y2, match))
                                
                                # Check if this is the selected/highlighted item
                                is_selected = (self.text_input_selected_index == i - 1)
                                
                                if is_selected:
                                    # Bright highlight for selected item
                                    cv2.rectangle(display_frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 200, 0), -1)
                                    cv2.rectangle(display_frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 255, 0), 2)
                                    text_color = (255, 255, 255)
                                else:
                                    # Subtle background box to show it's clickable
                                    cv2.rectangle(display_frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (50, 50, 50), -1)
                                    cv2.rectangle(display_frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (100, 100, 100), 1)
                                    text_color = (200, 200, 200)
                                
                                # Draw text on top
                                self._draw_text_with_border(display_frame, text, (x1, pred_y), 
                                          font, scale, text_color, thick)
                else:
                    # Show "Processing..." message while detection runs
                    pred_y = y2 + 40
                    self._draw_text_with_border(display_frame, "Processing detection...", (x1, pred_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
            
            return display_frame
        
        # Normal mode: chroma key or black background
        if self.args.overlay_only:
            # Transparent overlay mode - start with chroma key or black background
            if self.args.transparent:
                r, g, b = self.args.chroma
                display_frame = np.full((base_h, base_w, 3), (b, g, r), dtype=np.uint8)
            else:
                display_frame = np.zeros((base_h, base_w, 3), dtype=np.uint8)
        else:
            # Windowed mode - show actual captured frame
            display_frame = frame.copy()
        
        # Draw boxes if enabled (in both windowed and overlay mode)
        if self.args.show_bboxes:
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
        
        # Determine if we should show a preview with locked position and expanded hover zone
        show_preview = False
        card_to_preview = None
        currently_hovering = False
        
        if self.args.show_card_preview:
            # Get mouse position (same logic as _find_hovered_box)
            if self.args.transparent:
                gx, gy = self._get_global_mouse_pos()
                mx = gx - monitor['left']
                my = gy - monitor['top']
            else:
                mx, my = self.mouse_pos
            
            # Check for automatic detection hover zones (normal behavior with mouse tracking)
            # Manual selections now use a separate window, so this only handles automatic detections
            if self.stable_hover_zone:
                zx1, zy1, zx2, zy2 = self.stable_hover_zone
                is_in_zone = zx1 <= mx <= zx2 and zy1 <= my <= zy2
                
                if is_in_zone:
                    # Still within the stable hover zone - keep showing preview
                    show_preview = True
                    card_to_preview = self.stable_card_name
                    currently_hovering = True
                else:
                    # Left the hover zone - clear everything
                    self.stable_card_name = None
                    self.stable_card_image = None
                    self.stable_preview_rect = None
                    self.stable_hover_zone = None
            
            # If no stable zone, check if hovering over a new bbox
            if not currently_hovering and mouse_over_box:
                bx1, by1, bx2, by2, class_name = mouse_over_box
                
                print(f"[hover] Creating new preview for {class_name} at bbox ({bx1}, {by1}, {bx2}, {by2})")
                
                # Create new preview with locked position
                card_to_preview = class_name
                show_preview = True
                currently_hovering = True
                
                # Calculate and LOCK preview position (only calculated once)
                overlay_x = max(0, min(bx1, display_frame.shape[1] - card_w))
                overlay_y = max(0, min(by2 + 10, display_frame.shape[0] - card_h))
                self.stable_preview_rect = (overlay_x, overlay_y, overlay_x + card_w, overlay_y + card_h)
                
                # Create expanded hover zone (bbox + margin)
                margin = self.hover_zone_margin
                self.stable_hover_zone = (
                    max(0, bx1 - margin),
                    max(0, by1 - margin),
                    min(display_frame.shape[1], bx2 + margin),
                    min(display_frame.shape[0], by2 + margin)
                )
                
                # Load the card image
                self.stable_card_name = card_to_preview
                image_url = self.get_image_url_by_name(card_to_preview)
                if image_url:
                    card_img = self.get_card_image(image_url)
                    if card_img:
                        self.stable_card_image = card_img
            
            # Check linger duration after leaving hover zone
            if not currently_hovering and self.stable_card_name is not None:
                time_since_hover = time.time() - self.preview_last_active_time
                if time_since_hover < self.preview_linger_duration:
                    # Still within linger period - keep showing
                    show_preview = True
                    card_to_preview = self.stable_card_name
        
        # Update last active time if currently hovering
        if currently_hovering:
            self.preview_last_active_time = time.time()
        
        # Display the locked preview
        self.last_overlay_rect = None
        if show_preview and card_to_preview and self.stable_card_image and self.stable_preview_rect:
            px1, py1, px2, py2 = self.stable_preview_rect
            card_np = np.array(self.stable_card_image.resize((card_w, card_h)))
            card_np = cv2.cvtColor(card_np, cv2.COLOR_RGBA2BGR)
            
            # Store overlay region
            self.last_overlay_rect = self.stable_preview_rect
            
            # Blend card with alpha (avoid magenta edges)
            roi = display_frame[py1:py2, px1:px2]
            
            if roi.shape[:2] == card_np.shape[:2]:
                alpha = np.array(self.stable_card_image.resize((card_w, card_h)))[:, :, 3:] / 255.0
                mask = alpha > 0.3
                roi[:] = np.where(mask,
                                (alpha * card_np + (1 - alpha) * roi).astype(np.uint8),
                                roi)
        elif not show_preview:
            # Clear everything when linger duration expires
            self.stable_card_name = None
            self.stable_card_image = None
            self.stable_preview_rect = None
            self.stable_hover_zone = None
        
        # Display manual preview (independent of normal hover preview)
        if self.manual_preview_card_image and self.manual_preview_expiry > 0:
            if time.time() < self.manual_preview_expiry:
                # Calculate position based on side preference
                margin = 50  # Distance from edge (left or right)
                margin_bottom = 59  # 59px from bottom
                
                # Get preview side preference (default to left)
                preview_side = getattr(self.args, 'manual_preview_side', 'left')
                
                if preview_side == 'right':
                    # Position from right edge (mirror of left position)
                    manual_px1 = display_frame.shape[1] - card_w - margin
                else:
                    # Position from left edge (default)
                    manual_px1 = margin
                
                manual_py1 = display_frame.shape[0] - card_h - margin_bottom
                manual_px2 = manual_px1 + card_w
                manual_py2 = manual_py1 + card_h
                
                # Render manual preview card
                manual_card_np = np.array(self.manual_preview_card_image.resize((card_w, card_h)))
                manual_card_np = cv2.cvtColor(manual_card_np, cv2.COLOR_RGBA2BGR)
                
                # Blend with alpha (only blend where alpha > threshold to avoid magenta edges)
                manual_roi = display_frame[manual_py1:manual_py2, manual_px1:manual_px2]
                if manual_roi.shape[:2] == manual_card_np.shape[:2]:
                    manual_alpha = np.array(self.manual_preview_card_image.resize((card_w, card_h)))[:, :, 3:] / 255.0
                    # Only blend where alpha is significant (> 0.3) to avoid color bleed
                    mask = manual_alpha > 0.3
                    manual_roi[:] = np.where(mask, 
                                            (manual_alpha * manual_card_np + (1 - manual_alpha) * manual_roi).astype(np.uint8),
                                            manual_roi)
            else:
                # Expired - clear state
                self.manual_preview_card_name = None
                self.manual_preview_card_image = None
                self.manual_preview_expiry = 0
                print("[manual] Manual preview expired after 15 seconds")
        
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
                SWP_SHOWWINDOW = 0x0040
                
                # Use SHOWWINDOW instead of NOACTIVATE to ensure window is active
                user32.SetWindowPos(hwnd, HWND_TOPMOST if topmost else HWND_NOTOPMOST,
                                   0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW)
        except Exception:
            pass
    
    def _set_window_foreground(self, window_name):
        """Set window as foreground/active window (Windows)."""
        try:
            user32 = ctypes.windll.user32
            hwnd = user32.FindWindowW(None, window_name)
            if hwnd:
                SW_RESTORE = 9
                HWND_TOPMOST = -1
                HWND_NOTOPMOST = -2
                SWP_NOMOVE = 0x0002
                SWP_NOSIZE = 0x0001
                SWP_SHOWWINDOW = 0x0040
                
                # 1. Show/restore window
                user32.ShowWindow(hwnd, SW_RESTORE)
                
                # 2. Make topmost temporarily then reset (forces to front)
                user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW)
                user32.SetWindowPos(hwnd, HWND_NOTOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW)
                
                # 3. Bring to top of Z-order
                user32.BringWindowToTop(hwnd)
                
                # 4. Set as foreground (active) window
                foreground = user32.GetForegroundWindow()
                if foreground != hwnd:
                    foreground_thread = user32.GetWindowThreadProcessId(foreground, None)
                    current_thread = ctypes.windll.kernel32.GetCurrentThreadId()
                    user32.AttachThreadInput(current_thread, foreground_thread, True)
                    user32.SetForegroundWindow(hwnd)
                    user32.SetFocus(hwnd)
                    user32.SetActiveWindow(hwnd)
                    user32.AttachThreadInput(current_thread, foreground_thread, False)
                else:
                    user32.SetForegroundWindow(hwnd)
                    user32.SetFocus(hwnd)
                    user32.SetActiveWindow(hwnd)
        except Exception as e:
            print(f"[window] Error bringing window to front: {e}")
    
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
            
            if click_through:
                # Enable click-through: mouse passes through transparent areas
                new_style = current | WS_EX_LAYERED | WS_EX_TRANSPARENT
            else:
                # Disable click-through: remove WS_EX_TRANSPARENT flag
                new_style = (current | WS_EX_LAYERED) & ~WS_EX_TRANSPARENT
            
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_style)
            
            # Apply chromakey transparency
            r, g, b = rgb
            colorref = (r & 0xFF) | ((g & 0xFF) << 8) | ((b & 0xFF) << 16)
            user32.SetLayeredWindowAttributes(hwnd, colorref, 0, LWA_COLORKEY)
            
            # Force window update
            SWP_FRAMECHANGED = 0x0020
            SWP_NOMOVE = 0x0002
            SWP_NOSIZE = 0x0001
            SWP_NOZORDER = 0x0004
            user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0, 
                               SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER)
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
