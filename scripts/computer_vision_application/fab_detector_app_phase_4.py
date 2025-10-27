#!/usr/bin/env python3
"""
FaB Card Detector Phase 4 - Hero-Aware Detection with Confidence Boosting
==========================================================================

Real-time Flesh and Blood card detection with advanced hero-aware filtering:
1. Windowed Mode: Shows captured screen with detection boxes
2. Transparent Overlay Mode: Invisible window that shows card preview on hover

Phase 4 Features:
- Hero detection and tracking (dual-hero support)
- Pre-detection filtering (restricts YOLO to legal card classes only)
- Confidence boosting based on competitive meta usage (card_weights_all_printings.json)
- Combined "Active Hero Weight Adjustment" toggle for A/B testing
- Multi-monitor support
- Configurable confidence thresholds
- Card image preview on hover
- Performance metrics
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

# Import confidence booster
try:
    from confidence_booster import ConfidenceBooster
except ImportError:
    # Try relative import
    try:
        from .confidence_booster import ConfidenceBooster
    except ImportError:
        print("[warning] ConfidenceBooster not found - confidence adjustment disabled")
        ConfidenceBooster = None


class DetectorGUI:
    """GUI launcher for the card detector."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("FaB Card Detector Phase 4")
        self.root.geometry("650x850")
        self.root.resizable(True, True)
        self.root.minsize(600, 750)
        
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
        title = tk.Label(scrollable_frame, text="FaB Card Live Detector (Phase 4)", 
                        font=("Arial", 20, "bold"))
        title.pack(pady=20)
        
        # Model selection
        model_frame = ttk.LabelFrame(scrollable_frame, text="Model Configuration", padding=10)
        model_frame.pack(fill="x", padx=20, pady=10)
        
        ttk.Label(model_frame, text="Model Weights:").grid(row=0, column=0, sticky="w", pady=5)
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
        
        # Hero filtering toggle (NEW - Phase 4)
        hero_frame = ttk.LabelFrame(scrollable_frame, text="Phase 4: Hero-Aware Filtering & Override", padding=10)
        hero_frame.pack(fill="x", padx=20, pady=10)
        
        self.hero_filtering_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(hero_frame, text="Active Hero Weight Adjustment", 
                       variable=self.hero_filtering_enabled).pack(anchor="w", pady=2)
        
        ttk.Label(hero_frame, text="When enabled:", font=("Arial", 9, "bold")).pack(anchor="w", pady=(5, 2))
        ttk.Label(hero_frame, text="• Detects active hero(es) from gameplay", 
                 font=("Arial", 8)).pack(anchor="w", padx=20)
        ttk.Label(hero_frame, text="• Restricts YOLO to only legal cards (pre-detection)", 
                 font=("Arial", 8)).pack(anchor="w", padx=20)
        ttk.Label(hero_frame, text="• Boosts confidence for meta-relevant cards", 
                 font=("Arial", 8)).pack(anchor="w", padx=20)
        ttk.Label(hero_frame, text="• Reduces wasted GPU compute on illegal cards", 
                 font=("Arial", 8)).pack(anchor="w", padx=20)
        
        # Hero override controls
        ttk.Separator(hero_frame, orient='horizontal').pack(fill='x', pady=10)
        ttk.Label(hero_frame, text="Hero Override (Optional):", font=("Arial", 9, "bold")).pack(anchor="w", pady=(5, 2))
        
        # Load hero list
        self.hero_names = self._load_hero_names()
        hero_options = ["Auto Detect"] + self.hero_names
        
        hero_grid = ttk.Frame(hero_frame)
        hero_grid.pack(fill="x", pady=5)
        
        ttk.Label(hero_grid, text="Player 1:").grid(row=0, column=0, sticky="w", padx=5)
        self.hero1_var = tk.StringVar(value="Auto Detect")
        hero1_combo = ttk.Combobox(hero_grid, textvariable=self.hero1_var, 
                                   values=hero_options, state="readonly", width=30)
        hero1_combo.grid(row=0, column=1, padx=5)
        self.hero1_status_var = tk.StringVar(value="")
        self.hero1_status_label = ttk.Label(hero_grid, textvariable=self.hero1_status_var, 
                                            foreground="green", font=("Arial", 8, "italic"))
        self.hero1_status_label.grid(row=0, column=2, sticky="w", padx=5)
        
        ttk.Label(hero_grid, text="Player 2:").grid(row=1, column=0, sticky="w", padx=5)
        self.hero2_var = tk.StringVar(value="Auto Detect")
        hero2_combo = ttk.Combobox(hero_grid, textvariable=self.hero2_var, 
                                   values=hero_options, state="readonly", width=30)
        hero2_combo.grid(row=1, column=1, padx=5)
        self.hero2_status_var = tk.StringVar(value="")
        self.hero2_status_label = ttk.Label(hero_grid, textvariable=self.hero2_status_var, 
                                            foreground="green", font=("Arial", 8, "italic"))
        self.hero2_status_label.grid(row=1, column=2, sticky="w", padx=5)
        
        # Buttons
        button_grid = ttk.Frame(hero_frame)
        button_grid.pack(fill="x", pady=5)
        ttk.Button(button_grid, text="Apply Hero Override", 
                  command=self._apply_hero_override).pack(side="left", padx=5)
        ttk.Button(button_grid, text="Reset Auto-Detection", 
                  command=self._reset_hero_detection).pack(side="left", padx=5)
        
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
        
        self.show_card_preview = tk.BooleanVar(value=True)
        ttk.Checkbutton(overlay_frame, text="Show card preview on hover", 
                       variable=self.show_card_preview).pack(anchor="w", pady=2)
        
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
        config_path = Path("detector_config_phase4.json")
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                self.model_path.set(config.get('model_path', self.model_path.get()))
                self.conf_threshold.set(config.get('conf_threshold', 0.69))
                self.iou_threshold.set(config.get('iou_threshold', 0.50))
                self.hero_filtering_enabled.set(config.get('hero_filtering_enabled', False))
            except Exception as e:
                print(f"Could not load config: {e}")
    
    def _save_config(self):
        """Save current settings to config."""
        config = {
            'model_path': self.model_path.get(),
            'conf_threshold': self.conf_threshold.get(),
            'iou_threshold': self.iou_threshold.get(),
            'hero_filtering_enabled': self.hero_filtering_enabled.get(),
        }
        try:
            with open("detector_config_phase4.json", 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Could not save config: {e}")
    
    def _browse_model(self):
        """Browse for model file."""
        # Start in the model directory if it exists, otherwise current model's directory
        initial_dir = None
        current_path = self.model_path.get()
        
        if current_path and os.path.exists(current_path):
            initial_dir = os.path.dirname(os.path.abspath(current_path))
        elif os.path.exists("models"):
            initial_dir = os.path.abspath("models")
        elif os.path.exists("runs/train"):
            initial_dir = os.path.abspath("runs/train")
        
        filename = filedialog.askopenfilename(
            title="Select Model Weights",
            initialdir=initial_dir,
            filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")]
        )
        if filename:
            self.model_path.set(filename)
    
    def _load_hero_names(self):
        """Load list of hero names from card data."""
        possible_paths = [
            Path('data/card.json'),
            Path(__file__).parent.parent.parent / 'data' / 'card.json',
            Path.cwd() / 'data' / 'card.json'
        ]
        
        for card_json_path in possible_paths:
            if card_json_path.exists():
                try:
                    with open(card_json_path, 'r', encoding='utf-8') as f:
                        cards = json.load(f)
                        # Extract hero names
                        hero_names = []
                        for card in cards:
                            card_types = set(card.get('types', []))
                            if 'Hero' in card_types or 'Young' in card_types:
                                hero_names.append(card.get('name', ''))
                        return sorted(set(hero_names))  # Remove duplicates and sort
                except Exception as e:
                    print(f"Could not load heroes from {card_json_path}: {e}")
        
        return []  # Return empty list if no heroes found
    
    def _apply_hero_override(self):
        """Apply manual hero override to detector."""
        if not self.detector_running:
            messagebox.showwarning("Not Running", "Start detection first before overriding heroes.")
            return
        
        # This will be handled via queue in the detector thread
        # For now, just update status labels
        hero1 = self.hero1_var.get()
        hero2 = self.hero2_var.get()
        
        if hero1 != "Auto Detect":
            self.hero1_status_var.set("(Manual Override)")
        else:
            self.hero1_status_var.set("")
        
        if hero2 != "Auto Detect":
            self.hero2_status_var.set("(Manual Override)")
        else:
            self.hero2_status_var.set("")
        
        print(f"[GUI] Hero override: P1={hero1}, P2={hero2}")
    
    def _reset_hero_detection(self):
        """Reset hero detection (between games)."""
        self.hero1_var.set("Auto Detect")
        self.hero2_var.set("Auto Detect")
        self.hero1_status_var.set("")
        self.hero2_status_var.set("")
        print(f"[GUI] Hero detection reset")
    
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
        
        if self.hero_filtering_enabled.get() and ConfidenceBooster is None:
            messagebox.showwarning("Warning", "ConfidenceBooster module not found. Hero filtering will be disabled.")
            self.hero_filtering_enabled.set(False)
        
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
                show_card_preview=self.show_card_preview.get(),
                hero_filtering_enabled=self.hero_filtering_enabled.get(),
                video=None
            )
            
            # Run detector
            detector = CardDetector(args, stop_callback=lambda: not self.detector_running, gui=self)
            detector.run()
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Detection Error", str(e)))
        finally:
            self.root.after(0, self._stop_detection)


class CardDetector:
    """Core detection engine with hero-aware filtering and confidence boosting."""
    
    def __init__(self, args, stop_callback=None, gui=None):
        self.args = args
        self.stop_callback = stop_callback
        self.gui = gui  # Reference to GUI for updating hero status
        
        # Load model
        print(f"[init] Loading model from {args.weights}...")
        self.model = YOLO(args.weights)
        print(f"[init] Model loaded with {len(self.model.names)} classes")
        
        # Load card data
        self.card_data = self._load_card_data()
        self.card_image_cache = {}
        
        # Hero detection state
        self.detected_hero1 = None
        self.detected_hero2 = None
        self.legal_card_names = None  # Set of legal card names (lowercase)
        self.legal_class_ids = None   # List of legal class IDs for YOLO filtering
        self.hero_override_active = False  # Track if manual override is active
        
        # Hero detection thresholds (dynamic adjustment)
        self.hero_detection_threshold = 0.69
        self.hero_threshold_min = 0.40
        self.hero_threshold_step = 0.05
        self.hero_threshold_interval = 1.0  # seconds
        self.hero_threshold_last_update = time.time()
        
        # Initialize confidence booster (if hero filtering enabled)
        self.confidence_booster = None
        if args.hero_filtering_enabled and ConfidenceBooster is not None:
            try:
                self.confidence_booster = ConfidenceBooster(
                    weights_path='data/card_weights_all_printings.json',
                    enabled=True
                )
                print(f"[init] ConfidenceBooster initialized with {len(self.confidence_booster.card_weights)} heroes")
            except Exception as e:
                print(f"[warning] Could not initialize ConfidenceBooster: {e}")
                self.confidence_booster = None
        
        # Tracking
        self.last_overlay_rect = None
        self.mouse_pos = (0, 0)
        self.fps_history = []
        
    def _load_card_data(self):
        """Load card metadata."""
        possible_paths = []
        
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        if getattr(sys, 'frozen', False):
            base_path = Path(sys._MEIPASS)
            possible_paths.append(base_path / 'data' / 'card.json')
        
        # Development paths
        script_dir = Path(__file__).parent
        possible_paths.extend([
            script_dir.parent.parent / 'data' / 'card.json',  # Up two levels from scripts/computer_vision_application
            Path('data/card.json'),
            Path.cwd() / 'data' / 'card.json'
        ])
        
        for card_json_path in possible_paths:
            if card_json_path.exists():
                try:
                    with open(card_json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        print(f"[init] Loaded {len(data)} cards from {card_json_path}")
                        return data
                except Exception as e:
                    print(f"[warning] Could not load {card_json_path}: {e}")
        
        print("[warning] card.json not found - card preview and hero filtering will not work")
        return []
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for window."""
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_pos = (x, y)
    
    def _is_hero_card(self, card_name):
        """Check if a card is a hero."""
        search_name = card_name.replace('_', ' ').lower().strip()
        for card in self.card_data:
            if card.get('name', '').lower().strip() == search_name:
                card_types = set(card.get('types', []))
                return 'Hero' in card_types or 'Young' in card_types
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
            if 'Hero' in card_types or 'Young' in card_types:
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
    
    def _get_legal_class_ids_for_hero(self, hero_name):
        """Get list of legal class IDs for pre-detection filtering."""
        if not self.legal_card_names:
            return None
        
        legal_ids = []
        names = self.model.names  # Dict: {class_id: class_name}
        
        for class_id, class_name in names.items():
            # Normalize class name for comparison
            class_name_lower = class_name.replace('_', ' ').lower().strip()
            
            # Always allow hero detections
            if self._is_hero_card(class_name):
                legal_ids.append(class_id)
                continue
            
            # Check if card is in legal pool
            if class_name_lower in self.legal_card_names:
                legal_ids.append(class_id)
        
        print(f"[hero] Pre-detection filter: {len(legal_ids)} legal class IDs (out of {len(names)})")
        return legal_ids
    
    def _rebuild_legal_pool(self):
        """Rebuild the legal card pool based on detected heroes."""
        if not self.detected_hero1 and not self.detected_hero2:
            self.legal_card_names = None
            self.legal_class_ids = None
            return
        
        # Build pools for each hero
        pool1 = self._build_legal_card_pool(self.detected_hero1) if self.detected_hero1 else set()
        pool2 = self._build_legal_card_pool(self.detected_hero2) if self.detected_hero2 else set()
        
        # Union of both pools
        self.legal_card_names = pool1 | pool2
        
        # Build class ID list for pre-detection filtering
        self.legal_class_ids = self._get_legal_class_ids_for_hero(self.detected_hero1 or self.detected_hero2)
        
        # Update confidence booster with detected hero
        if self.confidence_booster and self.detected_hero1:
            self.confidence_booster.set_active_hero(self.detected_hero1)
            print(f"[hero] ConfidenceBooster activated for {self.detected_hero1}")
    
    def _apply_hero_filtering(self, boxes, clss, confs, names, current_time):
        """Apply hero-based detection, filtering, and confidence boosting."""
        if len(boxes) == 0:
            return boxes, clss, confs
        
        # Check for manual hero override from GUI
        if self.gui and self.args.hero_filtering_enabled:
            hero1_override = self.gui.hero1_var.get() if hasattr(self.gui, 'hero1_var') else "Auto Detect"
            hero2_override = self.gui.hero2_var.get() if hasattr(self.gui, 'hero2_var') else "Auto Detect"
            
            # Apply manual override if changed
            if hero1_override != "Auto Detect" and hero1_override != self.detected_hero1:
                self.detected_hero1 = hero1_override
                self.hero_override_active = True
                print(f"[hero] Manual override: Hero 1 = {hero1_override}")
                self._rebuild_legal_pool()
                self._update_gui_hero_status()
            
            if hero2_override != "Auto Detect" and hero2_override != self.detected_hero2:
                self.detected_hero2 = hero2_override
                self.hero_override_active = True
                print(f"[hero] Manual override: Hero 2 = {hero2_override}")
                self._rebuild_legal_pool()
                self._update_gui_hero_status()
            
            # Handle reset (both set to Auto Detect)
            if hero1_override == "Auto Detect" and self.detected_hero1 and self.hero_override_active:
                self.detected_hero1 = None
                self.hero_override_active = False
                print(f"[hero] Reset: Hero 1 cleared")
                self._rebuild_legal_pool()
                self._update_gui_hero_status()
            
            if hero2_override == "Auto Detect" and self.detected_hero2 and self.hero_override_active:
                self.detected_hero2 = None
                self.hero_override_active = False
                print(f"[hero] Reset: Hero 2 cleared")
                self._rebuild_legal_pool()
                self._update_gui_hero_status()
        
        # Only do auto-detection if not manually overridden
        if not self.hero_override_active:
            # Dynamic threshold adjustment: Lower threshold every second until both heroes detected or min reached
            if (self.detected_hero1 is None or self.detected_hero2 is None):
                if current_time - self.hero_threshold_last_update >= self.hero_threshold_interval:
                    if self.hero_detection_threshold > self.hero_threshold_min:
                        self.hero_detection_threshold = max(
                            self.hero_threshold_min,
                            self.hero_detection_threshold - self.hero_threshold_step
                        )
                        print(f"[hero] Lowering detection threshold to {self.hero_detection_threshold:.2f}")
                        self.hero_threshold_last_update = current_time
            
            # Step 1: Check for hero detections (auto mode only)
            for i, cls_idx in enumerate(clss):
                class_id = int(cls_idx)
                if class_id < len(names):
                    card_name = names[class_id]
                    confidence = float(confs[i])
                    
                    # Check if this is a hero card
                    if self._is_hero_card(card_name):
                        if confidence >= self.hero_detection_threshold:
                            # Assign to first available hero slot
                            if self.detected_hero1 is None:
                                self.detected_hero1 = card_name
                                print(f"[hero] Detected Hero 1: {card_name} (conf: {confidence:.2f})")
                                self._rebuild_legal_pool()
                                self._update_gui_hero_status()
                            elif self.detected_hero2 is None and card_name.lower() != self.detected_hero1.lower():
                                self.detected_hero2 = card_name
                                print(f"[hero] Detected Hero 2: {card_name} (conf: {confidence:.2f})")
                                self._rebuild_legal_pool()
                                self._update_gui_hero_status()
        
        # Step 2: Apply confidence boosting if enabled
        if self.confidence_booster and self.legal_card_names is not None:
            new_confs = []
            for i, cls_idx in enumerate(clss):
                class_id = int(cls_idx)
                if class_id < len(names):
                    card_name = names[class_id]
                    confidence = float(confs[i])
                    
                    # Skip hero cards from boosting
                    if self._is_hero_card(card_name):
                        new_confs.append(confidence)
                        continue
                    
                    # Apply confidence boost based on meta usage
                    boosted_conf, reason = self.confidence_booster.adjust_confidence(card_name, confidence)
                    new_confs.append(boosted_conf)
                    
                    if boosted_conf != confidence:
                        print(f"[boost] {card_name}: {confidence:.3f} → {boosted_conf:.3f} ({reason})")
            
            confs = np.array(new_confs)
        
        return boxes, clss, confs
    
    def _update_gui_hero_status(self):
        """Update GUI with current hero detection status."""
        if not self.gui:
            return
        
        def update():
            if self.detected_hero1:
                status = "(Auto-Detected)" if not self.hero_override_active else "(Manual Override)"
                self.gui.hero1_status_var.set(status)
                if not self.hero_override_active:
                    # Update dropdown to show detected hero
                    self.gui.hero1_var.set(self.detected_hero1)
            else:
                self.gui.hero1_status_var.set("")
                if not self.hero_override_active:
                    self.gui.hero1_var.set("Auto Detect")
            
            if self.detected_hero2:
                status = "(Auto-Detected)" if not self.hero_override_active else "(Manual Override)"
                self.gui.hero2_status_var.set(status)
                if not self.hero_override_active:
                    # Update dropdown to show detected hero
                    self.gui.hero2_var.set(self.detected_hero2)
            else:
                self.gui.hero2_status_var.set("")
                if not self.hero_override_active:
                    self.gui.hero2_var.set("Auto Detect")
        
        # Schedule GUI update on main thread
        if hasattr(self.gui, 'root'):
            self.gui.root.after(0, update)
    
    def get_image_url_by_name(self, card_name):
        """Get card image URL by name."""
        import re
        search_name = card_name.replace('_', ' ').lower().strip()
        
        for card in self.card_data:
            card_display_name = card.get('name', '').lower().strip()
            if card_display_name == search_name:
                printings = card.get('printings', [])
                if printings:
                    return printings[0].get('image_url')
            
            # Fuzzy match without punctuation
            card_no_punct = re.sub(r'[^\w\s]', '', card_display_name)
            search_no_punct = re.sub(r'[^\w\s]', '', search_name)
            if card_no_punct == search_no_punct:
                printings = card.get('printings', [])
                if printings:
                    return printings[0].get('image_url')
        
        # Try matching without set code
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
            self.card_image_cache[url] = img
            return img
        except Exception as e:
            print(f"[warning] Could not load card image: {e}")
            return None
    
    def run(self):
        """Main detection loop."""
        window_name = 'FaB Card Detector Phase 4'
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
                current_time = time.time()
                
                # Capture screen
                img = np.array(sct.grab(monitor))
                frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                
                # Run detection (with or without pre-filtering)
                boxes, clss, confs, names = self._detect(frame)
                
                # Apply hero-aware filtering and confidence boosting
                if self.args.hero_filtering_enabled:
                    boxes, clss, confs = self._apply_hero_filtering(boxes, clss, confs, names, current_time)
                
                # Filter boxes by geometry
                boxes, clss, confs = self._filter_boxes(boxes, clss, confs, frame.shape)
                
                # Find hovered box
                mouse_over_box = self._find_hovered_box(boxes, clss, names, monitor)
                
                # Create display frame
                display_frame = self._create_display_frame(
                    frame, base_h, base_w, boxes, clss, confs, names, mouse_over_box, card_w, card_h
                )
                
                # Calculate FPS
                fps = 1.0 / (time.time() - start_time)
                self.fps_history.append(fps)
                if len(self.fps_history) > 30:
                    self.fps_history.pop(0)
                avg_fps = sum(self.fps_history) / len(self.fps_history)
                
                # Draw FPS and status
                cv2.putText(display_frame, f"FPS: {avg_fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Cards: {len(boxes)}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Draw hero status if filtering enabled
                if self.args.hero_filtering_enabled:
                    hero_text = f"Hero Filter: {'ON' if self.legal_card_names else 'Detecting...'}"
                    cv2.putText(display_frame, hero_text, (10, 110),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    if self.detected_hero1:
                        cv2.putText(display_frame, f"H1: {self.detected_hero1}", (10, 150),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    if self.detected_hero2:
                        cv2.putText(display_frame, f"H2: {self.detected_hero2}", (10, 190),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                cv2.imshow(window_name, display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
        
        cv2.destroyAllWindows()
    
    def _detect(self, frame):
        """Run YOLO detection on frame with optional pre-filtering."""
        try:
            # Apply pre-detection filtering if hero filtering is enabled and we have legal class IDs
            kwargs = {
                'source': frame,
                'imgsz': self.args.imgsz,
                'conf': self.args.conf,
                'iou': self.args.iou,
                'verbose': False
            }
            
            if self.args.hero_filtering_enabled and self.legal_class_ids is not None:
                kwargs['classes'] = self.legal_class_ids
            
            results = self.model.predict(**kwargs)
            
            if results and len(results) > 0:
                r0 = results[0]
                if hasattr(r0, 'boxes') and r0.boxes is not None:
                    boxes = r0.boxes.xyxy.detach().cpu().numpy()
                    clss = r0.boxes.cls.detach().cpu().numpy()
                    confs = r0.boxes.conf.detach().cpu().numpy()
                    names = r0.names if hasattr(r0, 'names') else self.model.names
                    return boxes, clss, confs, names
        except Exception as e:
            print(f"[error] Detection failed: {e}")
        
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
    
    def _create_display_frame(self, frame, base_h, base_w, boxes, clss, confs, names, mouse_over_box, card_w, card_h):
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
            confidence = float(confs[i]) if i < len(confs) else 0.0
            
            if isinstance(names, dict):
                label = names.get(class_id, str(class_id))
            else:
                label = str(class_id)
            
            # Add confidence to label
            label_with_conf = f"{label} ({confidence:.2f})"
            
            # Draw box
            color = (0, 255, 0) if mouse_over_box and (x1, y1, x2, y2, label) == mouse_over_box else (255, 0, 0)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            (w, h), _ = cv2.getTextSize(label_with_conf, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(display_frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
            cv2.putText(display_frame, label_with_conf, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw card preview if hovering
        if mouse_over_box and self.args.show_card_preview:
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
    if len(sys.argv) > 1:
        # Command-line mode
        parser = argparse.ArgumentParser(description='FaB Card Detector Phase 4')
        parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
        parser.add_argument('--conf', type=float, default=0.69, help='Confidence threshold')
        parser.add_argument('--iou', type=float, default=0.50, help='IOU threshold')
        parser.add_argument('--mode', type=str, choices=['windowed', 'overlay'], default='windowed')
        parser.add_argument('--hero-filtering', action='store_true', help='Enable hero filtering')
        
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
