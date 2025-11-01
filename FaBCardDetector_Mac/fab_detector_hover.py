#!/usr/bin/env python3
"""
FaB Card Detector with Hover Preview
- Hover over cards to see preview in separate window
- No bounding boxes shown (clean view)
- Press B to toggle bounding boxes
- Press Q to quit
- Press H for help
"""

import cv2
import numpy as np
from ultralytics import YOLO
import mss
import time
import os
import json
import requests
from PIL import Image
import io

# Configuration
MODEL_PATH = "../models/Phase3/best.pt"
CARD_JSON_PATH = "../data/card.json"
HEROES_JSON_PATH = "../data/heroes_card.json"
CAPTURE_MONITOR = 2
CONFIDENCE_THRESHOLD = 0.07
IOU_THRESHOLD = 0.10
DETECTION_SCALE = 0.5

class CardImageCache:
    """Cache for downloaded card images"""
    def __init__(self, card_json_path):
        self.cache = {}
        self.card_data = {}
        
        # Load card data as a list, then convert to dict by name for faster lookup
        if os.path.exists(card_json_path):
            with open(card_json_path, 'r') as f:
                cards = json.load(f)
                for card in cards:
                    card_name = card.get('name', '')
                    if card_name:
                        self.card_data[card_name.lower()] = card
            print(f"Loaded {len(self.card_data)} cards from database")
        else:
            print(f"Warning: card.json not found at {card_json_path}")
    
    def get_card_url(self, card_name):
        """Get image URL for a card"""
        # Model outputs names like: "Card_Name_SET001"
        # card.json has names like: "Card Name"
        
        # Try exact match first (replace underscores with spaces)
        import re
        search_name = card_name.replace('_', ' ').lower().strip()
        
        for card in self.card_data.values():
            card_display_name = card.get('name', '').lower().strip()
            
            # Exact match
            if card_display_name == search_name:
                printings = card.get('printings', [])
                if printings and len(printings) > 0:
                    return printings[0].get('image_url')
            
            # Fuzzy match without punctuation
            card_no_punct = re.sub(r'[^\w\s]', '', card_display_name)
            search_no_punct = re.sub(r'[^\w\s]', '', search_name)
            if card_no_punct == search_no_punct:
                printings = card.get('printings', [])
                if printings and len(printings) > 0:
                    return printings[0].get('image_url')
        
        # Try matching without set code (e.g., "Card Name SET001" -> "Card Name")
        name_without_set = re.sub(r'[A-Z]{3}\d{3}$', '', card_name).replace('_', ' ').strip().lower()
        if name_without_set != search_name:
            for card in self.card_data.values():
                card_display_name = card.get('name', '').lower().strip()
                if card_display_name == name_without_set:
                    printings = card.get('printings', [])
                    if printings and len(printings) > 0:
                        return printings[0].get('image_url')
        
        return None
    
    def get_image(self, card_name, max_height=300):
        """Get cached or download card image"""
        cache_key = f"{card_name}_{max_height}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Try to download image
        url = self.get_card_url(card_name)
        if url:
            try:
                print(f"Downloading image for {card_name}...")
                print(f"URL: {url}")
                response = requests.get(url, timeout=3)
                
                if response.status_code == 200:
                    image = Image.open(io.BytesIO(response.content))
                    
                    # Convert to OpenCV format
                    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    
                    # Resize to specified preview size
                    if img_cv.shape[0] > max_height:
                        ratio = max_height / img_cv.shape[0]
                        new_width = int(img_cv.shape[1] * ratio)
                        img_cv = cv2.resize(img_cv, (new_width, max_height))
                    
                    self.cache[cache_key] = img_cv
                    print(f"Successfully cached image for {card_name}")
                    return img_cv
                else:
                    print(f"Failed to download (HTTP {response.status_code})")
                    
            except Exception as e:
                print(f"Failed to download image for {card_name}: {e}")
                return None
        else:
            print(f"No image URL found for {card_name}")
        
        return None


class FaBDetectorHover:
    def __init__(self, model_path, card_json_path, capture_monitor=2):
        self.model_path = model_path
        self.capture_monitor = capture_monitor
        self.confidence = CONFIDENCE_THRESHOLD
        self.iou_threshold = IOU_THRESHOLD
        self.model = None
        self.sct = mss.mss()
        self.running = False
        self.show_boxes = True  # Start with boxes ON
        
        # Card image cache
        self.image_cache = CardImageCache(card_json_path)
        
        # Hero detection system
        self.detected_hero1 = None
        self.detected_hero2 = None
        self.hero1_confidence_history = []  # List of (card_name, conf, timestamp)
        self.hero2_confidence_history = []
        self.hero_detection_threshold = 0.15  # Much lower threshold to catch heroes with low confidence
        self.hero_threshold_min = 0.07  # Match the general detection threshold
        self.hero_threshold_step = 0.02  # Amount to lower threshold
        self.hero_threshold_frame_interval = 60  # Lower threshold every N frames (1 second at ~60fps)
        self.hero_threshold_frame_counter = 0
        self.hero_confirmation_window = 5.0  # Look at last 5 seconds of detections
        self.hero_confirmation_count = 3  # Need only 3 detections to confirm (since we're at low FPS)
        self.legal_card_names = None  # Set of legal card names (lowercase)
        
        # Reset button (clickable area in top-right)
        self.reset_button_bounds = None  # Will be (x1, y1, x2, y2)
        
        # Hover tracking
        self.mouse_x = 0
        self.mouse_y = 0
        self.current_detections = []
        self.hovered_card = None
        self.hovered_card_bbox = None  # (x1, y1, x2, y2) of hovered card
        self.preview_window_name = "Card Preview"
        
        # Get monitor info
        self.monitor = self.sct.monitors[self.capture_monitor]
        self.full_width = self.monitor["width"]
        self.full_height = self.monitor["height"]
        
        # Detection size (half resolution)
        self.detect_width = int(self.full_width * DETECTION_SCALE)
        self.detect_height = int(self.full_height * DETECTION_SCALE)
        
        print(f"Capture Monitor: {self.capture_monitor}")
        print(f"Full Resolution: {self.full_width}x{self.full_height}")
        print(f"Detection Resolution: {self.detect_width}x{self.detect_height}")
        print(f"Bounding Boxes: {'ON' if self.show_boxes else 'OFF'}")
        print(f"Hero Detection: Active (threshold: {self.hero_detection_threshold:.2f}, needs {self.hero_confirmation_count} detections)")
        
    def load_model(self):
        """Load the YOLO model"""
        print(f"Loading model from {self.model_path}...")
        self.model = YOLO(self.model_path)
        print("Model loaded successfully!")
    
    def _is_hero_card(self, card_name):
        """Check if a detected card is a hero."""
        search_name = card_name.replace('_', ' ').lower().strip()
        
        for card in self.image_cache.card_data.values():
            card_display_name = card.get('name', '').lower().strip()
            if card_display_name == search_name:
                types = card.get('types', [])
                return 'Hero' in types or 'Young Hero' in types
        return False
    
    def _build_legal_card_pool(self, hero_name):
        """Build set of legal card names for a given hero."""
        search_name = hero_name.replace('_', ' ').lower().strip()
        hero_card = None
        
        for card in self.image_cache.card_data.values():
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
        
        hero_types = set(hero_card.get('types', []))
        hero_classes = hero_types & ALL_CLASSES
        hero_talents = hero_types & ALL_TALENTS
        
        # Check for "Essence of X" keywords
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
        
        legal_cards = set()
        for card in self.image_cache.card_data.values():
            card_name = card.get('name', '')
            card_types = set(card.get('types', []))
            card_classes = card_types & ALL_CLASSES
            card_talents = card_types & ALL_TALENTS
            
            if 'Hero' in card_types or 'Young Hero' in card_types:
                continue
            
            if 'Generic' in card_types or 'Token' in card_types:
                legal_cards.add(card_name.lower())
                continue
            
            if not hero_talents and card_talents:
                continue
            
            card_classes_match = not card_classes or bool(card_classes & hero_classes)
            card_talents_match = not card_talents or bool(card_talents & hero_talents)
            
            if not card_classes_match or not card_talents_match:
                continue
            
            legal_cards.add(card_name.lower())
        
        print(f"[hero] Legal pool size for {hero_name}: {len(legal_cards)} cards")
        return legal_cards
    
    def _rebuild_legal_pool(self):
        """Rebuild the legal card pool based on detected heroes."""
        if not self.detected_hero1 and not self.detected_hero2:
            self.legal_card_names = None
            return
        
        pool1 = self._build_legal_card_pool(self.detected_hero1) if self.detected_hero1 else set()
        pool2 = self._build_legal_card_pool(self.detected_hero2) if self.detected_hero2 else set()
        
        self.legal_card_names = pool1 | pool2
        print(f"[hero] Combined legal pool: {len(self.legal_card_names)} cards")
    
    def reset_hero_detection(self):
        """Reset hero auto-detection to start fresh."""
        self.detected_hero1 = None
        self.detected_hero2 = None
        self.hero1_confidence_history = []
        self.hero2_confidence_history = []
        self.legal_card_names = None
        self.hero_detection_threshold = 0.15
        self.hero_threshold_frame_counter = 0
        print("[hero] Detection reset - starting fresh with threshold 0.15")
    
    def _apply_hero_filtering(self, detections, current_time):
        """Apply hero detection and filtering. Returns filtered detections list."""
        if len(detections) == 0:
            return detections
        
        # Dynamic threshold adjustment
        if (self.detected_hero1 is None or self.detected_hero2 is None):
            self.hero_threshold_frame_counter += 1
            if self.hero_threshold_frame_counter >= self.hero_threshold_frame_interval:
                if self.hero_detection_threshold > self.hero_threshold_min:
                    self.hero_detection_threshold = max(
                        self.hero_threshold_min,
                        self.hero_detection_threshold - self.hero_threshold_step
                    )
                    print(f"[hero] Lowering detection threshold to {self.hero_detection_threshold:.2f}")
                    self.hero_threshold_frame_counter = 0
        else:
            self.hero_threshold_frame_counter = 0
        
        # Look for hero cards in detections
        for x1, y1, x2, y2, card_name, conf in detections:
            if self._is_hero_card(card_name) and conf >= self.hero_detection_threshold:
                # Add to appropriate history based on which hero is missing
                if self.detected_hero1 is None:
                    self.hero1_confidence_history.append((card_name, conf, current_time))
                    print(f"[hero] Hero 1 candidate: {card_name} (conf: {conf:.2f}, history: {len(self.hero1_confidence_history)})")
                elif self.detected_hero2 is None and card_name != self.detected_hero1:
                    self.hero2_confidence_history.append((card_name, conf, current_time))
                    print(f"[hero] Hero 2 candidate: {card_name} (conf: {conf:.2f}, history: {len(self.hero2_confidence_history)})")
        
        # Clean old history entries
        cutoff_time = current_time - self.hero_confirmation_window
        self.hero1_confidence_history = [(n, c, t) for n, c, t in self.hero1_confidence_history if t > cutoff_time]
        self.hero2_confidence_history = [(n, c, t) for n, c, t in self.hero2_confidence_history if t > cutoff_time]
        
        # Try to confirm Hero 1
        if self.detected_hero1 is None and len(self.hero1_confidence_history) >= self.hero_confirmation_count:
            from collections import Counter
            name_counts = Counter([name for name, _, _ in self.hero1_confidence_history])
            most_common_name, count = name_counts.most_common(1)[0]
            if count >= self.hero_confirmation_count:
                self.detected_hero1 = most_common_name
                print(f"[hero] Detected Hero 1: {self.detected_hero1}")
                self._rebuild_legal_pool()
        
        # Try to confirm Hero 2
        if self.detected_hero2 is None and len(self.hero2_confidence_history) >= self.hero_confirmation_count:
            from collections import Counter
            name_counts = Counter([name for name, _, _ in self.hero2_confidence_history])
            most_common_name, count = name_counts.most_common(1)[0]
            if count >= self.hero_confirmation_count:
                self.detected_hero2 = most_common_name
                print(f"[hero] Detected Hero 2: {self.detected_hero2}")
                self._rebuild_legal_pool()
        
        # Filter detections based on legal pool
        if self.legal_card_names is not None:
            filtered = []
            filtered_count = 0
            for detection in detections:
                x1, y1, x2, y2, card_name, conf = detection
                search_name = card_name.replace('_', ' ').lower().strip()
                
                # Allow heroes through
                if self._is_hero_card(card_name):
                    filtered.append(detection)
                # Check if card is legal
                elif search_name in self.legal_card_names:
                    filtered.append(detection)
                else:
                    filtered_count += 1
            
            if filtered_count > 0:
                print(f"[hero] Filtered {filtered_count} illegal cards")
            
            return filtered
        
        return detections
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse movement and clicks"""
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_x = x
            self.mouse_y = y
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Check if clicked on reset button
            if self.reset_button_bounds:
                x1, y1, x2, y2 = self.reset_button_bounds
                if x1 <= x <= x2 and y1 <= y <= y2:
                    print("Reset Heroes button clicked!")
                    self.reset_hero_detection()
            
    def check_hover(self):
        """Check if mouse is hovering over any detection. Returns (card_name, bbox) or (None, None)"""
        for detection in self.current_detections:
            x1, y1, x2, y2, card_name, conf = detection
            if x1 <= self.mouse_x <= x2 and y1 <= self.mouse_y <= y2:
                return card_name, (x1, y1, x2, y2)
        return None, None
        
    def show_card_preview(self, card_name, bbox):
        """Show card preview positioned near the detected card"""
        if card_name != self.hovered_card:
            self.hovered_card = card_name
            self.hovered_card_bbox = bbox
            
            if card_name:
                # Try to get cached or download image (smaller size: 300px height)
                img = self.image_cache.get_image(card_name, max_height=300)
                
                if img is not None:
                    # Position the preview window near the card
                    if bbox:
                        x1, y1, x2, y2 = bbox
                        card_width = x2 - x1
                        card_height = y2 - y1
                        
                        # Try to position to the right of the card
                        # If not enough room, position to the left
                        preview_x = x2 + 20  # 20px margin
                        if preview_x + img.shape[1] > self.full_width:
                            # Not enough room on right, try left
                            preview_x = x1 - img.shape[1] - 20
                            if preview_x < 0:
                                # Not enough room on either side, position below
                                preview_x = x1
                        
                        # Set window position (note: this only works on some platforms)
                        cv2.namedWindow(self.preview_window_name, cv2.WINDOW_NORMAL)
                        cv2.moveWindow(self.preview_window_name, preview_x, y1)
                    
                    # Show in preview window
                    cv2.imshow(self.preview_window_name, img)
                else:
                    # Show placeholder with card name
                    placeholder = np.zeros((200, 300, 3), dtype=np.uint8)
                    
                    # Word wrap the card name
                    words = card_name.split()
                    lines = []
                    current_line = ""
                    
                    for word in words:
                        test_line = current_line + " " + word if current_line else word
                        (w, h), _ = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        
                        if w < 280:
                            current_line = test_line
                        else:
                            if current_line:
                                lines.append(current_line)
                            current_line = word
                    
                    if current_line:
                        lines.append(current_line)
                    
                    # Draw text lines
                    y_offset = 80
                    for line in lines:
                        (w, h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        x = (300 - w) // 2
                        cv2.putText(placeholder, line, (x, y_offset), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        y_offset += 25
                    
                    cv2.putText(placeholder, "Image not available", (50, 180), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
                    
                    cv2.imshow(self.preview_window_name, placeholder)
            else:
                # Close preview window if not hovering
                try:
                    cv2.destroyWindow(self.preview_window_name)
                except:
                    pass
                
    def show_help(self):
        """Print keyboard shortcuts"""
        print("\n" + "="*50)
        print("KEYBOARD SHORTCUTS")
        print("="*50)
        print("Q: Quit")
        print("H: Show this help")
        print("B: Toggle bounding boxes")
        print("R: Reset hero detection")
        print("+/=: Increase confidence (+0.05)")
        print("-/_: Decrease confidence (-0.05)")
        print("[: Decrease IOU (-0.05)")
        print("]: Increase IOU (+0.05)")
        print("\nHERO DETECTION")
        print("="*50)
        print("Auto-detects 2 heroes and filters illegal cards")
        print("Click 'Reset Heroes' button (top-right) or press R")
        print("\nHOVER CONTROLS")
        print("="*50)
        print("Move mouse over detection to see card preview")
        print("="*50 + "\n")
        
    def draw_overlay(self, frame):
        """Draw status overlay on frame"""
        # Dark semi-transparent background for text (left side)
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 180), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Status text
        y_pos = 35
        cv2.putText(frame, f"Confidence: {self.confidence:.2f}", (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 25
        cv2.putText(frame, f"IOU: {self.iou_threshold:.2f}", (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 25
        cv2.putText(frame, f"Boxes: {'ON' if self.show_boxes else 'OFF'}", (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 25
        cv2.putText(frame, f"Detections: {len(self.current_detections)}", (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Hero status
        y_pos += 30
        hero1_text = f"Hero 1: {self.detected_hero1 if self.detected_hero1 else 'Detecting...'}"
        cv2.putText(frame, hero1_text, (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_pos += 25
        hero2_text = f"Hero 2: {self.detected_hero2 if self.detected_hero2 else 'Detecting...'}"
        cv2.putText(frame, hero2_text, (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Reset Heroes button (top-right corner)
        button_width = 180
        button_height = 50
        button_x1 = frame.shape[1] - button_width - 20
        button_y1 = 20
        button_x2 = button_x1 + button_width
        button_y2 = button_y1 + button_height
        
        # Store button bounds for click detection
        self.reset_button_bounds = (button_x1, button_y1, button_x2, button_y2)
        
        # Check if mouse is hovering over button
        is_hovering = (button_x1 <= self.mouse_x <= button_x2 and 
                      button_y1 <= self.mouse_y <= button_y2)
        
        # Draw button
        button_color = (0, 200, 255) if is_hovering else (0, 150, 200)
        cv2.rectangle(frame, (button_x1, button_y1), (button_x2, button_y2), button_color, -1)
        cv2.rectangle(frame, (button_x1, button_y1), (button_x2, button_y2), (255, 255, 255), 2)
        
        # Button text
        button_text = "Reset Heroes (R)"
        (text_w, text_h), _ = cv2.getTextSize(button_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        text_x = button_x1 + (button_width - text_w) // 2
        text_y = button_y1 + (button_height + text_h) // 2
        cv2.putText(frame, button_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Instructions at bottom
        cv2.putText(frame, "Hover over cards for preview | Press H for help | Q to quit", 
                   (10, frame.shape[0] - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
        
    def run(self):
        """Main detection loop"""
        if self.model is None:
            self.load_model()
            
        self.running = True
        self.show_help()
        
        window_name = "FaB Card Detector (Hover Mode)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        fps_start_time = time.time()
        fps_counter = 0
        fps_display = 0.0
        
        print("Starting detection... Hover over cards to preview. Press Q to quit")
        
        try:
            while self.running:
                # Capture screen
                screenshot = self.sct.grab(self.monitor)
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
                # Resize for detection (speed optimization)
                detect_frame = cv2.resize(frame, (self.detect_width, self.detect_height))
                
                # Run detection
                results = self.model(detect_frame, conf=self.confidence, iou=self.iou_threshold, verbose=False)
                
                # Store detections
                self.current_detections = []
                current_time = time.time()
                
                # Collect all detections first
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        # Get box coordinates (scaled back to full resolution)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1 = int(x1 / DETECTION_SCALE)
                        y1 = int(y1 / DETECTION_SCALE)
                        x2 = int(x2 / DETECTION_SCALE)
                        y2 = int(y2 / DETECTION_SCALE)
                        
                        # Get confidence and class
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        card_name = self.model.names[cls]
                        
                        # Store detection
                        self.current_detections.append((x1, y1, x2, y2, card_name, conf))
                
                # Apply hero detection and filtering
                self.current_detections = self._apply_hero_filtering(self.current_detections, current_time)
                
                # Draw filtered detections on full-size frame
                for x1, y1, x2, y2, card_name, conf in self.current_detections:
                    # Draw bounding box if enabled
                    if self.show_boxes:
                        label = f"{card_name} {conf:.2f}"
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw label background
                        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), (0, 255, 0), -1)
                        
                        # Draw label text
                        cv2.putText(frame, label, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                # Check hover and show preview
                hovered_card, hovered_bbox = self.check_hover()
                self.show_card_preview(hovered_card, hovered_bbox)
                
                # Calculate FPS
                fps_counter += 1
                if time.time() - fps_start_time >= 1.0:
                    fps_display = fps_counter / (time.time() - fps_start_time)
                    fps_counter = 0
                    fps_start_time = time.time()
                
                # Draw FPS
                cv2.putText(frame, f"FPS: {fps_display:.1f}", (frame.shape[1] - 150, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Draw status overlay
                frame = self.draw_overlay(frame)
                
                # Show frame
                cv2.imshow(window_name, frame)
                
                # Handle keyboard input (MUST have cv2.waitKey for window to update!)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    print("Quitting...")
                    self.running = False
                    break
                    
                elif key == ord('h') or key == ord('H'):
                    self.show_help()
                    
                elif key == ord('b') or key == ord('B'):
                    self.show_boxes = not self.show_boxes
                    print(f"Bounding Boxes: {'ON' if self.show_boxes else 'OFF'}")
                
                elif key == ord('r') or key == ord('R'):
                    print("Resetting hero detection...")
                    self.reset_hero_detection()
                    
                elif key == ord('+') or key == ord('='):
                    self.confidence = max(0.05, min(0.95, self.confidence + 0.05))
                    print(f"Confidence: {self.confidence:.2f}")
                    
                elif key == ord('-') or key == ord('_'):
                    self.confidence = max(0.05, min(0.95, self.confidence - 0.05))
                    print(f"Confidence: {self.confidence:.2f}")
                    
                elif key == ord('['):
                    self.iou_threshold = max(0.1, min(0.9, self.iou_threshold - 0.05))
                    print(f"IOU: {self.iou_threshold:.2f}")
                    
                elif key == ord(']'):
                    self.iou_threshold = max(0.1, min(0.9, self.iou_threshold + 0.05))
                    print(f"IOU: {self.iou_threshold:.2f}")
                        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            
        finally:
            cv2.destroyAllWindows()
            print("Detection stopped")

if __name__ == "__main__":
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        exit(1)
    
    # Create and run detector
    detector = FaBDetectorHover(MODEL_PATH, CARD_JSON_PATH, CAPTURE_MONITOR)
    detector.run()
