
from ultralytics import YOLO
import cv2
import numpy as np
import mss
import json
import requests
from PIL import Image
from io import BytesIO

model = YOLO('runs/detect/train12/weights/best.pt')

# Mouse position tracking
mouse_pos = (0, 0)
def mouse_callback(event, x, y, flags, param):
    global mouse_pos
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_pos = (x, y)

# Load card data
with open('data/card.json', 'r', encoding='utf-8') as f:
    card_data = json.load(f)

 # Helper to get image_url for a card id (from class name)
def get_image_url_by_id(card_id):
    for card in card_data:
        for printing in card.get('printings', []):
            if printing.get('id', '').upper() == card_id.upper():
                return printing.get('image_url')
    return None

# Helper to load card images from URL and cache them
card_image_cache = {}
def get_card_image(url):
    if url in card_image_cache:
        return card_image_cache[url]
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert('RGBA')
        card_image_cache[url] = img
        return img
    except Exception:
        return None

with mss.mss() as sct:
    capture_monitor = 1  # Change to 2 to capture monitor 2
    monitor = sct.monitors[capture_monitor]
    print(sct.monitors)
    display_monitor = 2 if capture_monitor == 1 else 1
    display_info = sct.monitors[display_monitor]

    cv2.namedWindow('YOLO Screen Capture', cv2.WINDOW_NORMAL)
    cv2.moveWindow('YOLO Screen Capture', display_info["left"] + 100, display_info["top"] + 100)
    cv2.setMouseCallback('YOLO Screen Capture', mouse_callback)

    while True:
        img = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        results = model(frame)
        boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []
        names = results[0].names if hasattr(results[0], 'names') else {}
        clss = results[0].boxes.cls.cpu().numpy() if results[0].boxes is not None else []

        # Draw boxes, but only show label if mouse is over the box
        mouse_over_box = None
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(v) for v in box]
            class_id = int(clss[i]) if len(clss) > i else -1
            label = names[class_id] if class_id in names else str(class_id)
            # Only draw rectangle if mouse IS inside box
            mx, my = mouse_pos
            if x1 <= mx <= x2 and y1 <= my <= y2:
                mouse_over_box = (x1, y1, x2, y2, label)
                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                font_scale = 0.7
                thickness = 2
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

        # Overlay card image if mouse is over a box
        if mouse_over_box:
            _, _, _, _, class_name = mouse_over_box
            card_id = class_name[-6:] if len(class_name) >= 6 else class_name
            image_url = get_image_url_by_id(card_id)
            if image_url:
                card_img = get_card_image(image_url)
                if card_img:
                    # Resize for overlay (now 300x420)
                    card_img = card_img.resize((300, 420))
                    card_np = np.array(card_img)
                    # Convert RGBA to BGRA for OpenCV
                    card_np = cv2.cvtColor(card_np, cv2.COLOR_RGBA2BGRA)
                    # Overlay at mouse position, offset so it doesn't cover cursor
                    overlay_x = min(mx + 20, frame.shape[1] - card_np.shape[1])
                    overlay_y = min(my + 20, frame.shape[0] - card_np.shape[0])
                    # Blend overlay
                    overlay = frame[overlay_y:overlay_y+card_np.shape[0], overlay_x:overlay_x+card_np.shape[1]]
                    alpha_card = card_np[:, :, 3] / 255.0
                    alpha_bg = 1.0 - alpha_card
                    for c in range(3):
                        overlay[:, :, c] = (alpha_card * card_np[:, :, c] + alpha_bg * overlay[:, :, c]).astype(np.uint8)
                    frame[overlay_y:overlay_y+card_np.shape[0], overlay_x:overlay_x+card_np.shape[1]] = overlay

        cv2.imshow('YOLO Screen Capture', frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty('YOLO Screen Capture', cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()