from ultralytics import YOLO
import cv2
import numpy as np
import importlib
# Dynamically import mss to avoid static import errors if not installed
mss = None
try:
    mss = importlib.import_module('mss')
except Exception:
    mss = None
import json
import os
import requests
from PIL import Image
from io import BytesIO
import argparse
import time
import ctypes
from ctypes import wintypes

"""
screen_detect.py

Overlay-oriented detector UI:
- Captures frames (screen via MSS or a video file), runs YOLO, and shows a minimal
    overlay window that is empty except for the currently selected card preview.
- Prevents "feedback" detection loops by optionally masking the previous overlay
    region out of the next captured frame before inference.

Notes:
- True transparent/click-through overlays require OS-specific APIs; here we keep a
    minimal black background window so only the card stands out. If you want a fully
    transparent desktop overlay, we can wire up a Win32 layered window as a follow-up.
"""


def build_argparser():
    parser = argparse.ArgumentParser(description="Screen/Video detection overlay")
    parser.add_argument("--conf", type=float, default=0.69,
        help="Confidence threshold for detection (default: 0.69, from F1 curve)")
    parser.add_argument("--iou", type=float, default=0.50,
        help="IoU threshold for NMS (default: 0.50)")
    parser.add_argument("--imgsz", type=int, default=640,
        help="Image size for inference (default: 640)")
    parser.add_argument("--weights", type=str,
                        default=os.path.join("runs", "detect", "train13", "weights", "best.pt"),
                        help="Path to YOLO weights (.pt)")
    parser.add_argument("--capture-monitor", type=int, default=1,
                        help="Monitor index to capture with MSS (1-based). Ignored when --video is set.")
    parser.add_argument("--display-monitor", type=int, default=None,
                        help="Monitor index to place the overlay window. Default picks the other monitor if available.")
    parser.add_argument("--video", type=str, default=None,
                        help="Optional: path to a video file instead of screen capture")
    parser.add_argument("--overlay-only", action="store_true", default=True,
                        help="Render a minimal overlay window (black background) with only the card image")
    parser.add_argument("--no-overlay-only", dest="overlay_only", action="store_false",
                        help="Disable minimal overlay; draw on top of captured frame (legacy mode)")
    parser.add_argument("--mask-overlay", action="store_true", default=True,
                        help="Black out the previous overlay region in the next frame before detection to avoid feedback")
    parser.add_argument("--no-mask-overlay", dest="mask_overlay", action="store_false",
                        help="Disable overlay masking")
    parser.add_argument("--card-size", type=int, nargs=2, metavar=("W", "H"), default=(300, 420),
                        help="Overlay card width height in pixels")
    parser.add_argument("--topmost", action="store_true", default=True,
                        help="Keep overlay window always-on-top (Windows only)")
    parser.add_argument("--no-topmost", dest="topmost", action="store_false",
                        help="Disable always-on-top for the overlay window")
    parser.add_argument("--transparent", action="store_true", default=True,
                        help="Use chroma-key transparency so only the card shows (Windows layered window)")
    parser.add_argument("--no-transparent", dest="transparent", action="store_false",
                        help="Disable window transparency (overlay window will be opaque)")
    parser.add_argument("--chroma", type=int, nargs=3, metavar=("R","G","B"), default=(255, 0, 255),
                        help="Chroma-key RGB color to make transparent (default magenta)")
    parser.add_argument("--click-through", action="store_true", default=False,
                        help="Make overlay window ignore mouse clicks (Windows only)")
    return parser


def move_window_to_monitor(window_name: str, monitor_rect: dict, offset_xy=(100, 100)):
        x = int(monitor_rect.get("left", 0) + offset_xy[0])
        y = int(monitor_rect.get("top", 0) + offset_xy[1])
        cv2.moveWindow(window_name, x, y)


def set_window_topmost(window_name: str, topmost: bool = True):
    """Set the OpenCV window always-on-top using Win32 APIs on Windows."""
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, window_name)
        if hwnd:
            HWND_TOPMOST = -1
            HWND_NOTOPMOST = -2
            SWP_NOSIZE = 0x0001
            SWP_NOMOVE = 0x0002
            SWP_NOACTIVATE = 0x0010
            user32.SetWindowPos(hwnd,
                                HWND_TOPMOST if topmost else HWND_NOTOPMOST,
                                0, 0, 0, 0,
                                SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE)
    except Exception:
        # Non-fatal: if anything goes wrong, we just skip topmost
        pass


def enable_chromakey_transparency(window_name: str, rgb=(255, 0, 255), click_through=False):
    """Make an OpenCV window chroma-key transparent using WS_EX_LAYERED + LWA_COLORKEY.
    Note: Any pixel matching the chroma color will be invisible. Choose a rare color.
    """
    try:
        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32
        hwnd = user32.FindWindowW(None, window_name)
        if not hwnd:
            return
        GWL_EXSTYLE = -20
        WS_EX_LAYERED = 0x00080000
        WS_EX_TRANSPARENT = 0x00000020
        LWA_COLORKEY = 0x00000001
        # get and set layered style
        current = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        new_style = current | WS_EX_LAYERED
        if click_through:
            new_style |= WS_EX_TRANSPARENT
        ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_style)
        # pack COLORREF via RGB macro (r | g<<8 | b<<16)
        r, g, b = rgb
        colorref = (r & 0xFF) | ((g & 0xFF) << 8) | ((b & 0xFF) << 16)
        ctypes.windll.user32.SetLayeredWindowAttributes(hwnd, colorref, 0, LWA_COLORKEY)
    except Exception:
        # Non-fatal; fall back to opaque overlay
        pass


def get_global_mouse_pos():
    pt = wintypes.POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return (pt.x, pt.y)

# Mouse position tracking
mouse_pos = (0, 0)
def mouse_callback(event, x, y, flags, param):
    global mouse_pos
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_pos = (x, y)

# Load card data once at module import
with open(os.path.join('data', 'card.json'), 'r', encoding='utf-8') as f:
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

def main():
    args = build_argparser().parse_args()

    print("Starting screen_detect.py")

    # Load model weights with friendly error if missing/invalid
    try:
        model = YOLO(args.weights)
        print("Model loaded successfully")
        print(f"[INFO] weights: {args.weights}")
        print(f"[INFO] imgsz/conf/iou: {args.imgsz}/{args.conf}/{args.iou}")
        print(f"[INFO] classes: {len(model.names)} → {list(model.names.values())[:5]} ...")
    except Exception as e:
        print(f"Failed to load weights '{args.weights}': {e}")
        print("Verify the path to your .pt file or train has completed and produced best.pt")
        return


    # Tracking the last overlay rectangle to mask it from detection
    last_overlay_rect = None  # (x, y, w, h) in display window space

    window_name = 'YOLO Overlay'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    print("Window created")

    if args.video:
        if not os.path.exists(args.video):
            print(f"Video not found: {args.video}")
            return
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"Failed to open video: {args.video}")
            return
        # Size for overlay background when overlay_only is True
        ret, tmp = cap.read()
        if not ret:
            print("Empty video stream")
            return
        base_h, base_w = tmp.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        sct = None
        display_rect = {"left": 0, "top": 0}
    else:
        if mss is None:
            print("Missing dependency: mss. Please install it with 'pip install mss' or run with --video")
            return
        sct = mss.mss()
        monitors = sct.monitors  # 1-based index
        cap_mon = max(1, min(args.capture_monitor, len(monitors) - 1))
        monitor = monitors[cap_mon]
        # pick display monitor: same as capture by default, or other if available
        if args.display_monitor is None:
            if len(monitors) > 2:
                disp_mon = 2 if cap_mon == 1 else 1
            else:
                disp_mon = cap_mon  # default to same monitor
        else:
            disp_mon = max(1, min(args.display_monitor, len(monitors) - 1))
        display_info = monitors[disp_mon]
        base_h = monitor['height']
        base_w = monitor['width']
        display_rect = display_info
        same_monitor = (disp_mon == cap_mon)

    # Place window on chosen monitor and remember its screen origin
    window_offset = (100, 100)
    move_window_to_monitor(window_name, display_rect, window_offset)
    window_screen_left = display_rect.get('left', 0) + window_offset[0]
    window_screen_top = display_rect.get('top', 0) + window_offset[1]
    # Resize to the capture dimensions so placement feels natural
    try:
        cv2.resizeWindow(window_name, int(base_w), int(base_h))
    except Exception:
        pass
    # Keep the overlay above other windows if desired
    set_window_topmost(window_name, topmost=args.topmost)
    # Enable chroma-key transparency if requested
    if args.overlay_only and args.transparent:
        enable_chromakey_transparency(window_name, rgb=tuple(args.chroma), click_through=args.click_through)

    # card preview size
    card_w, card_h = args.card_size
    names_base = model.names if hasattr(model, 'names') else {}

    while True:
        print("Loop iteration started")
        # 1) Capture the next frame
        if args.video:
            ok, frame = cap.read()
            if not ok:
                break
        else:
            img = np.array(sct.grab(monitor))  # BGRA
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # 2) If enabled, mask the last overlay from the frame BEFORE inference
        if args.mask_overlay and last_overlay_rect is not None:
            ox, oy, ow, oh = last_overlay_rect
            # Clamp to frame bounds just in case
            x1, y1 = max(0, ox), max(0, oy)
            x2, y2 = min(frame.shape[1], ox + ow), min(frame.shape[0], oy + oh)
            if x2 > x1 and y2 > y1:
                frame[y1:y2, x1:x2] = 0  # black out

        # 3) Run detection (robust)
        boxes, clss = [], []
        names = names_base
        try:
            results = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                agnostic_nms=False,
                max_det=300,
                verbose=False
            )
            if results:
                r0 = results[0]
                if hasattr(r0, 'boxes') and r0.boxes is not None:
                    # xyxy (N,4), cls (N,)
                    boxes = r0.boxes.xyxy.detach().cpu().numpy() if hasattr(r0.boxes, 'xyxy') else []
                    clss = r0.boxes.cls.detach().cpu().numpy() if hasattr(r0.boxes, 'cls') else []
                # Prefer per-result names, else model-level
                if hasattr(r0, 'names') and r0.names:
                    names = r0.names
        except Exception as e:
            # Non-fatal: skip drawing this frame's detections
            # print(f"Detection error: {e}")
            boxes, clss = [], []

        # (Optional) Post-filter for aspect ratio and area
        H, W = frame.shape[:2]
        keep = []
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            ar = h / w  # cards are tall-ish
            area = (w * h) / (W * H)  # relative to screen
            if 0.6 <= ar <= 2.2 and 0.004 <= area <= 0.15:
                keep.append(i)
        boxes = boxes[keep]
        clss = clss[keep]

        # 4) Determine if mouse hovers a box
        mouse_over_box = None
        if args.video:
            mx, my = mouse_pos
        else:
            gx, gy = get_global_mouse_pos()
            mx = gx - monitor['left']
            my = gy - monitor['top']
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(v) for v in box]
            class_id = int(clss[i]) if len(clss) > i else -1
            # Robust label lookup for dict or list names
            if isinstance(names, dict):
                label = names.get(class_id, str(class_id))
            elif isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
                label = str(names[class_id])
            else:
                label = str(class_id)
            if x1 <= mx <= x2 and y1 <= my <= y2:
                mouse_over_box = (x1, y1, x2, y2, label)
                break  # first match is fine

        # 5) Build the display frame
        if args.overlay_only:
            # Fill background with chroma color when transparent, else black
            if args.transparent:
                r, g, b = args.chroma
                display_frame = np.full((base_h, base_w, 3), (b, g, r), dtype=np.uint8)  # BGR
            else:
                display_frame = np.zeros((base_h, base_w, 3), dtype=np.uint8)
        else:
            # Legacy mode: show captured frame with annotations
            display_frame = frame.copy()

        # 6) Draw all detected boxes and labels (legacy mode)
        if not args.overlay_only:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = [int(v) for v in box]
                class_id = int(clss[i]) if len(clss) > i else -1
                if isinstance(names, dict):
                    label = names.get(class_id, str(class_id))
                elif isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
                    label = str(names[class_id])
                else:
                    label = str(class_id)
                # Draw box and label
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                font_scale = 0.7
                thickness = 2
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                cv2.rectangle(display_frame, (x1, y1 - h - 10), (x1 + w, y1), (0, 255, 0), -1)
                cv2.putText(display_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

        # 7) Only show card image overlay when mouse is hovering a box (both modes)
        next_overlay_rect = None
        if mouse_over_box:
            bx1, by1, bx2, by2, class_name = mouse_over_box
            card_id = class_name[-6:] if len(class_name) >= 6 else class_name
            image_url = get_image_url_by_id(card_id)
            if image_url:
                card_img = get_card_image(image_url)
                if card_img:
                    card_img = card_img.resize((card_w, card_h))
                    card_np = np.array(card_img)
                    card_np = cv2.cvtColor(card_np, cv2.COLOR_RGBA2BGRA)
                    # place below the box, offset if needed
                    overlay_x = max(0, min(bx1, display_frame.shape[1] - card_np.shape[1]))
                    overlay_y = max(0, min(by2 + 10, display_frame.shape[0] - card_np.shape[0]))
                    roi = display_frame[overlay_y:overlay_y + card_np.shape[0], overlay_x:overlay_x + card_np.shape[1]]
                    if roi.shape[2] == 3:
                        roi_bgra = cv2.cvtColor(roi, cv2.COLOR_BGR2BGRA)
                    else:
                        roi_bgra = roi.copy()
                    alpha_card = (card_np[:, :, 3] / 255.0)[:, :, None]
                    roi_bgra[:, :, :3] = (alpha_card * card_np[:, :, :3] + (1 - alpha_card) * roi_bgra[:, :, :3]).astype(np.uint8)
                    display_frame[overlay_y:overlay_y + card_np.shape[0], overlay_x:overlay_x + card_np.shape[1]] = roi_bgra[:, :, :3]
                    next_overlay_rect = (overlay_x, overlay_y, card_np.shape[1], card_np.shape[0])

        # 8) Remember rect for masking on next iteration (if video or different monitor, do not mask)
        if args.video or ('same_monitor' in locals() and not same_monitor):
            last_overlay_rect = None
        elif next_overlay_rect is not None:
            # Coordinates align only when display and capture are the same monitor
            last_overlay_rect = next_overlay_rect

        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    if not args.video and 'sct' in locals() and sct is not None:
        sct.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()