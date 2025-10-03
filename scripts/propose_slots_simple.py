# propose_slots_simple.py
# Runs your YOLO model on unlabeled backgrounds, proposes card-like boxes (high recall),
# writes YOLO .txt labels (dummy class 0) next to copied images in your existing template tree,
# and saves overlay previews so you can spot-check.

import os, glob, json, argparse
import cv2
import numpy as np
from ultralytics import YOLO

def iou_xyxy(a, b):
    x1, y1, x2, y2 = a
    X1, Y1, X2, Y2 = b
    xi1, yi1 = max(x1, X1), max(y1, Y1)
    xi2, yi2 = min(x2, X2), min(y2, Y2)
    iw, ih = max(0, xi2 - xi1), max(0, yi2 - yi1)
    inter = iw * ih
    ua = (x2 - x1) * (y2 - y1) + (X2 - X1) * (Y2 - Y1) - inter
    return inter / ua if ua > 0 else 0.0

def dedup(boxes, iou_t=0.20):
    kept = []
    for b in boxes:
        if all(iou_xyxy(b, k) <= iou_t for k in kept):
            kept.append(b)
    return kept

def xyxy_to_yolo(x1,y1,x2,y2,W,H):
    w = (x2-x1)/W; h = (y2-y1)/H
    cx = (x1+x2)/(2*W); cy = (y1+y2)/(2*H)
    return cx,cy,w,h

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to your trained YOLO weights (best.pt)")
    ap.add_argument("--input_dir", required=True, help="Folder with unlabeled backgrounds")
    ap.add_argument("--template_base", default=r"data/images/YouTube_Labeled/FaB Card Detection.v4i.yolov11",
                    help="Existing template base (train/valid/test/images|labels)")
    ap.add_argument("--split", default="train", choices=["train","valid","test"], help="Which split to add to")
    ap.add_argument("--conf", type=float, default=0.20, help="Low-ish conf for recall")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--max_slots", type=int, default=30)
    ap.add_argument("--out_jsonl", default=r"data/slot_candidates.jsonl")
    args = ap.parse_args()

    # Card-ish geometry filters (tweak if needed)
    AR_MIN, AR_MAX = 1.25, 1.55        # h/w ≈ 1.40 ± ~10%
    AREA_MIN, AREA_MAX = 0.005, 0.50   # 0.5%–50% of image
    IOU_DEDUP = 0.25

    out_img_dir = os.path.join(args.template_base, args.split, "images")
    out_lbl_dir = os.path.join(args.template_base, args.split, "labels")
    overlay_dir = os.path.join("overlays", args.split)
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    model = YOLO(args.model)

    images = []
    for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
        images += glob.glob(os.path.join(args.input_dir, ext))
    images.sort()

    total_boxes = 0
    with open(args.out_jsonl, "w", encoding="utf-8") as jout:
        for ip in images:
            img = cv2.imread(ip)
            if img is None:
                print(f"Skip unreadable: {ip}")
                continue
            H, W = img.shape[:2]

            r = model.predict(
                source=img, device="cpu", conf=args.conf, iou=0.5,
                agnostic_nms=True, max_det=300, imgsz=args.imgsz, verbose=False
            )[0]

            cand = []
            if r.boxes is not None and len(r.boxes) > 0:
                for xyxy in r.boxes.xyxy.cpu().numpy():
                    x1,y1,x2,y2 = map(float, xyxy)
                    w,h = x2-x1, y2-y1
                    if w<=0 or h<=0: continue
                    ar = h / w
                    area_frac = (w*h)/(W*H)
                    if (AR_MIN <= ar <= AR_MAX) and (AREA_MIN <= area_frac <= AREA_MAX):
                        cand.append([x1,y1,x2,y2])

            cand = dedup(cand, IOU_DEDUP)[:args.max_slots]
            total_boxes += len(cand)

            # overlay
            viz = img.copy()
            for (x1,y1,x2,y2) in cand:
                cv2.rectangle(viz, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
            ov_name = os.path.join(overlay_dir, os.path.splitext(os.path.basename(ip))[0] + "_ov.jpg")
            cv2.imwrite(ov_name, viz, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

            # copy image into template tree and write label file (dummy class 0)
            base = os.path.splitext(os.path.basename(ip))[0]
            out_img_path = os.path.join(out_img_dir, base + ".jpg")
            cv2.imwrite(out_img_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

            out_lbl_path = os.path.join(out_lbl_dir, base + ".txt")
            with open(out_lbl_path, "w") as lf:
                for (x1,y1,x2,y2) in cand:
                    cx,cy,w,h = xyxy_to_yolo(x1,y1,x2,y2,W,H)
                    lf.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

            rec = {"image": out_img_path, "slots": cand, "candidates": cand}
            jout.write(json.dumps(rec) + "\n")

    print(f"Processed {len(images)} backgrounds, wrote ~{total_boxes} slots total.")
    print(f"Overlays: {os.path.abspath(overlay_dir)}")
    print(f"Added to: {os.path.abspath(os.path.join(args.template_base, args.split))}")

if __name__ == "__main__":
    main()
