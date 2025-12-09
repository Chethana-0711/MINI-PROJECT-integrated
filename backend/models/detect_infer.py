# backend/models/detect_infer.py
import os
import numpy as np

MODEL_PATH = os.environ.get("YOLO_WEIGHTS_PATH", None)

USE_ULTRALYTICS = False
model = None

try:
    if MODEL_PATH and os.path.exists(MODEL_PATH):
        from ultralytics import YOLO
        print(f"[INFO] Loading YOLO model from: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
        USE_ULTRALYTICS = True
    else:
        print("[WARN] MODEL_PATH not found or YOLO_WEIGHTS_PATH not set")
except Exception as e:
    print("[ERROR] Failed to load YOLO:", e)
    USE_ULTRALYTICS = False


def detect_infer(img_bgr):
    """
    Return list of boxes:
    [x1, y1, x2, y2, score, class_id]
    """
    if not USE_ULTRALYTICS:
        print("[WARN] YOLO not available → returning empty detections")
        return []

    try:
        results = model.predict(img_bgr, imgsz=640, conf=0.25, verbose=False)
        boxes = []

        for r in results:
            if r.boxes is None:
                continue

            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                score = float(b.conf[0])
                class_id = int(b.cls[0])
                boxes.append([x1, y1, x2, y2, score, class_id])

        return boxes

    except Exception as e:
        print("[ERROR] YOLO inference failed:", e)
        return []
