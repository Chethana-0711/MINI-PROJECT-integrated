# backend/models/seg_infer.py
import os
import numpy as np
import cv2
from PIL import Image

MODEL_PATH = os.environ.get("SINET_CHECKPOINT_PATH", None)

# If you implement real SINet, load model weights here (PyTorch).
USE_REAL = False
try:
    if MODEL_PATH and os.path.exists(MODEL_PATH):
        USE_REAL = True
except Exception:
    USE_REAL = False

def seg_infer(img_bgr: np.ndarray):
    """
    Args:
      img_bgr: HxWx3 (BGR) numpy array
    Returns:
      mask: HxW bool numpy array
      prob_map: HxW float in [0..1]
    """
    h,w = img_bgr.shape[:2]

    if USE_REAL:
        # TODO: put your SINet model loading and inference here.
        # Example (pseudocode):
        # model = load_sinet(MODEL_PATH)
        # pred = model.predict(preprocess(img_bgr))
        # prob_map = cv2.resize(pred, (w,h))
        # mask = prob_map > 0.5
        # return mask.astype(bool), prob_map
        raise NotImplementedError("Real SINet inference not implemented in stub.")
    else:
        # Simple heuristic fallback: find low-contrast regions via local variance
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        # normalized local std filter
        blur = cv2.GaussianBlur(gray, (9,9), 0)
        diff = cv2.absdiff(gray, blur)
        # threshold; low diff -> likely camouflaged (smooth regions)
        thresh = np.percentile(diff, 40)
        mask = (diff <= thresh).astype(np.uint8)
        # morphological cleanup
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        prob_map = cv2.normalize(1 - (diff.astype(np.float32)/255.0), None, 0.0, 1.0, cv2.NORM_MINMAX)
        return mask.astype(bool), prob_map
