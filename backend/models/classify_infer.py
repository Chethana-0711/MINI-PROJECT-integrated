# backend/models/classify_infer.py
import os
import numpy as np
# You can use torchvision/timm to load a classifier when available
MODEL_PATH = os.environ.get("CLASSIFIER_WEIGHTS_PATH", None)
USE_REAL = False
try:
    if MODEL_PATH and os.path.exists(MODEL_PATH):
        USE_REAL = True
except Exception:
    USE_REAL = False

def classify_infer(crop_bgr):
    """
    Input: BGR crop np.array
    Returns: (species_name, prob)
    """
    if USE_REAL:
        # Implement real classifier inference.
        raise NotImplementedError("Classifier not implemented in stub.")
    else:
        # simple heuristic: return None/0.0 so the LLM fallback still works
        return None, 0.0