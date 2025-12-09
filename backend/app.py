# backend/app.py
from dotenv import load_dotenv
load_dotenv()

# backend/app.py
import os
import io
import base64
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import requests
from PIL import Image
import numpy as np

# model wrappers (stubs + real wrappers)
from models.seg_infer import seg_infer
from models.detect_infer import detect_infer
from models.classify_infer import classify_infer
from models.llm_infer import generate_description

from utils.io_utils import read_imagefile, pil_to_bytes
from utils.camo_utils import compute_camo_percentage, regions_from_mask, mask_to_heatmap, box_to_pct

app = FastAPI()

# allow your frontend origin(s)
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeResponse(BaseModel):
    detected: bool
    species: Optional[str]
    camouflagePercentage: int
    confidence: int
    description: str
    adaptations: list
    boundingBox: Optional[dict]
    camouflageRegions: list

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(image_url: Optional[str] = Form(None), file: Optional[UploadFile] = File(None)):
    """
    Accept either multipart file upload (file) or an image_url form field.
    Returns JSON with segmentation mask-derived statistics, bounding box, species guess, etc.
    """

    if (not image_url) and (not file):
        raise HTTPException(status_code=400, detail="Provide file or image_url")

    # Read image bytes
    if image_url:
        try:
            r = requests.get(image_url, timeout=10)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch image_url: {e}")
    else:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

    # Convert PIL -> numpy (BGR for OpenCV-friendly)
    img_np = np.array(img)[:, :, ::-1].copy()

    # 1) Detector (returns list of boxes: [x1,y1,x2,y2,score,cls])
    boxes = detect_infer(img_np)  # stub or real

    # 2) Segmentation (mask: HxW bool, prob_map: HxW float 0..1)
    mask, prob_map = seg_infer(img_np)

    # 3) Choose primary bounding box (if any) - prefer detection that overlaps mask
    primary_box = None
    if boxes:
        # choose box with largest IoU with mask bounding area
        best = None
        best_score = -1
        for b in boxes:
            x1,y1,x2,y2,score,cls_id = b
            bx1,by1,bx2,by2 = map(int,[x1,y1,x2,y2])
            # compute overlap area with mask
            mask_crop = mask[by1:by2, bx1:bx2]
            overlap = mask_crop.sum() if mask_crop.size>0 else 0
            if overlap > best_score:
                best_score = overlap
                best = b
        primary_box = best if best_score>0 else boxes[0]  # fallback to top box

    # If no boxes, compute bbox from mask
    if (not primary_box) and mask.sum()>0:
        ys, xs = np.where(mask)
        x1,y1,x2,y2 = xs.min(), ys.min(), xs.max(), ys.max()
        primary_box = [int(x1), int(y1), int(x2), int(y2), 0.95, 0]  # synthetic score

    # 4) Species classification (on crop if primary_box exists)
    species = None
    species_prob = 0.0
    if primary_box:
        x1,y1,x2,y2,_,_ = map(int, primary_box)
        crop = img_np[y1:y2, x1:x2] if y2>y1 and x2>x1 else img_np
        species, species_prob = classify_infer(crop)

    # 5) camo percentage (algorithmic)
    camo_pct = int(round(compute_camo_percentage(img_np, mask)))

    # 6) confidence (combine mask mean prob + species prob + detection confidence)
    mask_conf = float(prob_map[mask].mean()) if mask.sum()>0 else 0.0
    detect_conf = float(primary_box[4]) if primary_box else 0.0
    combined_conf = (0.5*mask_conf + 0.3*species_prob + 0.2*detect_conf)
    confidence = int(round(max(0,min(1,combined_conf))*100))

    # 7) regions
    regions = regions_from_mask(mask, img_np.shape)

    # 8) heatmap (optional - base64 image to store later)
    # heatmap_img = mask_to_heatmap(mask, img_np)  # returns PIL image if you want to upload to supabase

    # 9) LLM description & adaptations
    desc, adaptations = generate_description({
        "species": species,
        "species_prob": species_prob,
        "camo_pct": camo_pct,
        "regions": regions,
        "bbox": box_to_pct(primary_box, img_np.shape) if primary_box else None
    })

    response = {
        "detected": (camo_pct > 0) or (primary_box is not None),
        "species": species,
        "camouflagePercentage": camo_pct,
        "confidence": confidence,
        "description": desc,
        "adaptations": adaptations,
        "boundingBox": box_to_pct(primary_box, img_np.shape) if primary_box else None,
        "camouflageRegions": regions
    }
    return response

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="info")