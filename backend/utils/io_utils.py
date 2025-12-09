# backend/utils/io_utils.py
from PIL import Image
import io

def read_imagefile(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")

def pil_to_bytes(pil_img, fmt="PNG"):
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return buf.getvalue()
