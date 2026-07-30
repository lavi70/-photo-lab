from __future__ import annotations
from io import BytesIO
from typing import List, Tuple
from PIL import Image, ImageOps

OUTPUT_FORMAT = "JPEG"
OUTPUT_QUALITY = 100

def split_grid_into_quadrants(grid_image_bytes: bytes) -> List[Image.Image]:
    with Image.open(BytesIO(grid_image_bytes)) as src:
        img = ImageOps.exif_transpose(src.convert("RGB"))

    width, height = img.size
    half_w, half_h = width // 2, height // 2

    boxes = [
        (0, 0, half_w, half_h),
        (half_w, 0, width, half_h),
        (0, half_h, half_w, height),
        (half_w, half_h, width, height),
    ]

    return [img.crop(box) for box in boxes]

def resize_and_center_crop(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    width, height = img.size
    if width == target_w and height == target_h:
        return img

    scale = max(target_w / width, target_h / height)
    new_w, new_h = round(width * scale), round(height * scale)
    resized = img.resize((new_w, new_h), Image.LANCZOS)

    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))

def encode_jpeg(img: Image.Image) -> bytes:
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format=OUTPUT_FORMAT, quality=OUTPUT_QUALITY, optimize=True)
    return buf.getvalue()

def process_grid_image(grid_image_bytes: bytes, target_size: Tuple[int, int] = (2000, 2000)) -> List[bytes]:
    target_w, target_h = target_size
    quadrants = split_grid_into_quadrants(grid_image_bytes)
    outputs = []
    for quadrant in quadrants:
        if quadrant.width < 10 or quadrant.height < 10:
            raise ValueError("Source grid image is too small to split.")
        processed = resize_and_center_crop(quadrant, target_w, target_h)
        outputs.append(encode_jpeg(processed))

    return outputs
