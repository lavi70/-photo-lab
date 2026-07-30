from __future__ import annotations
import asyncio
import base64
import json
import logging
import math
import os
from pathlib import Path
from typing import Optional
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from gemini_client import CONNECTION_LIMITS, REQUEST_TIMEOUT, build_grid_tasks
from image_processor import process_grid_image

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-image").strip()
GEMINI_IMAGE_SIZE = os.environ.get("GEMINI_IMAGE_SIZE", "2K").strip()

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}

ALLOWED_QUANTITIES = {4, 8, 12}
DEFAULT_QUANTITY = 12

# Each Gemini call always yields a 2x2 grid of exactly 4 photos, so an even
# primary/secondary split isn't always possible. These match the exact
# split requested; the 8 and 4 totals require generating one extra whole
# grid per variant and discarding the surplus crops to land on the exact
# odd count (5/3 and 3/1 respectively) — more Gemini calls than the visible
# output count for those two batch sizes.
SECONDARY_SPLITS = {
    12: (8, 4),
    8: (5, 3),
    4: (3, 1),
}

ASPECT_RATIO_DIMENSIONS = {
    "1:1": (2000, 2000),
    "9:16": (1080, 1920),
}
DEFAULT_ASPECT_RATIO = "1:1"

HEBREW_NICHE_TRANSLATIONS = {
    "עיצוב הבית": "Home & Living",
    "תכשיטים ואקססוריז": "Jewelry & Accessories",
    "טיפוח וקוסמטיקה": "Beauty & Skincare",
    "מטבח ואירוח": "Kitchen & Dining",
    "משרד וסביבת עבודה": "Office & Desk",
    "חוץ וגינה": "Outdoor & Garden",
    "אופנה וביגוד": "Fashion & Apparel",
    "אלקטרוניקה וגאדג'טים": "Electronics & Gadgets",
}

def resolve_niche_for_prompt(niche: str) -> str:
    """Map a Hebrew preset niche to its English equivalent for the Gemini prompt.
    Free-typed custom text (Hebrew or otherwise) is passed through unchanged,
    since there is no reliable translation for arbitrary user input here."""
    return HEBREW_NICHE_TRANSLATIONS.get(niche, niche)

app = FastAPI(title="AI Product Photo Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def check_config() -> None:
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")

@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "gemini_key_configured": bool(GEMINI_API_KEY),
        "model": GEMINI_MODEL,
        "image_size": GEMINI_IMAGE_SIZE,
    }

def _validate_upload(content_type: Optional[str], data: bytes, label: str) -> None:
    if content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail=f"{label}: unsupported image type.")
    if not data:
        raise HTTPException(status_code=400, detail=f"{label}: uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail=f"{label}: image is too large.")

@app.post("/api/generate")
async def generate_images(
    image: UploadFile = File(...),
    image2: Optional[UploadFile] = File(None),
    niche: str = Form(""),
    quantity: int = Form(DEFAULT_QUANTITY),
    aspect_ratio: str = Form(DEFAULT_ASPECT_RATIO),
) -> StreamingResponse:
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Server is missing GEMINI_API_KEY. Set it in backend/.env.")

    image_bytes = await image.read()
    _validate_upload(image.content_type, image_bytes, "Primary product photo")
    mime_type = image.content_type

    image2_bytes: Optional[bytes] = None
    mime_type2: Optional[str] = None
    if image2 is not None and image2.filename:
        image2_bytes = await image2.read()
        _validate_upload(image2.content_type, image2_bytes, "Secondary product variant")
        mime_type2 = image2.content_type

    niche = (niche or "").strip()
    if not niche:
        raise HTTPException(status_code=400, detail="Please select or enter a niche.")

    if quantity not in ALLOWED_QUANTITIES:
        raise HTTPException(status_code=400, detail=f"quantity must be one of {sorted(ALLOWED_QUANTITIES)}.")

    if aspect_ratio not in ASPECT_RATIO_DIMENSIONS:
        raise HTTPException(status_code=400, detail=f"aspect_ratio must be one of {list(ASPECT_RATIO_DIMENSIONS)}.")

    niche_for_prompt = resolve_niche_for_prompt(niche)
    target_size = ASPECT_RATIO_DIMENSIONS[aspect_ratio]

    if image2_bytes is not None:
        primary_count, secondary_count = SECONDARY_SPLITS[quantity]
        variants = [
            ("primary", image_bytes, mime_type, primary_count),
            ("secondary", image2_bytes, mime_type2, secondary_count),
        ]
    else:
        variants = [("primary", image_bytes, mime_type, quantity)]

    async def event_stream():
        warnings: list[str] = []
        images_sent = 0
        completed = 0
        remaining: dict[str, int] = {}

        async with httpx.AsyncClient(
            timeout=REQUEST_TIMEOUT, trust_env=False, verify=False, limits=CONNECTION_LIMITS
        ) as client:
            all_labeled_coros = []
            for variant_name, v_bytes, v_mime, v_count in variants:
                if v_count <= 0:
                    continue
                remaining[variant_name] = v_count
                grids_needed = math.ceil(v_count / 4)
                all_labeled_coros.extend(
                    build_grid_tasks(
                        image_bytes=v_bytes,
                        mime_type=v_mime,
                        niche=niche_for_prompt,
                        api_key=GEMINI_API_KEY,
                        client=client,
                        grids_count=grids_needed,
                        model=GEMINI_MODEL,
                        image_size=GEMINI_IMAGE_SIZE,
                        label_prefix=variant_name,
                    )
                )

            total_grids = len(all_labeled_coros)
            tasks = [asyncio.create_task(c) for c in all_labeled_coros]

            for finished in asyncio.as_completed(tasks):
                label, result, err = await finished
                completed += 1
                variant_name = label.split("-grid-")[0]

                if err is not None or result is None:
                    msg = f"{label} failed: {err}"
                    logger.warning(msg)
                    warnings.append(msg)
                    yield json.dumps(
                        {"type": "warning", "label": label, "message": str(err),
                         "completed": completed, "total": total_grids}
                    ) + "\n"
                    continue

                try:
                    quadrants = await asyncio.to_thread(process_grid_image, result.image_bytes, target_size)
                except ValueError as exc:
                    msg = f"{label} image processing failed: {exc}"
                    logger.warning(msg)
                    warnings.append(msg)
                    yield json.dumps(
                        {"type": "warning", "label": label, "message": str(exc),
                         "completed": completed, "total": total_grids}
                    ) + "\n"
                    continue

                # Trim to the exact count still needed for this variant, since
                # a grid always yields 4 crops even when fewer were requested.
                take = min(len(quadrants), remaining.get(variant_name, 0))
                quadrants = quadrants[:take]
                remaining[variant_name] = remaining.get(variant_name, 0) - take
                if not quadrants:
                    continue

                encoded = [
                    "data:image/jpeg;base64," + base64.b64encode(q).decode("ascii")
                    for q in quadrants
                ]
                images_sent += len(encoded)
                yield json.dumps(
                    {"type": "grid", "label": label, "variant": variant_name, "images": encoded,
                     "completed": completed, "total": total_grids}
                ) + "\n"

        if images_sent == 0:
            yield json.dumps(
                {"type": "error", "detail": "All Gemini generations failed. Details: " + " | ".join(warnings)}
            ) + "\n"
        else:
            yield json.dumps(
                {"type": "done", "niche": niche, "count": images_sent, "warnings": warnings}
            ) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")

if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
