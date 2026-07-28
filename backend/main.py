from __future__ import annotations
import asyncio
import base64
import json
import logging
import os
from pathlib import Path
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
STANDARD_GRIDS_PER_REQUEST = 3
INCLUDE_LIFESTYLE_GRID = True
TOTAL_GRIDS_PER_REQUEST = STANDARD_GRIDS_PER_REQUEST + (1 if INCLUDE_LIFESTYLE_GRID else 0)
TARGET_SIZE = 2000

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}

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

@app.post("/api/generate")
async def generate_images(
    image: UploadFile = File(...),
    niche: str = Form(""),
) -> StreamingResponse:
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Server is missing GEMINI_API_KEY. Set it in backend/.env.")

    if image.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported image type.")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="Image is too large.")

    niche = (niche or "").strip()
    if not niche:
        raise HTTPException(status_code=400, detail="Please select or enter a niche.")

    mime_type = image.content_type
    niche_for_prompt = resolve_niche_for_prompt(niche)

    async def event_stream():
        warnings: list[str] = []
        images_sent = 0
        completed = 0

        async with httpx.AsyncClient(
            timeout=REQUEST_TIMEOUT, trust_env=False, verify=False, limits=CONNECTION_LIMITS
        ) as client:
            labeled_coros = build_grid_tasks(
                image_bytes=image_bytes,
                mime_type=mime_type,
                niche=niche_for_prompt,
                api_key=GEMINI_API_KEY,
                client=client,
                count=STANDARD_GRIDS_PER_REQUEST,
                model=GEMINI_MODEL,
                image_size=GEMINI_IMAGE_SIZE,
                include_lifestyle_grid=INCLUDE_LIFESTYLE_GRID,
            )
            tasks = [asyncio.create_task(c) for c in labeled_coros]

            for finished in asyncio.as_completed(tasks):
                label, result, err = await finished
                completed += 1

                if err is not None or result is None:
                    msg = f"{label} failed: {err}"
                    logger.warning(msg)
                    warnings.append(msg)
                    yield json.dumps(
                        {"type": "warning", "label": label, "message": str(err),
                         "completed": completed, "total": TOTAL_GRIDS_PER_REQUEST}
                    ) + "\n"
                    continue

                try:
                    quadrants = await asyncio.to_thread(process_grid_image, result.image_bytes, TARGET_SIZE)
                except ValueError as exc:
                    msg = f"{label} image processing failed: {exc}"
                    logger.warning(msg)
                    warnings.append(msg)
                    yield json.dumps(
                        {"type": "warning", "label": label, "message": str(exc),
                         "completed": completed, "total": TOTAL_GRIDS_PER_REQUEST}
                    ) + "\n"
                    continue

                encoded = [
                    "data:image/jpeg;base64," + base64.b64encode(q).decode("ascii")
                    for q in quadrants
                ]
                images_sent += len(encoded)
                yield json.dumps(
                    {"type": "grid", "label": label, "images": encoded,
                     "completed": completed, "total": TOTAL_GRIDS_PER_REQUEST}
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
