from __future__ import annotations
import asyncio
import base64
import logging
from dataclasses import dataclass
from typing import Any, Optional
import httpx

logger = logging.getLogger("gemini_client")

INTERACTIONS_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/interactions"
DEFAULT_MODEL = "gemini-3.1-flash-image"
DEFAULT_IMAGE_SIZE = "2K"

REQUEST_TIMEOUT = httpx.Timeout(connect=10.0, read=45.0, write=20.0, pool=10.0)
CONNECTION_LIMITS = httpx.Limits(max_connections=10, max_keepalive_connections=10, keepalive_expiry=30.0)
MAX_ATTEMPTS = 2

class GeminiGenerationError(RuntimeError):
    pass

@dataclass
class GeminiImageResult:
    image_bytes: bytes
    mime_type: str

def _build_prompt(niche: str) -> str:
    niche_clean = (niche or "general product").strip() or "general product"
    return f"""Create a single square image composed of exactly four product photos arranged in a perfect 2x2 grid.

STRICT LAYOUT REQUIREMENTS (highest priority):
- The four images must be merged seamlessly into one continuous canvas.
- Absolutely no visible borders, lines, dividers, gaps, or spacing between the four sections.
- No white lines or grid structure should be visible at all.
- The transitions between the four quadrants must be completely invisible.
- The final result must look like one unified image, not a collage.
- The canvas must be fully filled edge-to-edge with zero margins and zero padding.

STRICT PRODUCT PRESERVATION (ZERO TOLERANCE FOR CHANGES):
You MUST use the exact product shown in the input image. Do NOT alter the product's shape, geometry, outline, color, interior contents, or material. The product itself must remain 100% pixel-identical in form to the uploaded reference image. ONLY change the background environment, angles, and lighting.
The reference image is the single source of truth for the product's structure, geometry, and silhouette in every one of the four quadrants — never deviate from it.

PRODUCT ACCURACY RULES:
- Preserve the exact original proportions, shape, and dimensions of the product as in the reference image.
- Do NOT stretch, squash, crop, or distort the product in any way.
- Maintain realistic and consistent product scale across all four images.
- The product must remain visually identical in form.
- DO NOT redesign the product.
- DO NOT change the silhouette.
- DO NOT generate a different object of the same category.

ALLOWED VARIATIONS:
- Each quadrant should present the product in a different setting.
- You may change background, lighting, and camera angle only. (Make the setting relevant to the niche: {niche_clean}).

STYLE REQUIREMENTS:
- No people.
- Clean, realistic interior environments (e.g. living room, kitchen, studio, bedroom).
- Professional product photography style.
- Natural lighting or soft studio lighting.
- No text, no logos, no watermarks.

FINAL CHECK (critical):
Ensure there are ZERO visible lines or separations between the four sections. The image must appear as one seamless composition. Ensure the product's shape, geometry, and silhouette are unchanged and identical to the reference image in all four quadrants."""

def _build_lifestyle_prompt(niche: str) -> str:
    niche_clean = (niche or "general product").strip() or "general product"
    return f"""Create a single square image composed of exactly four lifestyle product photos arranged in a perfect 2x2 grid.

STRICT LAYOUT REQUIREMENTS (highest priority):
- The four images must be merged seamlessly into one continuous canvas.
- Absolutely no visible borders, lines, dividers, gaps, or spacing between the four sections.
- No white lines or grid structure should be visible at all.
- The transitions between the four quadrants must be completely invisible.
- The final result must look like one unified image, not a collage.
- The canvas must be fully filled edge-to-edge with zero margins and zero padding.

STRICT PRODUCT PRESERVATION (ZERO TOLERANCE FOR CHANGES):
You MUST use the exact product shown in the input image. Do NOT alter the product's shape, geometry, outline, color, interior contents, or material. The product itself must remain 100% pixel-identical in form to the uploaded reference image.
The reference image is the single source of truth for the product's structure, geometry, and silhouette in every one of the four quadrants — never deviate from it.
- DO NOT redesign the product.
- DO NOT change the silhouette.
- DO NOT generate a different object of the same category.

LIFESTYLE / HUMAN PRESENTATION CONTENT (all four quadrants):
Macro and medium close-up shots of a model's hands gracefully holding and presenting the product. Show different angles of placement in hand. Crucially: Do not show the model's face or upper body; focus only from the chest/shoulders down to the hands and the product. Ensure the model's skin tone and style are natural and non-distracting.

ALLOWED VARIATIONS:
- Each quadrant should present a different hand angle, grip, or placement.
- You may change background, lighting, and camera angle only. (Make the setting relevant to the niche: {niche_clean}).

STYLE REQUIREMENTS:
- Clean, realistic, professional product photography style.
- Natural lighting or soft studio lighting.
- No text, no logos, no watermarks.
- No visible face or upper body of the model in any quadrant.

FINAL CHECK (critical):
Ensure there are ZERO visible lines or separations between the four sections. The image must appear as one seamless composition. Ensure the product's shape, geometry, and silhouette are unchanged and identical to the reference image in all four quadrants. Ensure no model face or upper body is visible in any quadrant."""

def _extract_image_block(payload: dict[str, Any]) -> Optional[dict[str, Any]]:
    output_image = payload.get("output_image")
    if isinstance(output_image, dict) and output_image.get("data"):
        return output_image

    last_image_block = None
    for step in payload.get("steps", []):
        if step.get("type") != "model_output":
            continue
        for block in step.get("content", []):
            if block.get("type") == "image" and block.get("data"):
                last_image_block = block
    return last_image_block

async def generate_grid_image(
    image_bytes: bytes,
    mime_type: str,
    niche: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    image_size: str = DEFAULT_IMAGE_SIZE,
    client: Optional[httpx.AsyncClient] = None,
    prompt_builder=_build_prompt,
) -> GeminiImageResult:
    prompt = prompt_builder(niche)
    b64_image = base64.b64encode(image_bytes).decode("ascii")

    body = {
        "model": model,
        "input": [
            {"type": "text", "text": prompt},
            {"type": "image", "mime_type": mime_type, "data": b64_image},
        ],
        "response_format": {
            "type": "image",
            "aspect_ratio": "1:1",
            "image_size": image_size,
        },
    }
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }

    owns_client = client is None
    http_client = client or httpx.AsyncClient(
        timeout=REQUEST_TIMEOUT, trust_env=False, verify=False, limits=CONNECTION_LIMITS
    )

    last_error: Optional[Exception] = None
    try:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                resp = await http_client.post(INTERACTIONS_ENDPOINT, headers=headers, json=body)
            except httpx.TransportError as exc:
                last_error = exc
                continue

            if resp.status_code == 429 or resp.status_code >= 500:
                last_error = GeminiGenerationError(f"Gemini returned HTTP {resp.status_code}")
                continue

            if resp.status_code >= 400:
                raise GeminiGenerationError(f"Gemini returned HTTP {resp.status_code}")

            payload = resp.json()
            image_block = _extract_image_block(payload)
            if image_block is None:
                raise GeminiGenerationError("Gemini responded successfully but did not include an image.")

            return GeminiImageResult(
                image_bytes=base64.b64decode(image_block["data"]),
                mime_type=image_block.get("mime_type", "image/png"),
            )

        assert last_error is not None
        raise GeminiGenerationError(f"Gemini call failed after {MAX_ATTEMPTS} attempts: {last_error}")
    finally:
        if owns_client:
            await http_client.aclose()

async def _labeled(label: str, coro) -> tuple[str, Optional[GeminiImageResult], Optional[BaseException]]:
    try:
        result = await coro
        return label, result, None
    except BaseException as exc:  # noqa: BLE001 - surfaced to caller, not swallowed
        return label, None, exc

def build_grid_tasks(
    image_bytes: bytes,
    mime_type: str,
    niche: str,
    api_key: str,
    client: httpx.AsyncClient,
    count: int = 3,
    model: str = DEFAULT_MODEL,
    image_size: str = DEFAULT_IMAGE_SIZE,
    include_lifestyle_grid: bool = True,
) -> list[Any]:
    """Build one labeled coroutine per grid so callers can consume results
    as each one finishes (via asyncio.as_completed) instead of waiting for
    every grid to complete before anything is usable."""
    labeled_coros = [
        _labeled(
            f"grid-{i + 1}",
            generate_grid_image(
                image_bytes=image_bytes,
                mime_type=mime_type,
                niche=niche,
                api_key=api_key,
                model=model,
                image_size=image_size,
                client=client,
                prompt_builder=_build_prompt,
            ),
        )
        for i in range(count)
    ]
    if include_lifestyle_grid:
        labeled_coros.append(
            _labeled(
                "lifestyle",
                generate_grid_image(
                    image_bytes=image_bytes,
                    mime_type=mime_type,
                    niche=niche,
                    api_key=api_key,
                    model=model,
                    image_size=image_size,
                    client=client,
                    prompt_builder=_build_lifestyle_prompt,
                ),
            )
        )
    return labeled_coros
