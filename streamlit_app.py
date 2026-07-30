from __future__ import annotations
import asyncio
import io
import os
import sys
import zipfile
from pathlib import Path

import httpx
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))
from gemini_client import (  # noqa: E402
    CONNECTION_LIMITS,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MODEL,
    REQUEST_TIMEOUT,
    build_grid_tasks,
)
from image_processor import process_grid_image  # noqa: E402

STANDARD_GRIDS_PER_REQUEST = 3
INCLUDE_LIFESTYLE_GRID = True
TOTAL_GRIDS_PER_REQUEST = STANDARD_GRIDS_PER_REQUEST + (1 if INCLUDE_LIFESTYLE_GRID else 0)
TARGET_SIZE = 2000
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}

# Mirrors backend/main.py's HEBREW_NICHE_TRANSLATIONS. Kept as a separate copy
# here (rather than importing main.py) so this Streamlit deploy doesn't need
# fastapi/uvicorn as a dependency.
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
CUSTOM_OPTION = "מותאם אישית / הקלדה חופשית"


def resolve_niche_for_prompt(niche: str) -> str:
    return HEBREW_NICHE_TRANSLATIONS.get(niche, niche)


def get_api_key() -> str:
    if "GEMINI_API_KEY" in st.secrets:
        return str(st.secrets["GEMINI_API_KEY"]).strip()
    return os.environ.get("GEMINI_API_KEY", "").strip()


def get_config_value(name: str, default: str) -> str:
    if name in st.secrets:
        return str(st.secrets[name]).strip()
    return os.environ.get(name, default).strip()


async def run_generation(image_bytes: bytes, mime_type: str, niche_for_prompt: str,
                          api_key: str, model: str, image_size: str, on_progress):
    warnings: list[str] = []
    all_quadrants: list[bytes] = []

    async with httpx.AsyncClient(
        timeout=REQUEST_TIMEOUT, trust_env=False, verify=False, limits=CONNECTION_LIMITS
    ) as client:
        labeled_coros = build_grid_tasks(
            image_bytes=image_bytes,
            mime_type=mime_type,
            niche=niche_for_prompt,
            api_key=api_key,
            client=client,
            count=STANDARD_GRIDS_PER_REQUEST,
            model=model,
            image_size=image_size,
            include_lifestyle_grid=INCLUDE_LIFESTYLE_GRID,
        )
        tasks = [asyncio.create_task(c) for c in labeled_coros]
        completed = 0

        for finished in asyncio.as_completed(tasks):
            label, result, err = await finished
            completed += 1

            if err is not None or result is None:
                warnings.append(f"{label} failed: {err}")
                on_progress(completed, TOTAL_GRIDS_PER_REQUEST, None)
                continue

            try:
                quadrants = await asyncio.to_thread(process_grid_image, result.image_bytes, TARGET_SIZE)
            except ValueError as exc:
                warnings.append(f"{label} image processing failed: {exc}")
                on_progress(completed, TOTAL_GRIDS_PER_REQUEST, None)
                continue

            all_quadrants.extend(quadrants)
            on_progress(completed, TOTAL_GRIDS_PER_REQUEST, quadrants)

    return all_quadrants, warnings


def build_zip(images: list[bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, img in enumerate(images, start=1):
            zf.writestr(f"product-photo-{i:02d}.jpg", img)
    return buf.getvalue()


st.set_page_config(page_title="Product Photo Lab", page_icon="📸", layout="wide")

st.markdown(
    """
    <style>
      .niche-label { direction: rtl; text-align: right; font-weight: 600; margin-bottom: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Product Photo Lab")
st.caption("Upload one product photo, choose a niche, get 16 shots back (12 environment + 4 lifestyle hand shots).")

if "result_images" not in st.session_state:
    st.session_state.result_images = []
    st.session_state.result_warnings = []

uploaded_file = st.file_uploader("Product photo", type=["jpg", "jpeg", "png", "webp"])

st.markdown('<div class="niche-label">בחר נישה / קטגוריה</div>', unsafe_allow_html=True)
niche_options = list(HEBREW_NICHE_TRANSLATIONS.keys()) + [CUSTOM_OPTION]
selected = st.selectbox("niche_select", niche_options, label_visibility="collapsed")

custom_niche = ""
if selected == CUSTOM_OPTION:
    custom_niche = st.text_input(
        "custom_niche_input",
        placeholder="תארו את הנישה או הסביבה הרצויה",
        label_visibility="collapsed",
        key="custom_niche_input",
    )

niche_display = custom_niche.strip() if selected == CUSTOM_OPTION else selected

submit = st.button("Generate photos", type="primary")

if submit:
    api_key = get_api_key()
    if not api_key:
        st.error("Server is missing GEMINI_API_KEY. Add it under Settings → Secrets on Streamlit Cloud.")
    elif uploaded_file is None:
        st.error("Please choose a product photo.")
    elif not niche_display:
        st.error("Please choose a niche, or select Custom and describe one.")
    else:
        image_bytes = uploaded_file.getvalue()
        mime_type = uploaded_file.type
        if mime_type not in ALLOWED_MIME_TYPES:
            st.error("Unsupported image type.")
        elif len(image_bytes) > MAX_UPLOAD_BYTES:
            st.error("Image is too large.")
        else:
            niche_for_prompt = resolve_niche_for_prompt(niche_display)
            model = get_config_value("GEMINI_MODEL", DEFAULT_MODEL)
            image_size = get_config_value("GEMINI_IMAGE_SIZE", DEFAULT_IMAGE_SIZE)

            progress_bar = st.progress(0.0)
            status = st.empty()

            def on_progress(completed, total, _quadrants):
                progress_bar.progress(completed / total)
                status.text(f"Received {completed}/{total} batches…")

            with st.spinner("Talking to the AI model… this can take up to 30-40 seconds."):
                images, warnings = asyncio.run(
                    run_generation(image_bytes, mime_type, niche_for_prompt, api_key, model, image_size, on_progress)
                )

            status.empty()
            progress_bar.empty()

            if not images:
                st.error("All Gemini generations failed. Details: " + " | ".join(warnings))
            else:
                st.session_state.result_images = images
                st.session_state.result_warnings = warnings
                st.session_state.result_niche = niche_display
                st.success(f"Done — {len(images)} images generated.")

if st.session_state.result_warnings:
    st.warning("\n\n".join(st.session_state.result_warnings))

if st.session_state.result_images:
    images = st.session_state.result_images
    st.subheader(f"{len(images)} images — \"{st.session_state.get('result_niche', '')}\"")

    st.download_button(
        "Download All as ZIP",
        data=build_zip(images),
        file_name="product-photos.zip",
        mime="application/zip",
        type="primary",
    )
    st.caption(
        "Streamlit can't trigger multiple separate browser downloads from one click the way the original "
        "custom HTML/JS UI did — use the ZIP above for a one-click batch, or the per-photo buttons below "
        "for individual downloads."
    )

    cols = st.columns(4)
    for i, img in enumerate(images):
        with cols[i % 4]:
            st.image(img, use_container_width=True)
            st.download_button(
                "⬇ Download",
                data=img,
                file_name=f"product-photo-{i + 1:02d}.jpg",
                mime="image/jpeg",
                key=f"dl-{i}",
            )
