"""
FastAPI backend for the MSFS-Net aerial image dehazing demo.

Run (from the backend/ folder):
    python app.py
or
    uvicorn app:app --reload --port 8000

Then open http://localhost:8000
"""
import base64
import io
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from engine import MAX_SIDE, DehazeEngine
from image_utils import (compute_metrics, error_map, error_map_plot_png,
                         error_map_raw, estimate_haze)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
MAX_UPLOAD_MB = 25
HAZE_TYPES = {"auto", "thin", "moderate", "thick"}

engine: DehazeEngine | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global engine
    engine = DehazeEngine()  # load weights once at startup
    yield


app = FastAPI(title="MSFS-Net Dehazing API", version="1.0.0", lifespan=lifespan)

# Allows the page to also be opened via VS Code "Live Server" (e.g. :5500)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def to_data_url(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


async def read_image(upload: UploadFile, what: str) -> Image.Image:
    data = await upload.read()
    if not data:
        raise HTTPException(400, f"The {what} file is empty.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"The {what} is larger than {MAX_UPLOAD_MB} MB.")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, f"The {what} is not a valid image (use PNG, JPG, BMP or TIFF).")


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health():
    return {
        "status": "ok" if engine else "loading",
        "device": str(engine.device) if engine else None,
        "parameters_million": round(engine.n_params / 1e6, 2) if engine else None,
        "max_side": MAX_SIDE,
    }


@app.post("/api/dehaze")
async def dehaze(
    image: UploadFile = File(..., description="Hazy aerial image"),
    haze_type: str = Form("auto"),
    reference: UploadFile | None = File(None, description="Optional ground-truth clear image"),
):
    if engine is None:
        raise HTTPException(503, "Model is still loading, try again in a moment.")

    haze_type = haze_type.lower()
    if haze_type not in HAZE_TYPES:
        raise HTTPException(400, f"haze_type must be one of {sorted(HAZE_TYPES)}")

    hazy_img = await read_image(image, "hazy image")
    ref_img = await read_image(reference, "reference image") if reference and reference.filename else None

    # ---- inference ------------------------------------------------------ #
    t0 = time.perf_counter()
    try:
        hazy, dehazed = await run_in_threadpool(engine.dehaze, hazy_img)
    except Exception as e:  # e.g. out-of-memory
        raise HTTPException(500, f"Inference failed: {e}")
    elapsed = time.perf_counter() - t0

    # ---- metrics + error map (vs ground truth if given, else vs hazy input) -- #
    h, w = dehazed.shape[:2]
    try:
        if ref_img is not None:
            if ref_img.size != (w, h):
                ref_img = ref_img.resize((w, h), Image.LANCZOS)
            ref = np.asarray(ref_img, dtype=np.uint8)
            mode = "reference"
        else:
            ref, mode = hazy, "input"
        metrics = compute_metrics(dehazed, ref)
        metrics["mode"] = mode
    except ValueError as e:
        raise HTTPException(400, str(e))

    err = error_map(dehazed, ref)
    metrics["mae"] = round(float(err.mean()), 4)
    plot_png = error_map_plot_png(err)

    return {
        "hazy": to_data_url(hazy),
        "dehazed": to_data_url(dehazed),
        "error_map": to_data_url(error_map_raw(err)),
        "error_map_plot": "data:image/png;base64," + base64.b64encode(plot_png).decode(),
        "error_vmax": 0.3,
        "metrics": metrics,
        "haze_type": haze_type,
        "haze_estimate": estimate_haze(hazy),
        "width": w,
        "height": h,
        "original_size": list(hazy_img.size),
        "inference_seconds": round(elapsed, 3),
        "device": str(engine.device),
    }


# Frontend (must be mounted last so it doesn't shadow /api)
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
