from __future__ import annotations

import io
import os
import re
import time
import uuid
from pathlib import Path
from typing import Optional, Tuple

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Response
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageEnhance, ImageOps


import pipeline  # uses your existing pipeline.main()

APP_DIR = Path(__file__).resolve().parent
WORK_DIR = APP_DIR / "work"
WORK_DIR.mkdir(exist_ok=True)

# ---- MVP constraints ----
MAX_UPLOAD_MB = 15
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

DETAIL_TO_MIN_FEATURE_MM = {
    "easy": 4.0,
    "medium": 2.0,
    "hard": 1.5,
}

PORTRAIT_COLORS = {15, 20, 25}
LANDSCAPE_COLORS = {14, 21, 28}

A4_RATIO = 1 / (2 ** 0.5)  # width/height for portrait A-series


app = FastAPI(title="Paint-by-Numbers MVP")

# Serve the single HTML page
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")


def _safe_filename(name: str) -> str:
    name = name.strip().replace("\\", "_").replace("/", "_")
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    return name[:120] if name else "upload"


def _read_upload_limited(upload: UploadFile, limit_bytes: int) -> bytes:
    # Read into memory with a hard limit
    buf = bytearray()
    chunk_size = 1024 * 1024
    while True:
        chunk = upload.file.read(chunk_size)
        if not chunk:
            break
        buf.extend(chunk)
        if len(buf) > limit_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Limit is {MAX_UPLOAD_MB} MB.",
            )
    return bytes(buf)


def _infer_orientation(img: Image.Image) -> str:
    return "landscape" if img.width > img.height else "portrait"


def _center_crop_to_ratio(img: Image.Image, target_ratio_w_over_h: float) -> Image.Image:
    """
    Center-crop image to target aspect ratio (w/h).
    """
    w, h = img.size
    current = w / h

    if abs(current - target_ratio_w_over_h) < 1e-6:
        return img

    if current > target_ratio_w_over_h:
        # too wide -> crop width
        new_w = int(round(h * target_ratio_w_over_h))
        x0 = (w - new_w) // 2
        return img.crop((x0, 0, x0 + new_w, h))
    else:
        # too tall -> crop height
        new_h = int(round(w / target_ratio_w_over_h))
        y0 = (h - new_h) // 2
        return img.crop((0, y0, w, y0 + new_h))


def _contain_on_a4_canvas(img: Image.Image, portrait: bool = True, bg=(255, 255, 255)) -> Image.Image:
    """
    Fit image inside A4 ratio without cropping, pad with background.
    Output keeps original pixel count scale roughly, just pads.
    """
    w, h = img.size
    target_ratio = A4_RATIO if portrait else 1 / A4_RATIO  # w/h

    current = w / h
    if abs(current - target_ratio) < 1e-6:
        return img

    # Determine new canvas size preserving max dimension
    if current > target_ratio:
        # too wide: canvas must be taller
        new_h = int(round(w / target_ratio))
        new_w = w
    else:
        # too tall: canvas must be wider
        new_w = int(round(h * target_ratio))
        new_h = h

    canvas = Image.new("RGB", (new_w, new_h), bg)
    x0 = (new_w - w) // 2
    y0 = (new_h - h) // 2
    canvas.paste(img, (x0, y0))
    return canvas


def _auto_saturate(img: Image.Image) -> Image.Image:
    """
    Gentle saturation boost. Keep it mild to avoid 'Instagram look'.
    """
    # Convert to RGB just in case
    if img.mode != "RGB":
        img = img.convert("RGB")
    enhancer = ImageEnhance.Color(img)
    return enhancer.enhance(1.10)  # 10% boost


def _downscale_long_side_jpeg(in_path: Path, out_path: Path, long_side: int = 2048) -> None:
    img = Image.open(str(in_path)).convert("RGB")
    w, h = img.size
    long_now = max(w, h)

    if long_now > long_side:
        scale = long_side / float(long_now)
        w2 = max(1, int(round(w * scale)))
        h2 = max(1, int(round(h * scale)))
        img = img.resize((w2, h2), resample=Image.LANCZOS)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path), format="JPEG", quality=90, optimize=True)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html_path = APP_DIR / "static" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.post("/api/preview")
def generate_preview(
    image: UploadFile = File(...),
    detail: str = Form(...),  # easy/medium/hard
    colors: int = Form(...),
    auto_crop: str = Form("false"),
    auto_saturation: str = Form("false"),
    orientation: Optional[str] = Form(None),  # portrait/landscape; optional
) -> JSONResponse:
    t0 = time.perf_counter()

    detail = (detail or "").strip().lower()
    if detail not in DETAIL_TO_MIN_FEATURE_MM:
        raise HTTPException(status_code=400, detail="Invalid detail level.")

    auto_crop_bool = str(auto_crop).lower() == "true"
    auto_sat_bool = str(auto_saturation).lower() == "true"

    raw = _read_upload_limited(image, MAX_UPLOAD_BYTES)

    try:
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img)
        img.load()
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot read image. Please upload JPG/PNG.")

    # Normalize to RGB
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    elif img.mode == "L":
        img = img.convert("RGB")

    # Determine orientation
    if orientation is None:
        orientation = _infer_orientation(img)
    orientation = orientation.strip().lower()
    if orientation not in ("portrait", "landscape"):
        raise HTTPException(status_code=400, detail="Invalid orientation.")

    # Validate colors by orientation
    if orientation == "portrait":
        if colors not in PORTRAIT_COLORS:
            raise HTTPException(status_code=400, detail="For portrait use 15/20/25 colors.")
        portrait = True
    else:
        if colors not in LANDSCAPE_COLORS:
            raise HTTPException(status_code=400, detail="For landscape use 14/21/28 colors.")
        portrait = False

    # Pre-processing toggles
    if auto_sat_bool:
        img = _auto_saturate(img)

    if auto_crop_bool:
        target_ratio = A4_RATIO if portrait else 1 / A4_RATIO
        img = _center_crop_to_ratio(img, target_ratio)
    else:
        img = _contain_on_a4_canvas(img, portrait=portrait)

    # Write a temp input file for the existing pipeline
    job_id = uuid.uuid4().hex
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    in_name = _safe_filename(image.filename or "upload.jpg")
    # Keep extension sane
    ext = Path(in_name).suffix.lower()
    if ext not in (".jpg", ".jpeg", ".png"):
        ext = ".jpg"
    input_path = job_dir / f"input{ext}"

    # Save as JPEG/PNG
    if ext in (".jpg", ".jpeg"):
        img.save(str(input_path), format="JPEG", quality=95)
    else:
        img.save(str(input_path), format="PNG", compress_level=6)

    min_feature_mm = DETAIL_TO_MIN_FEATURE_MM[detail]

    # Run existing pipeline synchronously (MVP)
    # Note: pipeline writes to ./output by default. We want per-job output:
    # easiest MVP: temporarily chdir into job_dir and set output to "output".
    old_cwd = os.getcwd()
    try:
        os.chdir(str(job_dir))
        # pipeline will create ./output
        pipeline.main(
            str(input_path),
            paper_size="A4",
            num_colors=int(colors),
            min_feature_mm=float(min_feature_mm),
        )
    finally:
        os.chdir(old_cwd)

    # Find produced PDF (pipeline uses output/{stem}_booklet.pdf)
    out_dir = job_dir / "output"
    # In your pipeline, base = input_path.stem, so "input_booklet.pdf"
    pdf_path = out_dir / "input_booklet.pdf"
    if not pdf_path.exists():
        # Fallback: take any *_booklet.pdf
        pdfs = sorted(out_dir.glob("*_booklet.pdf"))
        if not pdfs:
            raise HTTPException(status_code=500, detail="PDF was not generated.")
        pdf_path = pdfs[0]
        
    # --- build downscaled colored preview (not embedded into PDF) ---
    colored_candidates = sorted(out_dir.glob("*_pbn_colored.jpg"))
    colored_preview_path = job_dir / "preview_colored_2048.jpg"
    colored_url = None
    
    if colored_candidates:
        try:
            _downscale_long_side_jpeg(colored_candidates[0], colored_preview_path, long_side=2048)
            colored_url = f"/download/{job_id}/colored"
        except Exception:
            colored_url = None

    dt = time.perf_counter() - t0

    return JSONResponse(
        {
            "job_id": job_id,
            "pdf_url": f"/download/{job_id}/pdf",
            "colored_url": colored_url,
            "seconds": round(dt, 2),
            "meta": {
                "detail": detail,
                "min_feature_mm": min_feature_mm,
                "colors": int(colors),
                "orientation": orientation,
                "auto_crop": auto_crop_bool,
                "auto_saturation": auto_sat_bool,
                "paper": "A4",
                "upload_limit_mb": MAX_UPLOAD_MB,
            },
        }
    )


@app.get("/download/{job_id}/pdf")
def download_pdf(job_id: str) -> FileResponse:
    job_dir = WORK_DIR / job_id
    out_dir = job_dir / "output"
    pdf_path = out_dir / "input_booklet.pdf"
    if not pdf_path.exists():
        pdfs = sorted(out_dir.glob("*_booklet.pdf"))
        if not pdfs:
            raise HTTPException(status_code=404, detail="Not found.")
        pdf_path = pdfs[0]

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename="paint_by_numbers_A4.pdf",
    )


@app.get("/download/{job_id}/colored")
def download_colored(job_id: str) -> FileResponse:
    p = WORK_DIR / job_id / "preview_colored_2048.jpg"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Not found.")
    return FileResponse(
        path=str(p),
        media_type="image/jpeg",
        filename="colored_preview.jpg",
    )


@app.exception_handler(Exception)
async def any_exc(request, exc):
    return JSONResponse(
        status_code=500,
        content={"ok": False, "error": str(exc), "type": exc.__class__.__name__},
    )


@app.get("/")
def root():
    return {"ok": True}


@app.head("/")
def root_head():
    return Response(status_code=200)