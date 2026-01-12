from __future__ import annotations

import io
import os
import re
import time
import sys
import uuid
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Response, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageEnhance, ImageOps
import subprocess
import shutil

from pdf_booklet import build_pbn_pdf_booklet


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
    "demo_a3": 1.0,
    "demo_a2": 0.5,
}

PORTRAIT_COLORS = {15, 20, 25}
LANDSCAPE_COLORS = {14, 21, 28}

A4_RATIO = 1 / (2 ** 0.5)  # width/height for portrait A-series

JOB_DIR_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def cleanup_old_workdirs(base: Path, max_age_hours: int = 24) -> None:
    """Delete per-job folders older than max_age_hours."""
    now = time.time()
    cutoff = now - max_age_hours * 3600

    for entry in base.iterdir():
        try:
            st = entry.stat()
        except FileNotFoundError:
            continue

        if not JOB_DIR_PATTERN.match(entry.name):
            continue

        if st.st_mtime > cutoff:
            continue

        if entry.is_dir():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            try:
                entry.unlink()
            except FileNotFoundError:
                pass


cleanup_old_workdirs(WORK_DIR)

app = FastAPI(title="Paint-by-Numbers MVP")

# Serve the single HTML page
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")


def _safe_filename(name: str) -> str:
    name = name.strip().replace("\\", "_").replace("/", "_")
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    return name[:120] if name else "upload"


def _read_upload_limited(upload: UploadFile, limit_bytes: int) -> bytes:
    """Read uploaded file into memory with a hard size limit."""
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
    """Center-crop image to target aspect ratio (w/h)."""
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


def _contain_on_a4_canvas(
    img: Image.Image,
    portrait: bool = True,
    bg=(255, 255, 255),
) -> Image.Image:
    """
    Fit image inside A4 ratio without cropping, pad with background.

    Output keeps original pixel scale roughly, just adds padding.
    """
    w, h = img.size
    target_ratio = A4_RATIO if portrait else 1 / A4_RATIO  # w/h

    current = w / h
    if abs(current - target_ratio) < 1e-6:
        return img

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
    """Gentle saturation boost to make colors a bit richer."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    enhancer = ImageEnhance.Color(img)
    return enhancer.enhance(1.10)  # ~10% boost


def _downscale_image_long_side_jpeg(
    img: Image.Image,
    long_side: int = 2048,
) -> Image.Image:
    """Downscale image so that its long side is at most long_side pixels."""
    w, h = img.size
    long_now = max(w, h)

    if long_now > long_side:
        scale = long_side / float(long_now)
        w2 = max(1, int(round(w * scale)))
        h2 = max(1, int(round(h * scale)))
        img = img.resize((w2, h2), resample=Image.LANCZOS)
    return img


def _downscale_long_side_jpeg(in_path: Path, out_path: Path, long_side: int = 2048) -> None:
    img = Image.open(str(in_path))
    img = _downscale_image_long_side_jpeg(img.convert("RGB"), long_side)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path), format="JPEG", quality=90, optimize=True)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html_path = APP_DIR / "static" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.post("/api/preview")
def generate_preview(
    image: UploadFile = File(...),
    detail: str = Form(...),  # easy/medium/hard/demo_a3/demo_a2
    colors: int = Form(...),
    auto_crop: str = Form("false"),
    auto_saturation: str = Form("false"),
    orientation: Optional[str] = Form(None),  # portrait/landscape; optional
    paper: str = Form("A4"),  # A4 / A3 / A2 / A1, default is A4
) -> JSONResponse:
    t0 = time.perf_counter()

    detail = (detail or "").strip().lower()
    if detail not in DETAIL_TO_MIN_FEATURE_MM:
        raise HTTPException(status_code=400, detail="Invalid detail level.")

    paper = (paper or "A4").upper()
    if paper not in ("A4", "A3", "A2", "A1"):
        paper = "A4"

    auto_crop_bool = str(auto_crop).lower() == "true"
    auto_sat_bool = str(auto_saturation).lower() == "true"

    raw = _read_upload_limited(image, MAX_UPLOAD_BYTES)

    try:
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img)
        # Temporary: keep the input at a manageable resolution for the MVP.
        img = _downscale_image_long_side_jpeg(img, 2048)
        img.load()
        del raw
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

    portrait = orientation == "portrait"

    # Optional pre-processing
    if auto_sat_bool:
        img = _auto_saturate(img)

    if auto_crop_bool:
        target_ratio = A4_RATIO if portrait else 1 / A4_RATIO
        img = _center_crop_to_ratio(img, target_ratio)

    # Per-job folder
    job_id = uuid.uuid4().hex
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Always save internal input as JPEG
    input_path = job_dir / "input.jpg"
    img.save(str(input_path), format="JPEG", quality=99)

    min_feature_mm = DETAIL_TO_MIN_FEATURE_MM[detail]

    # Run pipeline synchronously, but WITHOUT PDF (build_pdf=0)
    old_cwd = os.getcwd()
    try:
        os.chdir(str(job_dir))
        out_dir = job_dir / "output"

        cmd = [
            sys.executable,
            str(APP_DIR / "main.py"),
            str(input_path),
            paper,
            str(min_feature_mm),
            str(int(colors)),
            "0",  # do not build PDF inside the pipeline
        ]

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["NUMEXPR_NUM_THREADS"] = "1"
        env["PYTHONUNBUFFERED"] = "1"

        log_path = job_dir / "pipeline.log"
        with log_path.open("w", encoding="utf-8") as log_f:
            p = subprocess.run(
                cmd,
                cwd=str(job_dir),
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
            )

        if p.returncode != 0:
            tail = ""
            try:
                tail = log_path.read_text(encoding="utf-8")[-4000:]
            except Exception:
                pass
            raise HTTPException(status_code=500, detail=f"Pipeline failed.\n{tail}")
    finally:
        os.chdir(old_cwd)

    # PDF URL is just a link; actual PDF will be built lazily on download
    pdf_url: Optional[str] = None
    if detail in ("easy", "medium", "hard"):
        pdf_url = f"/download/{job_id}/pdf"

    # Build downscaled colored preview (separate from PDF)
    colored_candidates = []
    for ext in ("jpg", "jpeg", "JPG", "JPEG"):
        colored_candidates.extend(out_dir.glob(f"*_pbn_colored.{ext}"))
    colored_candidates = sorted(colored_candidates)

    colored_preview_path = job_dir / "preview_colored_2048.jpg"
    colored_url: Optional[str] = None

    if colored_candidates:
        try:
            _downscale_long_side_jpeg(colored_candidates[0], colored_preview_path, long_side=2048)
            colored_url = f"/download/{job_id}/colored"
        except Exception:
            colored_url = None

    dt = time.perf_counter() - t0

    meta = {
        "detail": detail,
        "min_feature_mm": min_feature_mm,
        "colors": int(colors),
        "orientation": orientation,
        "auto_crop": auto_crop_bool,
        "auto_saturation": auto_sat_bool,
        "paper": paper,
        "upload_limit_mb": MAX_UPLOAD_MB,
    }

    # Persist meta per job for lazy PDF generation
    try:
        meta_path = job_dir / "job_meta.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f)
    except Exception:
        pass

    return JSONResponse(
        {
            "job_id": job_id,
            "pdf_url": pdf_url,  # None for demo_a3/demo_a2
            "colored_url": colored_url,
            "seconds": round(dt, 2),
            "meta": meta,
        }
    )


@app.get("/download/{job_id}/pdf")
def download_pdf(job_id: str) -> FileResponse:
    job_dir = WORK_DIR / job_id
    out_dir = job_dir / "output"

    if not out_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found.")

    # Main expected path for the booklet PDF
    pdf_path = out_dir / "input_booklet.pdf"

    # If PDF does not exist yet, generate it lazily from existing outputs
    if not pdf_path.exists():
        input_path = job_dir / "input.jpg"
        if not input_path.exists():
            raise HTTPException(status_code=404, detail="Input not found for this job.")

        base = input_path.stem  # "input"
        ext = input_path.suffix  # ".jpg"

        outline_path = out_dir / f"{base}_pbn_outline{ext}"
        palette_csv_path = out_dir / f"{base}_palette.csv"
        original_preview_path = out_dir / f"{base}_preview_original.jpg"
        outline_preview_path = out_dir / f"{base}_preview_outline.jpg"

        if not outline_path.exists() or not palette_csv_path.exists():
            raise HTTPException(status_code=404, detail="Required outputs not found.")

        # Read meta to restore paper size, fallback to A4
        paper_size = "A4"
        meta_path = job_dir / "job_meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                paper_size = str(meta.get("paper") or "A4").upper()
            except Exception:
                paper_size = "A4"

        try:
            build_pbn_pdf_booklet(
                root=str(input_path.with_suffix("")),
                outline_path=str(outline_path),
                palette_csv_path=str(palette_csv_path),
                pdf_name=str(pdf_path),
                paper_size=paper_size,
                num_regions=None,
                difficulty=None,
                original_preview_path=str(original_preview_path)
                if original_preview_path.exists()
                else None,
                outline_preview_path=str(outline_preview_path)
                if outline_preview_path.exists()
                else None,
            )
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"PDF generation failed: {e}"
            )

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename="paint_by_numbers.pdf",
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


ANALYTICS_LOG = APP_DIR / "analytics.log"


@app.post("/api/track_event")
async def track_event(request: Request) -> JSONResponse:
    """
    Simple analytics endpoint.
    Example log line:
      2026-01-10 20:15:23 paid_generate_click {'paper': 'A3', 'from_detail': 'demo_a3'}
    """
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    event_type = str(payload.get("event") or "unknown").strip()
    if not event_type:
        event_type = "unknown"

    event_data = {k: v for k, v in payload.items() if k != "event"}

    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {event_type} {event_data}\n"

    try:
        with ANALYTICS_LOG.open("a", encoding="utf-8") as f:
            f.write(line)
        stored = True
    except Exception:
        stored = False

    return JSONResponse({"ok": True, "stored": stored})


@app.exception_handler(Exception)
async def any_exc(request, exc):
    return JSONResponse(
        status_code=500,
        content={"ok": False, "error": str(exc), "type": exc.__class__.__name__},
    )


@app.head("/")
def root_head():
    return Response(status_code=200)
