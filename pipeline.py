from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import csv
import threading

from config import (
    DEFAULT_NUM_COLORS,
    DEFAULT_MIN_FEATURE_MM,
    DEFAULT_AREA_FACTOR,
    DEFAULT_MAX_EFFECTIVE_DPI,
    TARGET_NUMBER_HEIGHT_MM,
    DEFAULT_RANDOM_SEED,
    get_paper_long_side_mm,
    CONNECTIVITY4,
)
from quantization import quantize_kmeans_lab, smooth_cluster_map
from smoothing import estimate_min_region_pixels
from segmentation import (
    segment_regions_scipy,
    merge_small_regions_scipy,
    clean_small_final_regions,
    hard_cleanup_tiny_regions,
    segment_final_regions_scipy,
)
from palette_utils import (
    build_ordered_palette,
    save_palette_csv,
)
from rendering import render_outline_and_colored_highres, draw_numbers_on_outline_highres
from pdf_booklet import build_pbn_pdf_booklet

# --- Memory logging helper -------------------------------------------------

ENABLE_MEM_LOG = True

try:
    import psutil  # type: ignore

    _PROC = psutil.Process(os.getpid())
except Exception:
    psutil = None
    _PROC = None

_MAX_RSS_MB: float = 0.0


def get_memory():
    """
    Return current RSS of the process in MB if available, otherwise None.

    Caller is responsible for printing/logging the value.
    """
    global _MAX_RSS_MB

    if not ENABLE_MEM_LOG or _PROC is None:
        return None

    try:
        rss_bytes = _PROC.memory_info().rss
    except Exception:
        return None

    rss_mb = round(rss_bytes / (1024 * 1024))
    if rss_mb > _MAX_RSS_MB:
        _MAX_RSS_MB = rss_mb
    return rss_mb

    
_sampler_thread = None
_sampler_stop_flag = False


def start_memory_sampler(log_path: str = "mem_trace.csv", interval: float = 0.1) -> None:
    """
    Start a background thread that samples RSS every `interval` seconds.
    Values are written to a CSV file: timestamp_sec_since_start, rss_mb.
    """

    global _sampler_thread, _sampler_stop_flag
    if _PROC is None:
        print("[mem-sampler] psutil is not available, sampler disabled")
        return

    if _sampler_thread is not None:
        print("[mem-sampler] already running")
        return

    _sampler_stop_flag = False
    t0 = time.perf_counter()

    def _worker() -> None:
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["t_sec", "rss_mb"])
            while not _sampler_stop_flag:
                rss_mb = get_memory()
                if rss_mb is not None:
                    t = time.perf_counter() - t0
                    writer.writerow([f"{t:.3f}", rss_mb])
                time.sleep(interval)

    _sampler_thread = threading.Thread(target=_worker, daemon=True)
    _sampler_thread.start()
    print(f"[mem-sampler] started, interval={interval}s, log={log_path}")


def stop_memory_sampler() -> None:
    """
    Stop the background memory sampler if it is running.
    """

    global _sampler_thread, _sampler_stop_flag
    if _sampler_thread is None:
        return

    _sampler_stop_flag = True
    _sampler_thread.join()
    _sampler_thread = None
    print("[mem-sampler] stopped")
def build_output_paths(input_path: str, output_dir: str = "output") -> dict:
    """
    Generate output paths under output_dir.
    Takes only filename, not full directory path.
    """
    p = Path(input_path)
    base = p.stem      # "UAY_6763-002"
    ext = p.suffix     # ".jpg" or ".png"

    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    return {
        "quant": out / f"{base}_quantized{ext}",
        "outline": out / f"{base}_pbn_outline{ext}",
        "colored": out / f"{base}_pbn_colored{ext}",
        "palette_csv": out / f"{base}_palette.csv",
        "palette_img": out / f"{base}_palette.png",
        "pdf": str(out / f"{base}_booklet.pdf"),
    }


def estimate_font_size_px_for_print(
    image_long_px: int,
    print_long_mm: float,
    target_text_mm: float = TARGET_NUMBER_HEIGHT_MM,
    max_effective_dpi: int = DEFAULT_MAX_EFFECTIVE_DPI,
    min_px: int = 10,
    max_px: int = 48,
) -> int:
    """
    Compute an approximate font size in pixels so that the text height on paper
    is around target_text_mm, taking into account the target print size.

    The logic mirrors the smoothing / min-region calculations:
    we clamp the effective resolution by max_effective_dpi and derive mm/px.
    """
    max_effective_px = int(print_long_mm / 25.4 * max_effective_dpi)
    effective_long_px = min(image_long_px, max_effective_px)

    mm_per_px = print_long_mm / float(effective_long_px)

    font_size = target_text_mm / mm_per_px
    font_size_int = int(round(font_size))
    font_size_int = max(min_px, min(max_px, font_size_int))
    return font_size_int


def resize_for_print(
    img: Image.Image,
    print_long_mm: float,
    max_effective_dpi: int = DEFAULT_MAX_EFFECTIVE_DPI,
):
    """
    Downscale image if its long side is larger than what we actually need
    for the chosen paper size and max_effective_dpi.

    We do NOT upscale images that are already smaller.

    Returns
    -------
    resized_img : PIL.Image.Image
        Possibly resized image.
    scale       : float
        scale factor (new_size = old_size * scale).
        1.0 means "no change".
    """
    orig_w, orig_h = img.size
    long_px = max(orig_w, orig_h)

    max_long_px = int(round(print_long_mm / 25.4 * max_effective_dpi))

    if long_px <= max_long_px:
        return img, 1.0

    scale = max_long_px / float(long_px)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    resized = img.resize((new_w, new_h), Image.LANCZOS)
    print(
        f"[0] Resize for print: {orig_w}x{orig_h} -> {new_w}x{new_h}, "
        f"scale={scale:.3f}, target long≈{max_long_px}px"
    )
    return resized, scale


def load_and_resize_for_print(
    input_path: str,
    print_long_mm: float,
) -> tuple[np.ndarray, int]:
    """
    Load source image, apply EXIF orientation, downscale for print
    and return a NumPy array plus the effective long side in pixels.
    """
    orig_img = Image.open(input_path)
    orig_img = ImageOps.exif_transpose(orig_img).convert("RGB")

    # Slightly conservative DPI during the pipeline to save memory.
    resized_img, _ = resize_for_print(
        orig_img,
        print_long_mm=print_long_mm,
        max_effective_dpi=DEFAULT_MAX_EFFECTIVE_DPI / 2,
    )

    orig_arr = np.asarray(resized_img).astype(np.uint8)
    H, W, _ = orig_arr.shape
    image_long_px = max(H, W)

    del orig_img, resized_img

    print(
        f"[0] Input (effective): size: {W}x{H}, long={image_long_px}px, "
        f"mem={get_memory()}"
    )
    return orig_arr, image_long_px


def run_quantization_and_smoothing(
    orig_arr: np.ndarray,
    num_colors: int,
    image_long_px: int,
    print_long_mm: float,
    min_feature_mm: float,
    area_factor: float,
    quant_output_path: Path,
):
    """
    Perform KMeans quantization in Lab and smoothing of the cluster map.

    Returns
    -------
    cluster_id_img : np.ndarray (H, W)
    palette_final  : list of RGB colors
    """
    t = time.perf_counter()
    cluster_id_raw, palette_colors, quant_arr = quantize_kmeans_lab(
        orig_arr, num_colors
    )
    Image.fromarray(quant_arr, mode="RGB").save(quant_output_path)
    print(
        f"[1] KMeans quantization (Lab) -> {quant_output_path} "
        f"({time.perf_counter() - t:.2f}s), "
        f"mem={get_memory()}"
    )
    print(f"    Initial clusters: {len(palette_colors)}")

    t = time.perf_counter()
    cluster_id_img, palette_final = smooth_cluster_map(
        cluster_id_raw,
        palette_colors,
        image_long_px=image_long_px,
        print_long_mm=print_long_mm,
        min_feature_mm=min_feature_mm,
        area_factor=area_factor,
        max_effective_dpi=DEFAULT_MAX_EFFECTIVE_DPI,
        orig_arr=orig_arr,
    )

    # Drop heavy intermediates as soon as possible
    del cluster_id_raw, palette_colors, quant_arr, orig_arr

    print(
        f"[3] Cluster map smoothing ({time.perf_counter() - t:.2f}s), "
        f"mem={get_memory()}"
    )

    return cluster_id_img, palette_final


def classify_difficulty(num_regions: int) -> str:
    """
    Map number of regions to a human-readable difficulty level.
    """
    if num_regions < 500:
        return "easy"
    if num_regions < 1000:
        return "medium"
    if num_regions < 2000:
        return "hard"
    return "insane"


def run_regions_and_render(
    cluster_id_img,
    palette_final,
    print_long_mm: float,
    min_region_pixels: int,
    outline_path: Path,
    colored_path: Path,
    palette_csv_path: Path,
):
    """
    Full region-processing pipeline plus high-res rendering and palette CSV.

    Steps:
      1. Initial region segmentation.
      2. Merge small regions.
      3. Clean up tiny regions and re-segment for final regions.
      4. Palette ordering and paint numbering.
      5. Difficulty classification based on final region count.
      6. High-res outline + colored render with numbers.
      7. Save palette CSV.

    Returns
    -------
    paint_palette : list of RGB tuples
    num_regions   : int
    difficulty    : str
    """
    # 4. First segmentation
    t = time.perf_counter()
    region_id_img, region_color_id, region_area = segment_regions_scipy(cluster_id_img)
    print(
        f"[4] First segmentation: total regions {len(region_color_id)} "
        f"({time.perf_counter() - t:.2f}s), "
        f"mem={get_memory()}"
    )

    # cluster_id_img is not needed after the first segmentation
    del cluster_id_img

    # 6. Neighbour graph & merge small regions
    t = time.perf_counter()
    cluster_id_final, palette_merged = merge_small_regions_scipy(
        region_id_img=region_id_img,
        region_color_id=region_color_id,
        region_area=region_area,
        palette=palette_final,
        min_region_pixels=min_region_pixels,
        allow_fallback=True,
    )
    del region_id_img, region_color_id, region_area, palette_final
    print(
        f"[6] Merge small regions ({time.perf_counter() - t:.2f}s), "
        f"mem={get_memory()}"
    )

    # 7. Final clean-up segmentation
    t = time.perf_counter()
    cluster_id_refined, palette_refined = clean_small_final_regions(
        cluster_id_final,
        palette_merged,
        min_final_region_pixels=min_region_pixels,
        return_regions=False,
    )
    print(
        f"[7] Second segmentation: ({time.perf_counter() - t:.2f}s), "
        f"mem={get_memory()}"
    )

    # cluster_id_final and palette_merged are not needed after cleanup
    del cluster_id_final, palette_merged

    # Additional hard cleanup to guarantee a minimum region size
    t = time.perf_counter()
    cluster_id_hard, palette_hard = hard_cleanup_tiny_regions(
        cluster_id_refined,
        palette_refined,
        hard_min_pixels=min_region_pixels,
        max_iters=3,
    )
    print(
        f"[7b] hard cleanup: ({time.perf_counter() - t:.2f}s), "
        f"mem={get_memory()}"
    )

    cluster_id_refined = cluster_id_hard
    palette_refined = palette_hard
    del cluster_id_hard, palette_hard

    # Final regions for rendering
    t = time.perf_counter()
    final_regions = segment_final_regions_scipy(
        cluster_id_refined,
        connectivity4=CONNECTIVITY4,
    )
    num_regions = len(final_regions)
    print(
        f"[7c] Final regions for rendering: {num_regions} "
        f"({time.perf_counter() - t:.2f}s), "
        f"mem={get_memory()}"
    )

    # 8. Palette ordering & numbering (based on final palette)
    t = time.perf_counter()
    color_id_to_paint_index, paint_palette = build_ordered_palette(palette_refined)
    print(
        f"[8] Palette ordering & numbering: {len(paint_palette)} "
        f"({time.perf_counter() - t:.2f}s), "
        f"mem={get_memory()}"
    )

    difficulty = classify_difficulty(num_regions)
    print(
        f"[8b] Complexity: regions={num_regions}, difficulty={difficulty}, "
        f"mem={get_memory()}"
    )

    # 9. High-res rendering
    t = time.perf_counter()
    DPI = 300
    target_long_px = int(round(DPI * (print_long_mm / 25.4)))

    outline_img, colored_img, labels_big, scale_render = (
        render_outline_and_colored_highres(
            cluster_id_refined,
            palette_refined,
            color_id_to_paint_index,
            target_long_px=target_long_px,
        )
    )
    colored_img.save(colored_path, dpi=(DPI, DPI))
    del colored_img

    print(
        f"[9] Rendering ({time.perf_counter() - t:.2f}s), "
        f"mem={get_memory()}"
    )

    t = time.perf_counter()
    font_size_px_hi = estimate_font_size_px_for_print(
        image_long_px=target_long_px,
        print_long_mm=print_long_mm,
    )

    draw_numbers_on_outline_highres(
        outline_img,
        final_regions,
        labels_big,
        color_id_to_paint_index,
        font_size=font_size_px_hi,
        scale=scale_render,
    )

    outline_img.save(outline_path, dpi=(DPI, DPI))

    # Drop heavy rendering data as soon as files are written
    del (
        cluster_id_refined,
        palette_refined,
        final_regions,
        labels_big,
        outline_img,
    )

    print(
        f"[9b] draw_numbers_on_outline_highres ({time.perf_counter() - t:.2f}s), "
        f"mem={get_memory()}"
    )

    # Palette CSV is tiny; saving here does not affect memory much
    save_palette_csv(palette_csv_path, paint_palette)

    return paint_palette, num_regions, difficulty


def main(
    input_path: str,
    paper_size: str = "A3",
    num_colors: int = DEFAULT_NUM_COLORS,
    min_feature_mm: float = DEFAULT_MIN_FEATURE_MM,
    area_factor: float = DEFAULT_AREA_FACTOR,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> None:
    """
    End-to-end pipeline for building a paint-by-numbers booklet:

      1. Load and resize image for the chosen paper size.
      2. KMeans quantization in Lab and smoothing.
      3. Region segmentation, merging, cleanup, and numbering.
      4. High-res outline and reference colored render.
      5. Save palette CSV.
      6. Build 2-page PDF booklet.
    """
    #start_memory_sampler("mem_trace.csv", interval=0.05)
    np.random.seed(random_seed)
    t0 = time.perf_counter()

    input_path = os.path.expanduser(input_path)
    input_path = str(Path(input_path).resolve())
    paths = build_output_paths(input_path)

    # Target paper
    print_long_mm = get_paper_long_side_mm(paper_size)
    print(
        f"[0] Target paper: {paper_size} (long ≈ {print_long_mm:.0f} mm), "
        f"num_colors={num_colors}, min_feature≈{min_feature_mm} mm, "
        f"mem={get_memory()}"
    )

    # 1. Load + resize
    orig_arr, image_long_px = load_and_resize_for_print(
        input_path,
        print_long_mm=print_long_mm,
    )

    min_region_pixels = estimate_min_region_pixels(
        image_long_px=image_long_px,
        print_long_mm=print_long_mm,
        min_feature_mm=min_feature_mm,
        area_factor=area_factor,
        max_effective_dpi=DEFAULT_MAX_EFFECTIVE_DPI,
    )

    # 2–3. Quantization + smoothing
    cluster_id_img, palette_final = run_quantization_and_smoothing(
        orig_arr=orig_arr,
        num_colors=num_colors,
        image_long_px=image_long_px,
        print_long_mm=print_long_mm,
        min_feature_mm=min_feature_mm,
        area_factor=area_factor,
        quant_output_path=paths["quant"],
    )

    # 4–9. Regions, palette ordering, rendering, and palette CSV
    paint_palette, num_regions, difficulty = run_regions_and_render(
        cluster_id_img=cluster_id_img,
        palette_final=palette_final,
        print_long_mm=print_long_mm,
        min_region_pixels=min_region_pixels,
        outline_path=paths["outline"],
        colored_path=paths["colored"],
        palette_csv_path=paths["palette_csv"],
    )

    # These are no longer needed after palette CSV and renders are written
    del cluster_id_img, palette_final, paint_palette
    
    # 10–11. PDF booklet
    t_pdf = time.perf_counter()
    root, _ = os.path.splitext(input_path)
    build_pbn_pdf_booklet(
        root=root,
        original_path=input_path,
        outline_path=str(paths["outline"]),
        palette_csv_path=str(paths["palette_csv"]),
        pdf_name=paths["pdf"],
        paper_size=paper_size,
        num_regions=num_regions,
        difficulty=difficulty,
    )
    print(
        f"[11] PDF booklet generation ({time.perf_counter() - t_pdf:.2f}s), "
        f"mem={get_memory()}"
    )

    print(
        f"[done] Total time: {time.perf_counter() - t0:.2f}s, "
        f"mem={get_memory()}"
    )
    #stop_memory_sampler()
