from __future__ import annotations

import os
import time

import numpy as np
from PIL import Image, ImageOps

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

from pathlib import Path

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
    # how many pixels we effectively use for the long side at max_effective_dpi
    max_effective_px = int(print_long_mm / 25.4 * max_effective_dpi)
    effective_long_px = min(image_long_px, max_effective_px)

    # mm per pixel on paper
    mm_per_px = print_long_mm / float(effective_long_px)

    # desired text height in pixels
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

    # how many pixels we reasonably need on the long side
    max_long_px = int(round(print_long_mm / 25.4 * max_effective_dpi))

    if long_px <= max_long_px:
        # already small enough, no need to resize
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


def analyze_final_regions(final_regions, print_long_mm: float, image_long_px: int):
    """
    final_regions: output of segment_final_regions
    print_long_mm: long side of paper in mm (A4/A3/A2)
    image_long_px: long side of processed image in px
    """
    mm_per_px = print_long_mm / float(image_long_px)
    mm2_per_px = mm_per_px ** 2

    areas_px = np.array([reg["area"] for reg in final_regions], dtype=np.float64)
    areas_mm2 = areas_px * mm2_per_px

    total = len(final_regions)
    smaller_1 = int((areas_mm2 < 1.0).sum())
    smaller_2 = int((areas_mm2 < 2.0).sum())

    print("\n[Region statistics]")
    print(f"Total regions:        {total}")
    print(f"Area < 1 mm²:         {smaller_1}")
    print(f"Area < 2 mm²:         {smaller_2}")
    print(f"Min area px:          {areas_px.min():.0f}")
    print(f"Min area mm²:         {areas_mm2.min():.3f}")
    print(f"Median area mm²:      {np.median(areas_mm2):.3f}")
    print(f"1st percentile mm²:   {np.percentile(areas_mm2, 1):.3f}")
    print(f"5th percentile mm²:   {np.percentile(areas_mm2, 5):.3f}")
    print(f"25th percentile mm²:  {np.percentile(areas_mm2, 25):.3f}")

    return


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

      1. Load image.
      2. KMeans quantization in Lab.
      3. Smooth cluster map.
      4. Connected components segmentation.
      5. Merge small regions into neighbouring big ones.
      6. Optional final clean-up segmentation.
      7. Order palette and assign paint indices.
      8. Render outline and reference colored image.
      9. Save palette CSV and PNG.
      10. Build 2-page PDF booklet.
    """
    np.random.seed(random_seed)
    t0 = time.perf_counter()
    input_path = os.path.expanduser(input_path)
    input_path = str(Path(input_path).resolve())
    paths = build_output_paths(input_path)

    # 0. Target paper size
    print_long_mm = get_paper_long_side_mm(paper_size)
    print(
        f"[0] Target paper: {paper_size} (long ≈ {print_long_mm:.0f} mm), "
        f"num_colors={num_colors}, min_feature≈{min_feature_mm} mm"
    )

    # 1. Load and possibly resize image
    orig_img = Image.open(input_path)
    orig_img = ImageOps.exif_transpose(orig_img).convert("RGB")
    orig_img_resized, scale = resize_for_print(
        orig_img,
        print_long_mm=print_long_mm,
        max_effective_dpi=DEFAULT_MAX_EFFECTIVE_DPI/2,
    )

    orig_arr = np.asarray(orig_img_resized).astype(np.uint8)
    H, W, _ = orig_arr.shape
    image_long_px = max(H, W)
    print(f"[0] Input (effective): size: {W}x{H}, long={image_long_px}px")

    min_region_pixels = estimate_min_region_pixels(
        image_long_px=image_long_px,
        print_long_mm=print_long_mm,
        min_feature_mm=min_feature_mm,
        area_factor=area_factor,
        max_effective_dpi=DEFAULT_MAX_EFFECTIVE_DPI,
    )

    # 2. Quantization
    t = time.perf_counter()
    cluster_id_raw, palette_colors, quant_arr = quantize_kmeans_lab(
        orig_arr, num_colors
    )
    Image.fromarray(quant_arr, mode="RGB").save(paths["quant"])
    print(
        f"[1] KMeans quantization (Lab) -> {paths['quant']} "
        f"({time.perf_counter() - t:.2f}s)"
    )
    print(f"    Initial clusters: {len(palette_colors)}")

    # 3. Smooth cluster map
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

    print(f"[3] Cluster map smoothing ({time.perf_counter() - t:.2f}s)")

    # 4. First segmentation
    t = time.perf_counter()
    region_id_img, region_color_id, region_area = segment_regions_scipy(cluster_id_img)
    print(
        f"[4] First segmentation: total regions {len(region_color_id)} "
        f"({time.perf_counter() - t:.2f}s)"
    )

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
    print(
        f"[6] Merge small regions ({time.perf_counter() - t:.2f}s)"
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
        f"[7] Second segmentation: ({time.perf_counter() - t:.2f}s)"
    )
    # hard cleanup: no region smaller than min_region_pixels
    t = time.perf_counter()
    cluster_id_hard, palette_hard = hard_cleanup_tiny_regions(
        cluster_id_refined,
        palette_refined,
        hard_min_pixels=min_region_pixels,
        max_iters=3,
    )
    print(f"[7b] hard cleanup: ({time.perf_counter() - t:.2f}s)")
    
    cluster_id_refined = cluster_id_hard
    palette_refined = palette_hard

    t = time.perf_counter()
    final_regions = segment_final_regions_scipy(cluster_id_refined, connectivity4=CONNECTIVITY4)
    print(f"[7c] Final regions for rendering: {len(final_regions)} ({time.perf_counter() - t:.2f}s)")
    
    # 8. Palette ordering & numbering (based on final palette)
    t = time.perf_counter()
    color_id_to_paint_index, paint_palette = build_ordered_palette(palette_refined)
    print(
        f"[8] Palette ordering & numbering: {len(paint_palette)} "
        f"({time.perf_counter() - t:.2f}s)"
    )

    DPI = 300
    target_long_px = int(round(DPI * (print_long_mm / 25.4)))
    
    outline_img, colored_img, labels_big, scale_render = render_outline_and_colored_highres(
        cluster_id_refined, 
        palette_refined,
        color_id_to_paint_index,
        target_long_px=target_long_px,
    )
    
    # font size so that height on paper ~ TARGET_NUMBER_HEIGHT_MM
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
    
    outline_img.save(paths["outline"], dpi=(DPI, DPI))
    colored_img.save(paths["colored"], dpi=(DPI, DPI))
    
    print(f"[9] Rendering ({time.perf_counter() - t:.2f}s)")

    save_palette_csv(paths["palette_csv"], paint_palette)
    t = time.perf_counter()

    # 11. PDF booklet
    t_pdf = time.perf_counter()
    root, _ = os.path.splitext(input_path)
    build_pbn_pdf_booklet(
        root=root,
        original_path=input_path,
        outline_path=paths["outline"],
        palette_csv_path=paths["palette_csv"],
        pdf_name=paths['pdf'],
        paper_size=paper_size,
    )
    print(f"[11] PDF booklet generation ({time.perf_counter() - t_pdf:.2f}s)")

    print(f"[done] Total time: {time.perf_counter() - t0:.2f}s")
