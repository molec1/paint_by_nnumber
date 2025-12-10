from __future__ import annotations

import os
import time

import numpy as np
from PIL import Image

from config import (
    DEFAULT_NUM_COLORS,
    DEFAULT_MIN_FEATURE_MM,
    DEFAULT_AREA_FACTOR,
    DEFAULT_MAX_EFFECTIVE_DPI,
    LINE_THICKNESS_PX,
    get_paper_long_side_mm,
)
from quantization import quantize_kmeans_lab, smooth_cluster_map
from smoothing import estimate_min_region_pixels
from segmentation import (
    segment_regions,
    split_big_small_regions,
    build_adjacency_small_to_big,
    merge_small_regions,
    clean_small_final_regions,
)
from palette_utils import (
    build_ordered_palette,
    save_palette_csv,
    save_palette_image,
)
from rendering import render_outline_and_colored, draw_numbers_on_outline
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



def main(
    input_path: str,
    paper_size: str = "A3",
    num_colors: int = DEFAULT_NUM_COLORS,
    min_feature_mm: float = DEFAULT_MIN_FEATURE_MM,
    area_factor: float = DEFAULT_AREA_FACTOR,
    line_thickness_px: int = LINE_THICKNESS_PX,
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
    t0 = time.perf_counter()
    paths = build_output_paths(input_path)

    # 0. Target paper size
    print_long_mm = get_paper_long_side_mm(paper_size)

    # 1. Load image
    orig_img = Image.open(input_path).convert("RGB")
    orig_arr = np.asarray(orig_img).astype(np.uint8)
    H, W, _ = orig_arr.shape
    image_long_px = max(H, W)
    print(f"[0] Input: {input_path}, size: {W}x{H}")

    min_region_pixels = estimate_min_region_pixels(
        image_long_px=image_long_px,
        print_long_mm=print_long_mm,
        min_feature_mm=min_feature_mm,
        area_factor=area_factor,
        max_effective_dpi=DEFAULT_MAX_EFFECTIVE_DPI,
    )

    print(f"[0] MIN_REGION_PIXELS: {min_region_pixels}")

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
    )

    print(f"[3] Cluster map smoothing ({time.perf_counter() - t:.2f}s)")
    print(f"    Colors after smoothing: {len(palette_final)}")

    # 4. First segmentation
    t = time.perf_counter()
    regions, region_id_img = segment_regions(cluster_id_img)
    print(
        f"[4] First segmentation: total regions {len(regions)} "
        f"({time.perf_counter() - t:.2f}s)"
    )

    # 5. Split big/small
    big_region_ids, small_region_ids = split_big_small_regions(
        regions, min_region_pixels
    )
    print(f"[5] Big regions:   {len(big_region_ids)}")
    print(f"    Small regions: {len(small_region_ids)}")

    # 6. Neighbour graph & merge small regions
    t = time.perf_counter()
    adj_small_to_big = build_adjacency_small_to_big(
        region_id_img, big_region_ids, small_region_ids
    )
    cluster_id_final, palette_merged = merge_small_regions(
        cluster_id_img,
        regions,
        palette_final,
        big_region_ids,
        small_region_ids,
        adj_small_to_big,
    )
    print(
        f"[6] Merge small regions ({time.perf_counter() - t:.2f}s)"
    )
    print(f"    Colors after merging: {len(palette_merged)}")

    # 7. Final clean-up segmentation
    t = time.perf_counter()
    cluster_id_refined, palette_refined, final_regions = clean_small_final_regions(
        cluster_id_final,
        palette_merged,
        min_final_region_pixels=min_region_pixels,
    )
    print(
        f"[7] Second segmentation: final regions {len(final_regions)} "
        f"({time.perf_counter() - t:.2f}s)"
    )

    # 8. Palette ordering & numbering (based on final palette)
    t = time.perf_counter()
    color_id_to_paint_index, paint_palette = build_ordered_palette(palette_refined)
    print(
        f"[8] Palette ordering & numbering: {len(paint_palette)} "
        f"({time.perf_counter() - t:.2f}s)"
    )

    # 9. Rendering images
    t = time.perf_counter()
    outline_img, colored_img = render_outline_and_colored(
        final_regions,
        palette_refined,
        color_id_to_paint_index,
        line_thickness_px,
        size=(orig_arr.shape[0], orig_arr.shape[1]),
    )
    draw_numbers_on_outline(
        outline_img,
        final_regions,
        color_id_to_paint_index,
        font_size=28,
    )
    outline_img.save(paths["outline"])
    colored_img.save(paths["colored"])
    print(
        f"[9] Render images -> {paths['outline']}, {paths['colored']} "
        f"({time.perf_counter() - t:.2f}s)"
    )

    # 10. Palette CSV and image
    t = time.perf_counter()
    save_palette_csv(paths["palette_csv"], paint_palette)
    save_palette_image(paths["palette_img"], paint_palette)
    print(
        f"[10] Palette -> {paths['palette_csv']}, {paths['palette_img']} "
        f"({time.perf_counter() - t:.2f}s)"
    )

    # 11. PDF booklet
    t_pdf = time.perf_counter()
    root, _ = os.path.splitext(input_path)
    build_pbn_pdf_booklet(
        root=root,
        original_path=input_path,
        outline_path=paths["outline"],
        palette_csv_path=paths["palette_csv"],
        pdf_name=paths['pdf'],
    )
    print(f"[11] PDF booklet generation ({time.perf_counter() - t_pdf:.2f}s)")

    print(f"[done] Total time: {time.perf_counter() - t0:.2f}s")
