from __future__ import annotations

from typing import List, Tuple

# Default processing parameters
DEFAULT_NUM_COLORS = 20
DEFAULT_MIN_FEATURE_MM = 2.0      # minimal paintable feature size
DEFAULT_AREA_FACTOR = 4.0         # relates feature size to pixel area
DEFAULT_MAX_EFFECTIVE_DPI = 250
LINE_THICKNESS_PX: int = 1     # contour thickness in pixels

# Connectivity: 4-connected (True) or 8-connected (False)
CONNECTIVITY4: bool = True

if CONNECTIVITY4:
    NEIGHBORS: List[Tuple[int, int]] = [
        (-1, 0), (1, 0),
        (0, -1), (0, 1),
    ]
else:
    NEIGHBORS = [
        (-1, 0), (1, 0),
        (0, -1), (0, 1),
        (-1, -1), (-1, 1),
        (1, -1), (1, 1),
    ]


# Target paper long side in millimetres for A-series
PAPER_LONG_SIDE_MM = {
    "A5": 210.0,
    "A4": 297.0,
    "A3": 420.0,
    "A2": 594.0,
    "A1": 840.0,
}


def get_paper_long_side_mm(paper_size: str) -> float:
    """
    Return the long side (in mm) for the given A-series paper size.
    Defaults to A3 if unknown.
    """
    size = (paper_size or "A3").upper()
    return PAPER_LONG_SIDE_MM.get(size, PAPER_LONG_SIDE_MM["A3"])