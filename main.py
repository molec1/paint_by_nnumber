from __future__ import annotations

import sys

from pipeline import main


if __name__ == "__main__":
    """
    Minimal CLI:

      python main.py input_image.png

    If no argument is given, uses "kalemegdan.png" by default.
    """
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        input_path = "test.png"

    if len(sys.argv) > 2:
        paper_size = sys.argv[2]
    else:
        paper_size = "A3"

    if len(sys.argv) > 3:
        min_feature_mm = sys.argv[3]
    else:
        min_feature_mm = 2.

    main(input_path, paper_size=paper_size, min_feature_mm=min_feature_mm)
