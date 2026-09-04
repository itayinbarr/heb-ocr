"""Measure how many gold lines the segmenter finds on the benchmark pages.

Reports recall at a coverage threshold, which is the number that matters for
full-page word coverage: a gold line the segmenter never crops is a line the
recognizer can never contribute words from.
"""

import argparse

import numpy as np

from hebocr.data.benchmark import load_pages
from hebocr.page.segment import segment_page


def covered_fraction(gold, dets) -> float:
    """Largest fraction of the gold box's area contained in a single detection."""
    gx0, gy0, gx1, gy1 = gold
    area = max((gx1 - gx0) * (gy1 - gy0), 1)
    best = 0.0
    for dx0, dy0, dx1, dy1 in dets:
        ix = max(0, min(gx1, dx1) - max(gx0, dx0))
        iy = max(0, min(gy1, dy1) - max(gy0, dy0))
        best = max(best, (ix * iy) / area)
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--no-deskew", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    total_gold = total_hit = total_det = 0
    for page in load_pages():
        _, boxes, angle = segment_page(page.image, deskew=not args.no_deskew)
        dets = [(b.x, b.y, b.x + b.w, b.y + b.h) for b in boxes]
        gold = [
            (l["bbox"]["x"], l["bbox"]["y"],
             l["bbox"]["x"] + l["bbox"]["w"], l["bbox"]["y"] + l["bbox"]["h"])
            for l in page.lines
        ]
        hits = sum(covered_fraction(g, dets) >= args.threshold for g in gold)
        total_gold += len(gold)
        total_hit += hits
        total_det += len(dets)
        if not args.quiet:
            print(
                f"{page.page_id[:8]}: gold {len(gold):3d}  det {len(dets):4d}  "
                f"found {hits:3d} ({hits/len(gold):4.0%})  skew {angle:+.1f}deg"
            )

    print(
        f"\nline recall (>={args.threshold:.0%} area covered): "
        f"{total_hit}/{total_gold} = {total_hit/total_gold:.1%}   detections {total_det}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
