import numpy as np
from PIL import Image

from hebocr.page.segment import (
    LineBox, _oriented_line_kernel, binarize, estimate_line_pitch,
    estimate_skew, find_lines, reading_order, remove_rules, segment_page,
)


def synthetic_page(n_lines=8, pitch=60, width=800, rule_slope=0.0, rules=False, seed=0):
    """A fake ruled page of glyph-like marks at a fixed line pitch.

    Deliberately not solid bars: a bar is one long horizontal run, which is the
    very shape `remove_rules` is built to delete, and its autocorrelation looks
    nothing like text. Discrete marks with inter-word gaps exercise the same
    code paths real handwriting does.
    """
    rng = np.random.default_rng(seed)
    height = pitch * (n_lines + 2)
    arr = np.full((height, width), 245, np.uint8)
    glyph_h = pitch // 3
    for i in range(n_lines):
        y = pitch * (i + 1)
        x = 60
        while x < width - 80:
            w = int(rng.integers(8, 18))
            arr[y : y + glyph_h, x : x + w] = 40
            x += w + int(rng.integers(4, 9))          # letter gap
            if rng.random() < 0.18:
                x += int(rng.integers(12, 25))        # word gap
    if rules:
        for i in range(n_lines + 1):
            y0 = int(pitch * (i + 0.85))
            for x in range(width):
                y = int(y0 + rule_slope * x)
                if 0 <= y < height:
                    arr[y : y + 2, x] = 120
    return Image.fromarray(arr, mode="L")


def test_oriented_kernel_is_a_line_of_the_requested_slope():
    kernel = _oriented_line_kernel(101, 0.0)
    assert kernel.shape == (1, 101) and kernel.sum() == 101
    tilted = _oriented_line_kernel(101, 2.0)
    assert tilted.shape[0] > 1 and tilted.sum() >= 101 - 2


def test_pitch_estimate_recovers_known_line_spacing():
    """Pitch drives every other threshold, so it has to be close."""
    binary, scale = binarize(synthetic_page(n_lines=10, pitch=60))
    assert abs(estimate_line_pitch(binary) - 60 * scale) < 60 * scale * 0.25


def test_pitch_estimate_survives_a_blank_page():
    binary, _ = binarize(Image.fromarray(np.full((600, 600), 255, np.uint8), mode="L"))
    assert estimate_line_pitch(binary) > 0


def test_removing_sloped_rules_deletes_ink():
    """A flat kernel misses tilted rules -- the bug that fused every line."""
    page = synthetic_page(n_lines=8, rules=True, rule_slope=0.02)
    binary, _ = binarize(page)
    pitch = estimate_line_pitch(binary)
    assert remove_rules(binary, pitch).sum() < binary.sum()


def test_removing_rules_keeps_the_text():
    page = synthetic_page(n_lines=8, rules=True, rule_slope=0.02)
    binary, _ = binarize(page)
    cleaned = remove_rules(binary, estimate_line_pitch(binary))
    assert cleaned.sum() > binary.sum() * 0.4  # rules went, writing stayed


def test_finds_roughly_the_right_number_of_lines():
    page = synthetic_page(n_lines=8, pitch=60)
    binary, _ = binarize(page)
    pitch = estimate_line_pitch(binary)
    assert 6 <= len(find_lines(binary, pitch)) <= 12


def test_skew_estimate_is_near_zero_on_a_level_page():
    binary, _ = binarize(synthetic_page())
    assert abs(estimate_skew(binary)) <= 1.0


def test_reading_order_is_top_to_bottom():
    boxes = [LineBox(0, 300, 100, 20), LineBox(0, 100, 100, 20), LineBox(0, 200, 100, 20)]
    assert [b.y for b in reading_order(boxes, pitch=40)] == [100, 200, 300]


def test_reading_order_is_right_to_left_within_a_row():
    """Two cells of the same table row: Hebrew reads the right one first."""
    left = LineBox(0, 100, 100, 20)
    right = LineBox(400, 102, 100, 20)
    assert reading_order([left, right], pitch=60)[0].x == 400


def test_crop_stays_inside_the_page():
    image = Image.new("RGB", (200, 100), "white")
    crop = LineBox(190, 95, 50, 30).crop(image)
    assert crop.width > 0 and crop.height > 0


def test_segment_page_returns_aligned_crops_and_boxes():
    crops, boxes, angle = segment_page(synthetic_page(n_lines=6), deskew=False)
    assert len(crops) == len(boxes)
    assert all(c.width > 0 and c.height > 0 for c in crops)
    assert angle == 0.0


def test_segment_page_handles_an_empty_page():
    blank = Image.fromarray(np.full((800, 600), 255, np.uint8), mode="L").convert("RGB")
    crops, boxes, _ = segment_page(blank)
    assert len(crops) == len(boxes)
