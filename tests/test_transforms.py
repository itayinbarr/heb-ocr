import numpy as np
from PIL import Image

from hebocr.data.transforms import LINE_HEIGHT, MAX_WIDTH, Augment, preprocess


def _line(width=400, height=100):
    arr = np.full((height, width), 240, np.uint8)
    arr[height // 3 : 2 * height // 3, 20 : width - 20] = 30  # a dark ink band
    return Image.fromarray(arr, mode="L")


def test_preprocess_normalizes_height_and_range():
    out = preprocess(_line(400, 137))
    assert out.shape[0] == 1 and out.shape[1] == LINE_HEIGHT
    assert out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_preprocess_makes_ink_high_valued():
    """Zero padding a batch must add background, so ink has to be the high end."""
    out = preprocess(_line())
    rows = out[0].mean(axis=1)
    assert rows[LINE_HEIGHT // 2] > rows[2]


def test_preprocess_preserves_aspect_ratio():
    out = preprocess(_line(800, 200))
    assert out.shape[2] == 800 * LINE_HEIGHT // 200


def test_preprocess_caps_extreme_widths():
    out = preprocess(_line(40000, 100))
    assert out.shape[2] == MAX_WIDTH


def test_preprocess_accepts_rgb_and_arrays():
    colour = Image.merge("RGB", [_line()] * 3)
    assert preprocess(colour).shape == preprocess(_line()).shape
    assert preprocess(np.asarray(_line())).shape[1] == LINE_HEIGHT


def test_preprocess_survives_a_blank_image():
    """A flat crop must not divide by zero on contrast normalization."""
    out = preprocess(Image.fromarray(np.full((80, 300), 255, np.uint8), mode="L"))
    assert np.isfinite(out).all()


def test_augment_output_matches_the_preprocess_contract():
    augment = Augment(np.random.default_rng(0))
    for _ in range(25):
        out = augment(_line())
        assert out.shape[0] == 1 and out.shape[1] == LINE_HEIGHT
        assert out.shape[2] >= 1 and np.isfinite(out).all()
        assert 0.0 <= out.min() and out.max() <= 1.0


def test_augment_is_actually_random():
    augment = Augment(np.random.default_rng(0))
    outputs = [augment(_line()) for _ in range(6)]
    assert len({o.shape[2] for o in outputs}) > 1 or not np.allclose(outputs[0], outputs[1])


def test_augment_is_reproducible_from_a_seed():
    a = Augment(np.random.default_rng(7))(_line())
    b = Augment(np.random.default_rng(7))(_line())
    assert np.allclose(a, b)


def test_augment_compresses_toward_real_line_density():
    """Real hands pack ~20 px/char against DiffusionPen's ~34.5, so the
    stretch range has to reach well below 1.0."""
    augment = Augment(np.random.default_rng(3))
    widths = [augment(_line(600, 64)).shape[2] for _ in range(80)]
    assert min(widths) < 600 * 0.75
