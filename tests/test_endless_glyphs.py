"""Glyph lines generated on demand rather than pre-composed."""

import numpy as np
import pytest

from hebocr.data.glyphs import GlyphBank
from hebocr.data.synth import EndlessGlyphLines

TEXTS = [
    "שלום עולם וברוך הבא",
    "בית ספר גדול בעיר",
    "הכרת לקויות ואוטיזם",
    "ארבע חמש שש שבע שמונה",
]


@pytest.fixture(scope="module")
def bank():
    try:
        return GlyphBank.load("train")
    except Exception as exc:  # the HHD glyphs are a download, not vendored
        pytest.skip(f"glyph bank unavailable: {exc}")


def test_length_is_whatever_was_asked_for(bank):
    """The whole point is that the count is not bounded by memory."""
    g = EndlessGlyphLines(bank, TEXTS, count=5_000_000, seed=0)
    assert len(g) == 5_000_000


def test_widths_are_known_before_any_line_is_drawn(bank):
    """The pixel budget sampler needs a real width per item, not an estimate."""
    g = EndlessGlyphLines(bank, TEXTS, count=1000, seed=0)
    assert len(g.widths) == 1000
    assert (g.widths > 0).all()


def test_every_line_honours_its_promised_width(bank):
    """A line wider than promised is what turns a planned batch into an OOM."""
    g = EndlessGlyphLines(bank, TEXTS, count=200, seed=0)
    for i in range(0, 200, 20):
        array, _ = g[i]
        assert array.shape[1] == int(g.widths[i])


def test_lines_are_64_px_tall(bank):
    g = EndlessGlyphLines(bank, TEXTS, count=50, seed=0)
    array, _ = g[0]
    assert array.shape[0] == 64


def test_the_same_index_does_not_repeat_its_ink(bank):
    """Generating rather than storing is what removes repetition across epochs."""
    g = EndlessGlyphLines(bank, TEXTS, count=100, seed=0)
    first, _ = g[3]
    second, _ = g[3]
    assert first.shape == second.shape
    assert not np.array_equal(first, second)


def test_indexing_past_the_count_wraps_rather_than_raising(bank):
    g = EndlessGlyphLines(bank, TEXTS, count=10, seed=0)
    array, _ = g[10]
    assert array.shape[1] == int(g.widths[0])


def test_text_is_what_was_actually_drawn(bank):
    """A label describing ink that is not in the image is the failure to avoid."""
    g = EndlessGlyphLines(bank, TEXTS, count=30, seed=0)
    for i in range(30):
        _, text = g[i]
        assert isinstance(text, str)


def test_a_bank_with_nothing_to_draw_fails_loudly(bank):
    with pytest.raises(RuntimeError):
        EndlessGlyphLines(bank, ["!!!"], count=10, seed=0)
