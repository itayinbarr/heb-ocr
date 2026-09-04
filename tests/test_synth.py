import numpy as np
import torch
from PIL import Image

from hebocr.charset import Charset
from hebocr.data.synth import PixelBudgetSampler, collate, concat_rtl


def _marked(width: int, mark_left: bool) -> Image.Image:
    """A white strip with a black block at one end, so we can find it later."""
    arr = np.full((64, width), 255, np.uint8)
    if mark_left:
        arr[:, :10] = 0
    else:
        arr[:, -10:] = 0
    return Image.fromarray(arr, mode="L")


def test_concat_places_the_first_line_on_the_right():
    """Hebrew is right-to-left: joining "A" then "B" puts B's pixels LEFT of A.

    Getting this backwards trains the model on reversed text against forward
    labels, which is silent -- the loss still falls -- and ruinous.
    """
    first = _marked(200, mark_left=True)   # black block at ITS left edge
    second = Image.fromarray(np.full((64, 200), 255, np.uint8), mode="L")

    joined, text = concat_rtl([first, second], ["A", "B"])
    assert text == "A B"

    ink_columns = np.where(np.asarray(joined).min(axis=0) < 128)[0]
    # `first` occupies the right-hand half, so its mark lands past the midpoint.
    assert ink_columns.min() > joined.width * 0.5


def test_concat_label_order_is_logical_not_visual():
    parts = ["אחד", "שתיים", "שלוש"]
    images = [Image.fromarray(np.full((64, 120), 255, np.uint8), mode="L") for _ in parts]
    _, text = concat_rtl(images, parts)
    assert text == "אחד שתיים שלוש"


def test_concat_respects_the_width_cap():
    images = [Image.fromarray(np.full((64, 500), 255, np.uint8), mode="L") for _ in range(6)]
    joined, text = concat_rtl(images, list("abcdef"), max_width=1200)
    assert joined.width <= 1200
    assert len(text.split()) < 6  # it stopped early rather than overrunning


def test_concat_of_one_line_is_a_passthrough():
    image = _marked(100, mark_left=True)
    joined, text = concat_rtl([image], ["only"])
    assert text == "only" and joined.size == image.size


def test_pixel_budget_sampler_never_exceeds_its_budget():
    """This is the invariant that stops the 8 GB card from OOMing."""
    rng = np.random.default_rng(0)
    widths = rng.integers(150, 2400, size=3000)
    budget = 20000
    sampler = PixelBudgetSampler(widths, budget=budget, max_batch=64, concat_prob=0.0, seed=1)
    for batch in sampler:
        widest = max(widths[i] for group in batch for i in group)
        assert len(batch) * widest <= budget or len(batch) == 1


def test_pixel_budget_sampler_covers_every_sample_once():
    widths = np.random.default_rng(2).integers(150, 2400, size=1500)
    sampler = PixelBudgetSampler(widths, budget=20000, concat_prob=0.0, seed=3)
    seen = [i for batch in sampler for group in batch for i in group]
    assert sorted(seen) == list(range(len(widths)))


def test_pixel_budget_sampler_concat_groups_stay_within_max_width():
    widths = np.random.default_rng(4).integers(150, 1200, size=800)
    sampler = PixelBudgetSampler(
        widths, budget=20000, concat_prob=1.0, max_concat=3, max_width=2400, seed=5
    )
    for batch in sampler:
        for group in batch:
            assert len(group) <= 3


def test_collate_pads_on_the_right_with_background():
    charset = Charset.default()
    items = [
        (torch.ones(1, 64, 100), charset.encode("א"), "א"),
        (torch.ones(1, 64, 250), charset.encode("בג"), "בג"),
    ]
    batch = collate(items)
    assert batch.images.shape == (2, 1, 64, 250)
    assert batch.widths.tolist() == [100, 250]
    # Ink is high-valued, so padding must be zero -- background, not ink.
    assert batch.images[0, 0, :, 100:].abs().max() == 0
    assert batch.target_lengths.tolist() == [1, 2]
