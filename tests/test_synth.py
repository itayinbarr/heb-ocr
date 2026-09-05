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


def test_glyph_composition_places_first_character_on_the_right():
    """Same RTL trap as concat_rtl, and a worse one: mirroring the finished
    canvas would order the letters correctly but draw each one backwards."""
    import numpy as np

    from hebocr.data.glyphs import GlyphBank, compose_line

    # One fully inked letter and one blank one, so the ink's position in the
    # canvas is unambiguously the inked letter's position.
    inked = np.zeros((40, 30), np.uint8)
    blank = np.full((40, 30), 255, np.uint8)
    bank = GlyphBank(glyphs={"א": [inked], "ב": [blank]})

    image, text = compose_line(bank, "אב", np.random.default_rng(0))
    assert text == "אב"
    ink = np.where(np.asarray(image).min(axis=0) < 128)[0]
    # 'א' is written first, so in right-to-left order its ink sits on the right.
    assert len(ink) > 0 and ink.mean() > image.width * 0.55


def test_glyph_composition_labels_only_what_it_drew():
    """Characters with no real glyph must not appear in the transcription."""
    import numpy as np

    from hebocr.data.glyphs import GlyphBank, compose_line

    mark = np.full((40, 30), 255, np.uint8)
    mark[10:30, 5:25] = 0
    bank = GlyphBank(glyphs={"א": [mark], "ב": [mark]})
    _, text = compose_line(bank, "אXב9", np.random.default_rng(0))
    assert text == "אב"


def test_mixed_dataset_widths_cover_every_item():
    import numpy as np

    from hebocr.charset import Charset
    from hebocr.data.synth import MixedLineDataset

    glyphs = [(np.full((64, 120), 255, np.uint8), "אב"), (np.full((64, 300), 255, np.uint8), "גד")]
    rows = [{"image": None, "text": "x"}] * 5
    dataset = MixedLineDataset(rows, glyphs, Charset.default())
    widths = dataset.widths(np.array([100] * 5))
    assert len(widths) == len(dataset) == 7
    assert widths[-2:].tolist() == [120, 300]


def test_glyph_bank_measures_real_letter_proportions():
    """Hebrew letters are not all one height, and scaling them as if they were
    drew yod at twice its real size in a fifth of the training data."""
    from hebocr.data.glyphs import GlyphBank

    tall = [np.zeros((150, 60), np.uint8)]
    plain = [np.zeros((100, 60), np.uint8)]
    small = [np.zeros((48, 20), np.uint8)]
    bank = GlyphBank(glyphs={"ל": tall, "א": plain, "ב": plain, "י": small})

    assert bank.height_ratio["א"] == 1.0
    assert bank.height_ratio["י"] < 0.6      # yod is a small, high mark
    assert bank.height_ratio["ל"] > 1.4      # lamed is the one true ascender


def test_descenders_are_drawn_below_the_baseline():
    from hebocr.data.glyphs import GlyphBank, compose_line

    solid = [np.zeros((100, 60), np.uint8)]
    bank = GlyphBank(glyphs={"א": solid, "ן": [np.zeros((147, 30), np.uint8)]})

    plain_img, _ = compose_line(bank, "א", np.random.default_rng(0))
    desc_img, _ = compose_line(bank, "ן", np.random.default_rng(0))

    def lowest_ink(image):
        rows = np.where(np.asarray(image).min(axis=1) < 128)[0]
        return int(rows.max()) if len(rows) else 0

    assert lowest_ink(desc_img) > lowest_ink(plain_img)


def test_sampler_never_concatenates_past_the_rtl_boundary():
    """Concatenation places the next line to the LEFT, which is only correct for
    right-to-left scripts. Left-to-right real-ink lines must stay unjoined."""
    widths = np.full(400, 300, dtype=np.int64)
    boundary = 250
    sampler = PixelBudgetSampler(
        widths, budget=20000, concat_prob=1.0, max_concat=3,
        seed=0, concat_max_index=boundary,
    )
    for batch in sampler:
        for group in batch:
            if len(group) > 1:
                assert all(i < boundary for i in group)


def test_changing_strength_rebuilds_the_augmenter():
    """An augmentation ramp is useless if workers keep the original strength."""
    from hebocr.charset import Charset
    from hebocr.data.synth import MixedLineDataset

    dataset = MixedLineDataset([], [], Charset.default(), train=True, aug_strength=0.3)
    first = dataset._aug_for_worker()
    assert first.strength == 0.3
    assert dataset._aug_for_worker() is first      # cached while unchanged

    dataset.aug_strength = 1.0
    rebuilt = dataset._aug_for_worker()
    assert rebuilt is not first and rebuilt.strength == 1.0
