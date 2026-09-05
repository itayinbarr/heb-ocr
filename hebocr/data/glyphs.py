"""Compose training lines out of real handwritten Hebrew letters.

Motivation. Training purely on DiffusionPen leaves a large, measurable
transfer gap: by epoch 5 the model read synthetic validation lines at 0.095 CER
while the real benchmark sat at 0.475. Synthetic validation was close to solved,
so the remaining error was not capacity or fitting -- it was that generated ink
does not look like ballpoint on paper.

HHD (`sivan22/hebrew-handwritten-dataset`, CC-BY-3.0) is ~4k images of
*genuinely handwritten* Hebrew letters: all 27 letters including the five final
forms, plus comma. Individually they are useless for line recognition, but
stitched into lines they contribute exactly what DiffusionPen cannot -- real
stroke texture, real pen width, real letter-shape variation.

The honest caveat: a composed line mixes glyphs from many writers, so it has no
consistent hand and its spacing is synthetic. It is a complement to
DiffusionPen, not a replacement, and is mixed in at a modest rate.

Hebrew makes this far more workable than Latin would: printed and everyday
handwritten Hebrew letters are largely disconnected, so butting glyphs together
is not the gross distortion that breaking Latin cursive ligatures would be.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

HHD = "sivan22/hebrew-handwritten-dataset"

# Letters that hang below the baseline, and the one that rises above the others.
DESCENDERS = set("ךןףץק")
ASCENDERS = set("ל")

# Where each glyph's *bottom* sits, as a multiple of x-height relative to the
# baseline. Positive means below the baseline.
#
# Hebrew is unusually uniform in height, but not entirely: yod is a small mark
# written high in the letter band, not a full-height letter sitting on the
# baseline, and the final forms drop a tail below it. Measured on HHD, yod's
# median glyph is 46 px against 96 px for a plain letter and the descenders run
# 1.47x -- so scaling every glyph to one height, as the first version did, drew
# yod at twice its proper size in a fifth of the training data.
BASELINE_OFFSET = {
    "י": -0.42,   # floats above the baseline
    ",": 0.22,    # dips below it
}
DESCENDER_DROP = 0.45


def _tight_crop(image: Image.Image) -> np.ndarray:
    """Grayscale, inverted-threshold crop down to the ink."""
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    ink = gray < max(int(gray.mean()) - 20, 40)
    rows, cols = np.where(ink)
    if len(rows) == 0:
        return gray
    return gray[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


@dataclass
class GlyphBank:
    """Real handwritten glyphs, indexed by character.

    `height_ratio` records how tall each letter naturally is relative to an
    ordinary letter, measured from the corpus itself rather than hardcoded, so
    composed lines keep real Hebrew proportions.
    """

    glyphs: dict[str, list[np.ndarray]]
    height_ratio: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.height_ratio is None:
            self.height_ratio = self._measure_ratios()

    def _measure_ratios(self) -> dict[str, float]:
        medians = {
            c: float(np.median([g.shape[0] for g in gs]))
            for c, gs in self.glyphs.items() if gs
        }
        plain = [
            h for c, h in medians.items()
            if c not in DESCENDERS and c not in ASCENDERS and c.isalpha()
        ]
        base = float(np.median(plain)) if plain else 1.0
        return {c: h / base for c, h in medians.items()} if base else {}

    @property
    def alphabet(self) -> set[str]:
        return set(self.glyphs)

    @classmethod
    def load(cls, split: str = "train", max_per_class: int | None = None) -> "GlyphBank":
        from datasets import load_dataset

        dataset = load_dataset(HHD, split=split)
        names = dataset.features["label"].names
        bank: dict[str, list[np.ndarray]] = {}
        for row in dataset:
            char = names[row["label"]]
            bucket = bank.setdefault(char, [])
            if max_per_class is None or len(bucket) < max_per_class:
                bucket.append(_tight_crop(row["image"]))
        return cls(glyphs=bank)

    def can_render(self, text: str) -> bool:
        return all(c == " " or c in self.glyphs for c in text)

    def renderable(self, text: str) -> str:
        """Drop characters with no real glyph, collapsing the whitespace left."""
        kept = "".join(c for c in text if c == " " or c in self.glyphs)
        return " ".join(kept.split())


def compose_line(
    bank: GlyphBank,
    text: str,
    rng: np.random.Generator,
    height: int = 64,
    x_height_ratio: float = 0.5,
) -> tuple[Image.Image, str] | tuple[None, str]:
    """Stitch real glyphs into one line image. Returns (image, actual_text).

    `actual_text` is the transcription of what was drawn, which can be shorter
    than `text` if some characters had no glyph -- returning the requested text
    instead would pair an image with a label describing ink that is not in it.
    """
    text = bank.renderable(text)
    if not text:
        return None, ""

    x_height = max(8, int(height * x_height_ratio))
    letter_gap = max(1, int(x_height * 0.12))
    space_gap = max(3, int(x_height * 0.55))
    baseline = int(height * 0.72)

    pieces: list[tuple[np.ndarray, int]] = []  # (glyph, vertical offset of its top)
    width = 0
    for char in text:
        if char == " ":
            width += space_gap
            pieces.append((None, 0))
            continue

        choices = bank.glyphs[char]
        glyph = choices[int(rng.integers(len(choices)))]

        # Scale to this letter's own natural height, not to one common height.
        ratio = (bank.height_ratio or {}).get(char, 1.0)
        target_h = x_height * ratio * rng.uniform(0.88, 1.12)
        scale = target_h / max(glyph.shape[0], 1)
        new_w = max(2, int(glyph.shape[1] * scale))
        new_h = max(2, int(glyph.shape[0] * scale))
        glyph = cv2.resize(glyph, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Place by where the glyph's bottom belongs relative to the baseline.
        if char in DESCENDERS:
            bottom = baseline + x_height * DESCENDER_DROP
        else:
            bottom = baseline + x_height * BASELINE_OFFSET.get(char, 0.0)
        top = int(bottom) - new_h
        top += int(rng.normal(0, x_height * 0.05))  # baseline wobble
        pieces.append((glyph, top))
        width += new_w + letter_gap

    if width < 8:
        return None, ""

    canvas = np.full((height, width + 8), 255, np.uint8)
    x = 4
    # Hebrew is right-to-left, so the logically first character must sit at the
    # RIGHT edge. That means pasting the pieces in reverse order. Mirroring the
    # finished canvas would put them in the right places but leave every letter
    # backwards, which is a much harder bug to see than to avoid.
    for glyph, top in reversed(pieces):
        if glyph is None:
            x += space_gap
            continue
        h, w = glyph.shape
        y0 = int(np.clip(top, 0, height - 1))
        y1 = min(y0 + h, height)
        if y1 > y0:
            region = canvas[y0:y1, x : x + w]
            # Darkest wins, so overlapping strokes merge like ink rather than
            # one glyph's white box erasing its neighbour.
            canvas[y0:y1, x : x + w] = np.minimum(region, glyph[: y1 - y0, :])
        x += w + letter_gap

    return Image.fromarray(canvas), text


class GlyphLineDataset:
    """Endless supply of glyph-composed lines, drawn from a text corpus."""

    def __init__(self, bank: GlyphBank, texts: list[str], seed: int = 0):
        self.bank = bank
        self.texts = [t for t in (bank.renderable(t) for t in texts) if len(t) >= 4]
        self.seed = seed

    def __len__(self) -> int:
        return len(self.texts)

    def sample(self, rng: np.random.Generator):
        for _ in range(8):  # a few attempts before giving up on a bad draw
            text = self.texts[int(rng.integers(len(self.texts)))]
            image, actual = compose_line(self.bank, text, rng)
            if image is not None and actual:
                return image, actual
        return None, ""
