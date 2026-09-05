"""Loader for the ivrit.ai Hebrew Handwriting OCR benchmark.

This is a *test set only*. The dataset card is explicit: "There is no train
split, by design -- it exists to be held out." Nothing in this module is
reachable from the training code, and `assert_not_trainable` exists so that
stays true.
"""

from dataclasses import dataclass
from functools import lru_cache

from PIL import Image

DATASET = "ivrit-ai/hebrew-handwriting-ocr-benchmark"


@dataclass
class GoldLine:
    """One adjudicated line: a pre-cropped image and its gold transcription."""

    line_id: str
    page_id: str
    line_index: int
    text: str
    image: Image.Image
    n_votes: int
    reads: list[str]


@dataclass
class GoldPage:
    """One full page: the whole scan plus its transcript in reading order."""

    page_id: str
    text: str
    image: Image.Image
    n_lines: int
    lines: list[dict]


def _load(config: str):
    from datasets import load_dataset

    return load_dataset(DATASET, config, split="test")


@lru_cache(maxsize=1)
def load_lines() -> list[GoldLine]:
    """The 225 gold line crops."""
    return [
        GoldLine(
            line_id=r["line_id"],
            page_id=r["page_id"],
            line_index=r["line_index"],
            text=r["text"],
            image=r["image"].convert("RGB"),
            n_votes=r["n_votes"],
            reads=list(r["reads"]),
        )
        for r in _load("lines")
    ]


@lru_cache(maxsize=1)
def load_pages() -> list[GoldPage]:
    """The 10 gold pages. `text` is already joined in reading order."""
    return [
        GoldPage(
            page_id=r["page_id"],
            text=r["page_text"],
            image=r["image"].convert("RGB"),
            n_lines=r["n_lines"],
            lines=list(r["lines"]),
        )
        for r in _load("pages")
    ]


def assert_not_trainable(split_name: str) -> None:
    """Guard against anyone wiring the benchmark into a training loop."""
    raise RuntimeError(
        f"{DATASET} has no trainable split (asked for {split_name!r}). "
        "It is the held-out benchmark -- training on it makes every number "
        "reported by this repo meaningless."
    )
