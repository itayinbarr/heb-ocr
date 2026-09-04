"""Text canonicalization applied before scoring.

The leaderboard states only: "Punctuation and whitespace are canonicalized
before scoring (Hebrew text itself is untouched)." The harness that implements
that sentence (github.com/ivrit-ai/ocr-eval) is private, so what follows is a
reconstruction from that sentence plus the variation visible in the gold file's
own `reads` field, where two volunteers transcribing the same line disagree on
punctuation but not on letters -- e.g. the pair 'בס"ד' / "בס''ד".

Two rules follow from "Hebrew text itself is untouched":
  - niqqud and te'amim are NOT stripped
  - final letter forms (ך ם ן ף ץ) are NOT folded onto their medial forms

Both would be lossy edits to Hebrew, and the benchmark is unvocalized anyway.
"""

import re
import unicodedata

# Bidi control characters. These carry no textual content but do count as
# characters in an edit distance, so an RTL-aware model that emits them would be
# penalized for nothing. Stripping them is the single most important step here.
_BIDI = dict.fromkeys(
    map(
        ord,
        "‎‏"  # LRM RLM
        "‪‫‬‭‮"  # LRE RLE PDF LRO RLO
        "⁦⁧⁨⁩"  # LRI RLI FSI PDI
        "​‌‍"  # ZWSP ZWNJ ZWJ
        "﻿",  # BOM
    )
)

# Punctuation variants -> one canonical spelling. Hebrew geresh/gershayim are
# routinely typed as ASCII quotes (and vice versa); the gold's own reads show
# volunteers splitting both ways on the same line.
_PUNCT = {
    "׳": "'",  # HEBREW PUNCTUATION GERESH
    "״": '"',  # HEBREW PUNCTUATION GERSHAYIM
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "′": "'", "´": "'", "`": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "″": '"', "«": '"', "»": '"',
    "־": "-",  # HEBREW PUNCTUATION MAQAF
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": "-", "―": "-", "−": "-",
    "׃": ":",  # HEBREW PUNCTUATION SOF PASUQ
    "׀": "|",  # HEBREW PUNCTUATION PASEQ
    "…": "...",  # HORIZONTAL ELLIPSIS
    " ": " ", " ": " ", " ": " ", " ": " ", " ": " ",
    "　": " ", "\t": " ",
}

_PUNCT_RE = re.compile("|".join(map(re.escape, sorted(_PUNCT, key=len, reverse=True))))
_WS_RE = re.compile(r"\s+")
# Two apostrophes typed for one gershayim -- collapse so 'בס''ד' == 'בס"ד'.
_DOUBLE_APOS_RE = re.compile(r"''")


def normalize(text: str | None) -> str:
    """Canonicalize `text` for scoring. Returns "" for None/blank input."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_BIDI)
    text = _PUNCT_RE.sub(lambda m: _PUNCT[m.group()], text)
    text = _DOUBLE_APOS_RE.sub('"', text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def is_blank(text: str | None) -> bool:
    """True when a model returned nothing usable. The leaderboard drops these."""
    return not normalize(text)


def tokens(text: str | None) -> list[str]:
    """Whitespace tokens of the normalized text, for word-coverage scoring."""
    n = normalize(text)
    return n.split(" ") if n else []
