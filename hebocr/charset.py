"""Character vocabulary for the CTC head.

Index 0 is reserved for the CTC blank. Everything else is a literal character
of normalized text, so decoding is just a table lookup -- there is no tokenizer
to mishandle RTL, which is a large part of why a character CTC model is the
right choice for Hebrew.
"""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .normalize import normalize

BLANK = 0

# Hebrew letters including the five final forms, which are distinct glyphs and
# must never be folded onto their medial counterparts.
HEBREW_LETTERS = "אבגדהוזחטיכךלמםנןסעפףצץקרשת"

# Punctuation that actually carries meaning in these transcripts. Geresh and
# gershayim mark abbreviations and acronyms and are extremely common in Hebrew.
BASE_PUNCT = " '\"-.,:;!?()[]{}/\\|*+=%&#@_<>~^$"
DIGITS = "0123456789"

# The gold transcripts contain Latin characters (acronyms, loanwords, the odd
# English word mid-sentence). Without these, 12 of the 225 gold lines are
# unrepresentable and carry a guaranteed error floor.
LATIN = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass
class Charset:
    """Ordered character vocabulary. `chars[0]` is a placeholder for the blank."""

    chars: list[str]

    def __post_init__(self) -> None:
        if self.chars[0] != "":
            raise ValueError("index 0 is reserved for the CTC blank")
        self._to_idx = {c: i for i, c in enumerate(self.chars)}

    def __len__(self) -> int:
        return len(self.chars)

    @property
    def n_classes(self) -> int:
        """Vocabulary size including the blank -- the CTC head's output width."""
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        """Text -> label indices, silently dropping out-of-vocabulary chars.

        Dropping is deliberate: a rare stray character in a synthetic caption
        should cost us that character, not crash a training run.
        """
        return [self._to_idx[c] for c in normalize(text) if c in self._to_idx]

    def decode(self, indices) -> str:
        """Label indices -> text. Assumes CTC collapsing already happened."""
        return "".join(self.chars[i] for i in indices if i != BLANK)

    def coverage(self, text: str) -> float:
        """Fraction of `text` this charset can represent. 1.0 means lossless."""
        norm = normalize(text)
        if not norm:
            return 1.0
        return sum(c in self._to_idx for c in norm) / len(norm)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"chars": self.chars}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "Charset":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(chars=data["chars"])

    @classmethod
    def default(cls) -> "Charset":
        """The fixed vocabulary: Hebrew letters, digits, common punctuation."""
        seen = dict.fromkeys(HEBREW_LETTERS + DIGITS + BASE_PUNCT + LATIN)
        return cls(chars=[""] + list(seen))

    @classmethod
    def from_texts(cls, texts, min_count: int = 1) -> "Charset":
        """Build from a corpus, keeping characters seen at least `min_count`.

        The threshold filters transcription noise -- a single Latin character
        that slipped into one synthetic caption should not earn a class.
        """
        counts = Counter()
        for t in texts:
            counts.update(normalize(t))
        keep = sorted(c for c, n in counts.items() if n >= min_count)
        return cls(chars=[""] + keep)
