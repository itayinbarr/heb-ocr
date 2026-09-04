"""A character n-gram language model, for shallow fusion during decoding.

Character-level, not word-level, and that is a deliberate choice for Hebrew.
Prepositions and conjunctions fuse onto the following word (ב, ל, כ, ה, ו, ש, מ),
so word boundaries are unreliable and a word-level lexicon rejects perfectly
correct tokens it has never seen. Spelling also varies legitimately (ktiv
male/haser), which multiplies the surface forms of the same word. A character
model sidesteps both: it learns which letter sequences are plausible Hebrew
without ever committing to a vocabulary.

Scoring uses stupid backoff, which is not normalized but is the standard choice
for fusion, where only the relative ranking of candidate characters matters.
"""

import json
import math
import pickle
from collections import defaultdict
from pathlib import Path

from .normalize import normalize

BACKOFF = 0.4  # the usual stupid-backoff discount


class CharNGramLM:
    """Counts of character n-grams, queried as log P(next | context)."""

    def __init__(self, order: int = 6, backoff: float = BACKOFF):
        self.order = order
        self.backoff = backoff
        self.counts: list[dict] = [defaultdict(int) for _ in range(order + 1)]
        self.total = 0
        self._log_backoff = math.log(backoff)

    def train(self, texts, progress_every: int = 0) -> "CharNGramLM":
        """Accumulate n-gram counts over a corpus of already-normalized lines."""
        for i, text in enumerate(texts):
            line = normalize(text)
            if not line:
                continue
            padded = "\x02" + line  # start marker, so the first letter has context
            self.total += len(line)
            for n in range(1, self.order + 1):
                counts = self.counts[n]
                for j in range(len(padded) - n + 1):
                    counts[padded[j : j + n]] += 1
            if progress_every and (i + 1) % progress_every == 0:
                print(f"  {i+1} lines, {self.total} chars", flush=True)
        return self

    def logprob(self, context: str, char: str) -> float:
        """log P(char | context), backing off to shorter contexts as needed."""
        context = context[-(self.order - 1) :]
        penalty = 0.0
        for length in range(len(context), -1, -1):
            prefix = context[len(context) - length :] if length else ""
            numerator = self.counts[length + 1].get(prefix + char, 0)
            if numerator:
                denominator = self.counts[length].get(prefix, 0) if length else self.total
                if denominator:
                    return penalty + math.log(numerator / denominator)
            penalty += self._log_backoff
        # Unseen character: charge it the unigram floor rather than -inf, so a
        # rare-but-real letter is discouraged, not forbidden.
        return penalty + math.log(1.0 / max(self.total, 1))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "order": self.order,
                    "backoff": self.backoff,
                    "total": self.total,
                    "counts": [dict(c) for c in self.counts],
                },
                fh, protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: str | Path) -> "CharNGramLM":
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        lm = cls(order=data["order"], backoff=data["backoff"])
        lm.total = data["total"]
        lm.counts = [defaultdict(int, c) for c in data["counts"]]
        return lm

    def prune(self, min_count: int = 2) -> "CharNGramLM":
        """Drop rare high-order n-grams. Mostly a memory measure."""
        for n in range(3, self.order + 1):
            self.counts[n] = defaultdict(
                int, {k: v for k, v in self.counts[n].items() if v >= min_count}
            )
        return self

    def stats(self) -> dict:
        return {
            "order": self.order,
            "total_chars": self.total,
            "ngrams": {n: len(self.counts[n]) for n in range(1, self.order + 1)},
        }
