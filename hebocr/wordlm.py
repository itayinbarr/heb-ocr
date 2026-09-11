"""A word-level unigram model, fused at word boundaries during beam search.

`lm.py` argues against word-level modelling for Hebrew, and the objection is
correct as far as it goes: prepositions and conjunctions fuse onto the following
word (ב, ל, כ, ה, ו, ש, מ), spelling varies legitimately (ktiv male/haser), and
a lexicon that rejects unseen tokens throws away correct output.

That argument rules out a *constraint*. It does not rule out a *prior*. This
model never forbids anything. An unknown word is charged a fixed penalty and
stays in the beam; a known word gets a bonus proportional to its frequency. The
decoder can still emit any string the acoustic model likes, it just has to
prefer it more strongly than the alternative that happens to be a real word.

Two things make the prior fit Hebrew rather than fight it:

- The counts come from running text, so the fused forms (ובית, שבבית) are in the
  vocabulary as themselves. Nothing has to be analysed morphologically.
- A word that is unknown but becomes known after stripping one prefix letter is
  charged a discount rather than the full OOV penalty, which covers the fused
  forms that a finite corpus happened to miss.

The motivation is a specific failure in the shipped model: at CER 0.222 its word
coverage is only 0.376. It is getting most characters and few whole words, which
is the signature of output that is character-plausible and not word-plausible.
That is exactly the gap a word prior addresses.
"""

import math
import pickle
import re
from collections import Counter
from pathlib import Path

from .normalize import normalize

# The single-letter particles that attach to the front of a Hebrew word. Used
# only for backoff, never to segment: the vocabulary holds fused forms already.
PREFIXES = "בהוכלמש"

WORD = re.compile(r"[א-ת']+")


class WordUnigramLM:
    """log P(word), with a fixed floor for words the corpus never saw."""

    def __init__(self, oov_logprob: float = -14.0, prefix_discount: float = -2.5):
        self.counts: Counter = Counter()
        self.total = 0
        self.oov_logprob = oov_logprob
        self.prefix_discount = prefix_discount
        self._cache: dict[str, float] = {}

    def train(self, texts, progress_every: int = 0) -> "WordUnigramLM":
        for i, text in enumerate(texts):
            words = WORD.findall(normalize(text))
            self.counts.update(words)
            self.total += len(words)
            if progress_every and (i + 1) % progress_every == 0:
                print(f"  {i+1} lines, {self.total} words, {len(self.counts)} types", flush=True)
        return self

    def prune(self, min_count: int = 2) -> "WordUnigramLM":
        """Drop hapax legomena. Most are typos, transliterations or markup."""
        self.counts = Counter({w: c for w, c in self.counts.items() if c >= min_count})
        return self

    def logprob(self, word: str) -> float:
        """log P(word), never -inf.

        A word is scored directly if known. If not, one leading particle is
        stripped and the remainder scored at a discount, which recovers the
        fused forms a finite corpus missed. Failing both, the OOV floor.
        """
        if not word:
            return 0.0
        hit = self._cache.get(word)
        if hit is not None:
            return hit

        count = self.counts.get(word, 0)
        if count:
            score = math.log(count / self.total)
        elif len(word) > 2 and word[0] in PREFIXES and self.counts.get(word[1:], 0):
            score = math.log(self.counts[word[1:]] / self.total) + self.prefix_discount
        else:
            score = self.oov_logprob

        self._cache[word] = score
        return score

    def bonus(self, word: str) -> float:
        """How much better than unknown this word is. Never negative.

        `logprob` alone cannot be fused into a beam that is also scoring
        characters. Every word costs at least the OOV floor, so charging it
        makes a hypothesis with fewer words win on arithmetic rather than on
        evidence, and the decoder starts preferring short output and then empty
        output. Measured, not theorised: at word weight 0.3 the plain-logprob
        form lost 0.014 CER and produced blank lines the character model alone
        never produced.

        Rebasing on the floor removes the bias. An unknown word scores 0 and is
        neither helped nor punished; a known word earns a bonus that grows with
        its frequency. The prior can now only ever promote a real word over a
        non-word of equal acoustic score, which is the whole intent.
        """
        return max(0.0, self.logprob(word) - self.oov_logprob)

    def known(self, word: str) -> bool:
        return bool(self.counts.get(word, 0)) or (
            len(word) > 2 and word[0] in PREFIXES and bool(self.counts.get(word[1:], 0))
        )

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "counts": dict(self.counts),
                    "total": self.total,
                    "oov_logprob": self.oov_logprob,
                    "prefix_discount": self.prefix_discount,
                },
                fh, protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: str | Path) -> "WordUnigramLM":
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        lm = cls(oov_logprob=data["oov_logprob"], prefix_discount=data["prefix_discount"])
        lm.counts = Counter(data["counts"])
        lm.total = data["total"]
        return lm

    def stats(self) -> dict:
        return {"types": len(self.counts), "tokens": self.total}
