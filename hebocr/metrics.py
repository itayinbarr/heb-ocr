"""Scoring that mirrors the ivrit.ai Hebrew Handwriting OCR leaderboard.

Two modes, matching the two tables on the leaderboard page:

line mode
    Each gold line crop -> one predicted string. Ranked by the *median* line
    CER. Per the leaderboard: "Outputs with no text are dropped." That rule is
    load-bearing -- gemini-flash's headline 0.119 is a median over 212 of 225
    lines, with a micro CER of 2.19 on the same run -- so `line_report` also
    returns `cer_median_nodrop`, which charges a blank the full CER of 1.0.
    Quote both.

page mode
    Whole page image -> one transcript. Two numbers:
      word coverage -- order-independent multiset fraction of gold tokens
                       present anywhere in the output (the board sorts on this)
      page CER      -- gold lines joined in reading order, edit-distanced
                       against the whole output, micro-averaged over pages
"""

from collections import Counter
from dataclasses import dataclass, asdict
from statistics import median

from rapidfuzz.distance import Levenshtein

from .normalize import normalize, tokens


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate: edit distance to `reference`, over its length.

    An empty reference is degenerate: any output against it is either perfect
    (both empty) or unboundedly wrong, so we report 0.0 / 1.0 rather than
    dividing by zero.
    """
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return Levenshtein.distance(ref, hyp) / len(ref)


def word_coverage(reference: str, hypothesis: str) -> float:
    """Fraction of gold tokens that appear anywhere in the output.

    Multiset (so a word needed twice must appear twice) and order-independent,
    so reading-order and segmentation differences do not count against it.
    """
    ref_counts = Counter(tokens(reference))
    if not ref_counts:
        return 0.0
    hyp_counts = Counter(tokens(hypothesis))
    hit = sum(min(n, hyp_counts[tok]) for tok, n in ref_counts.items())
    return hit / sum(ref_counts.values())


def _quartiles(values: list[float]) -> tuple[float, float]:
    """25th and 75th percentile, linear interpolation, matching numpy's default."""
    if not values:
        return float("nan"), float("nan")
    ordered = sorted(values)
    def pct(p: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        pos = p * (len(ordered) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)
    return pct(0.25), pct(0.75)


@dataclass
class LineReport:
    """Line-mode scores. `cer_median` is the leaderboard's ranking number."""

    n_total: int
    n_scored: int
    n_blank: int
    cer_median: float
    cer_q1: float
    cer_q3: float
    cer_micro: float
    cer_median_nodrop: float
    wcov_macro: float
    wcov_micro: float

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"CER median {self.cer_median:.3f} "
            f"({self.cer_q1:.2f}-{self.cer_q3:.2f})  "
            f"micro {self.cer_micro:.3f}  "
            f"no-drop median {self.cer_median_nodrop:.3f}  "
            f"wcov {self.wcov_micro:.3f}  "
            f"scored {self.n_scored}/{self.n_total} ({self.n_blank} blank)"
        )


def line_report(references: list[str], hypotheses: list[str]) -> LineReport:
    """Score line mode over paired gold/predicted strings."""
    if len(references) != len(hypotheses):
        raise ValueError(
            f"{len(references)} references but {len(hypotheses)} hypotheses"
        )

    kept_cers, all_cers, covs = [], [], []
    dist_sum = ref_len_sum = 0
    hit_sum = tok_sum = 0
    blanks = 0

    for ref, hyp in zip(references, hypotheses):
        ref_n, hyp_n = normalize(ref), normalize(hyp)
        score = cer(ref_n, hyp_n)
        all_cers.append(score)

        if not hyp_n:
            # Dropped from the ranked median, but still charged in `nodrop`.
            blanks += 1
        else:
            kept_cers.append(score)
            dist_sum += Levenshtein.distance(ref_n, hyp_n)
            ref_len_sum += len(ref_n)

            ref_counts = Counter(tokens(ref_n))
            hyp_counts = Counter(tokens(hyp_n))
            hit = sum(min(n, hyp_counts[t]) for t, n in ref_counts.items())
            total = sum(ref_counts.values())
            hit_sum += hit
            tok_sum += total
            covs.append(hit / total if total else 0.0)

    q1, q3 = _quartiles(kept_cers)
    return LineReport(
        n_total=len(references),
        n_scored=len(kept_cers),
        n_blank=blanks,
        cer_median=median(kept_cers) if kept_cers else float("nan"),
        cer_q1=q1,
        cer_q3=q3,
        cer_micro=dist_sum / ref_len_sum if ref_len_sum else float("nan"),
        cer_median_nodrop=median(all_cers) if all_cers else float("nan"),
        wcov_macro=sum(covs) / len(covs) if covs else 0.0,
        wcov_micro=hit_sum / tok_sum if tok_sum else 0.0,
    )


@dataclass
class PageReport:
    """Full-page scores. The board sorts on `wcov_micro`."""

    n_pages: int
    n_scored: int
    n_blank: int
    wcov_macro: float
    wcov_micro: float
    cer_micro: float
    cer_median: float

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"word coverage {self.wcov_micro:.3f}  "
            f"page CER {self.cer_micro:.3f}  "
            f"scored {self.n_scored}/{self.n_pages} ({self.n_blank} blank)"
        )


def page_report(references: list[str], hypotheses: list[str]) -> PageReport:
    """Score full-page mode.

    `references` are gold page transcripts -- lines already joined in reading
    order (the benchmark ships this as the `page_text` field).
    """
    if len(references) != len(hypotheses):
        raise ValueError(
            f"{len(references)} references but {len(hypotheses)} hypotheses"
        )

    covs, cers = [], []
    dist_sum = ref_len_sum = 0
    hit_sum = tok_sum = 0
    blanks = 0

    for ref, hyp in zip(references, hypotheses):
        ref_n, hyp_n = normalize(ref), normalize(hyp)
        if not hyp_n:
            blanks += 1
            continue

        ref_counts = Counter(tokens(ref_n))
        hyp_counts = Counter(tokens(hyp_n))
        hit = sum(min(n, hyp_counts[t]) for t, n in ref_counts.items())
        total = sum(ref_counts.values())
        hit_sum += hit
        tok_sum += total
        covs.append(hit / total if total else 0.0)

        dist_sum += Levenshtein.distance(ref_n, hyp_n)
        ref_len_sum += len(ref_n)
        cers.append(cer(ref_n, hyp_n))

    return PageReport(
        n_pages=len(references),
        n_scored=len(covs),
        n_blank=blanks,
        wcov_macro=sum(covs) / len(covs) if covs else 0.0,
        wcov_micro=hit_sum / tok_sum if tok_sum else 0.0,
        cer_micro=dist_sum / ref_len_sum if ref_len_sum else float("nan"),
        cer_median=median(cers) if cers else float("nan"),
    )
