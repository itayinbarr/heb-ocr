"""Turning CTC log-probabilities into text."""

import torch

from .charset import BLANK, Charset


def greedy_decode(logprobs: torch.Tensor, lengths: torch.Tensor, charset: Charset) -> list[str]:
    """Best-path decode: argmax per step, collapse repeats, drop blanks.

    `lengths` is the valid time-step count per batch item, so padding added to
    square off a batch cannot contribute characters.
    """
    best = logprobs.argmax(dim=-1).cpu()
    lengths = lengths.cpu()
    out = []
    for row, valid in zip(best, lengths):
        row = row[: int(valid)]
        collapsed, previous = [], None
        for idx in row.tolist():
            if idx != previous and idx != BLANK:
                collapsed.append(idx)
            previous = idx
        out.append(charset.decode(collapsed))
    return out


def greedy_confidence(logprobs: torch.Tensor, lengths: torch.Tensor) -> list[float]:
    """Mean per-step probability of the best path -- a usable rejection signal."""
    probs = logprobs.exp().max(dim=-1).values.cpu()
    return [
        float(probs[i, : int(n)].mean()) if int(n) > 0 else 0.0
        for i, n in enumerate(lengths.cpu())
    ]


def _logsumexp(a: float, b: float) -> float:
    """Numerically safe log(exp(a) + exp(b)) for the two-term case."""
    import math

    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    hi, lo = (a, b) if a > b else (b, a)
    return hi + math.log1p(math.exp(lo - hi))


NEG_INF = float("-inf")


def beam_decode(
    logprobs: torch.Tensor,
    lengths: torch.Tensor,
    charset: Charset,
    beam_width: int = 16,
    top_k: int = 8,
    lm=None,
    lm_weight: float = 0.4,
    length_bonus: float = 0.6,
) -> list[str]:
    """CTC prefix beam search, optionally fused with a character n-gram LM.

    Greedy decoding commits to the best character at every frame independently,
    which on unclear handwriting throws away the fact that some letter sequences
    are Hebrew and others are not. Beam search keeps several hypotheses alive and
    lets an LM break the ties.

    `lm_weight` scales the LM's contribution and `length_bonus` offsets the LM's
    inherent bias toward short strings (every extra character costs probability).
    Both are decode-time knobs and need no retraining. With `lm=None` this is a
    plain prefix beam search.

    Only the `top_k` most likely characters per frame are considered: the tail of
    a 122-way softmax contributes nothing but cost.
    """
    probs = logprobs.cpu()
    lengths = lengths.cpu()
    results = []

    for row, valid in zip(probs, lengths):
        row = row[: int(valid)]
        # prefix -> [log p(ending in blank), log p(ending in a real char)]
        beams: dict[str, list[float]] = {"": [0.0, NEG_INF]}
        lm_cache: dict[str, float] = {"": 0.0}

        candidate_indices = row.topk(min(top_k, row.shape[-1]), dim=-1).indices

        for t in range(row.shape[0]):
            frame = row[t]
            next_beams: dict[str, list[float]] = {}

            for prefix, (p_blank, p_nonblank) in beams.items():
                p_total = _logsumexp(p_blank, p_nonblank)

                for index in candidate_indices[t].tolist():
                    p = float(frame[index])

                    if index == BLANK:
                        entry = next_beams.setdefault(prefix, [NEG_INF, NEG_INF])
                        entry[0] = _logsumexp(entry[0], p_total + p)
                        continue

                    char = charset.chars[index]
                    if prefix and char == prefix[-1]:
                        # Repeat of the last character: stays the same prefix
                        # unless a blank separated the two occurrences.
                        entry = next_beams.setdefault(prefix, [NEG_INF, NEG_INF])
                        entry[1] = _logsumexp(entry[1], p_nonblank + p)
                        extended_from = p_blank
                    else:
                        extended_from = p_total

                    if extended_from == NEG_INF:
                        continue

                    extended = prefix + char
                    if extended not in lm_cache:
                        lm_cache[extended] = (
                            lm_cache[prefix] + lm.logprob(prefix, char) if lm is not None else 0.0
                        )
                    bonus = (lm_weight * lm.logprob(prefix, char) + length_bonus) if lm is not None else 0.0

                    entry = next_beams.setdefault(extended, [NEG_INF, NEG_INF])
                    entry[1] = _logsumexp(entry[1], extended_from + p + bonus)

            beams = dict(
                sorted(
                    next_beams.items(),
                    key=lambda kv: _logsumexp(kv[1][0], kv[1][1]),
                    reverse=True,
                )[:beam_width]
            )
            if not beams:
                beams = {"": [0.0, NEG_INF]}

        best = max(beams.items(), key=lambda kv: _logsumexp(kv[1][0], kv[1][1]))[0]
        results.append(best)

    return results
