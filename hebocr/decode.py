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
