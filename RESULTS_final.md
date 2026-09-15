# Results

Checkpoint: `runs/final/best.pt`. Benchmark: `ivrit-ai/hebrew-handwriting-ocr-benchmark`
(225 gold lines, 10 pages). Published comparisons are the leaderboard's own
figures, generated 2026-09-04.

The benchmark was never trained on and never used to select a checkpoint;
selection used synthetic validation CER only.

## Line mode

| model | CER median | lines scored |
|---|---|---|
| *human, 2nd read* | 0.000 | *225* |
| gemini-flash | 0.119 | 212 |
| **this model (beam 12 + char LM)** | 0.200 | 221 |
| **this model (beam 12)** | 0.245 | 222 |
| **this model (greedy)** | 0.250 | 222 |
| gemini-flash-lite | 0.280 | 225 |
| gpt-5.6-sol | 0.440 | 225 |
| gpt-5.6-terra | 0.585 | 225 |
| claude-sonnet-5 | 0.615 | 225 |
| claude-opus-5 | 0.692 | 225 |
| gpt-5.6-luna | 0.735 | 225 |
| claude-haiku-4-5 | 0.905 | 225 |

Best configuration: **this model (beam 12 + char LM)** at 0.200 median CER.

Full statistics, including the no-drop median that charges blank outputs the
full 1.0 rather than dropping them from the median as the leaderboard does:

| configuration | CER median | no-drop median | micro CER | word cov | scored |
|---|---|---|---|---|---|
| this model (greedy) | 0.250 | 0.250 | 0.279 | 0.295 | 222/225 |
| this model (beam 12) | 0.245 | 0.250 | 0.280 | 0.297 | 222/225 |
| this model (beam 12 + char LM) | 0.200 | 0.200 | 0.242 | 0.426 | 221/225 |

## Full-page mode

| model | word coverage | page CER |
|---|---|---|
| *human, 2nd read* | 0.890 | - |
| gpt-5.6-sol | 0.462 | 0.400 |
| gemini-flash-lite | 0.453 | 0.353 |
| **this model (beam 12 + char LM)** | 0.422 | 0.333 |
| claude-opus-5 | 0.335 | 0.532 |
| **this model (beam 12)** | 0.306 | 0.376 |
| **this model (greedy)** | 0.300 | 0.363 |
| gpt-5.6-terra | 0.270 | 0.599 |
| gemini-flash | 0.215 | 0.764 |
| claude-sonnet-5 | 0.210 | 0.619 |
| gpt-5.6-luna | 0.199 | 1.297 |
| claude-haiku-4-5 | 0.197 | 0.793 |

Best configuration: **this model (beam 12 + char LM)** at 0.422 word coverage.

Segmentation recall (fraction of gold lines the segmenter finds, at 50% area
coverage) is **96.0%**, 216 of 225, measured by `scripts/eval_segmentation.py`.
That is the ceiling any recognizer can reach through this pipeline.

## The blank-dropping caveat, applied to ourselves

The leaderboard drops blank outputs before taking the median. When that rule
flatters someone else it is worth pointing out; when it flatters us it is worth
pointing out twice. Both readings of the best configuration:

| metric | this model | nearest published model |
|---|---|---|
| leaderboard rule (blanks dropped) | 0.200 over 221 lines | gemini-flash-lite 0.280 over 225 |
| no-drop (a blank scores 1.0) | 0.200 | 0.280 |

This model is ahead of gemini-flash-lite on both readings, so the ranking does not depend on the blank-dropping rule.

## Reading these numbers

The official scores are whatever the maintainers' harness produces: the
leaderboard is maintainer-run, and `ivrit-ai/ocr-eval` is private, so
`hebocr/metrics.py` is a reconstruction from the leaderboard's description of
the metric.

The line-mode median is a forgiving statistic. Blank outputs are dropped before
it is taken, which is how gemini-flash posts 0.119 while its micro CER on the
same run is 2.19. The no-drop column above is the honest counterpart.
