# Results

Checkpoint: `runs/base/best.pt`. Benchmark: `ivrit-ai/hebrew-handwriting-ocr-benchmark`
(225 gold lines, 10 pages). Published comparisons are the leaderboard's own
figures, generated 2026-09-04.

The benchmark was never trained on and never used to select a checkpoint;
selection used synthetic validation CER only.

## Line mode

| model | CER median | lines scored |
|---|---|---|
| *human, 2nd read* | 0.000 | *225* |
| gemini-flash | 0.119 | 212 |
| **this model (beam 12 + char LM)** | 0.273 | 217 |
| gemini-flash-lite | 0.280 | 225 |
| **this model (beam 12)** | 0.293 | 225 |
| **this model (greedy)** | 0.303 | 225 |
| gpt-5.6-sol | 0.440 | 225 |
| gpt-5.6-terra | 0.585 | 225 |
| claude-sonnet-5 | 0.615 | 225 |
| claude-opus-5 | 0.692 | 225 |
| gpt-5.6-luna | 0.735 | 225 |
| claude-haiku-4-5 | 0.905 | 225 |

Best configuration: **this model (beam 12 + char LM)** at 0.273 median CER.

Full statistics, including the no-drop median that charges blank outputs the
full 1.0 rather than dropping them from the median as the leaderboard does:

| configuration | CER median | no-drop median | micro CER | word cov | scored |
|---|---|---|---|---|---|
| this model (greedy) | 0.303 | 0.303 | 0.344 | 0.210 | 225/225 |
| this model (beam 12) | 0.293 | 0.293 | 0.336 | 0.218 | 225/225 |
| this model (beam 12 + char LM) | 0.273 | 0.286 | 0.324 | 0.316 | 217/225 |

## Full-page mode

| model | word coverage | page CER |
|---|---|---|
| *human, 2nd read* | 0.890 | - |
| gpt-5.6-sol | 0.462 | 0.400 |
| gemini-flash-lite | 0.453 | 0.353 |
| claude-opus-5 | 0.335 | 0.532 |
| gpt-5.6-terra | 0.270 | 0.599 |
| **this model (beam 12 + char LM)** | 0.229 | 0.529 |
| gemini-flash | 0.215 | 0.764 |
| claude-sonnet-5 | 0.210 | 0.619 |
| gpt-5.6-luna | 0.199 | 1.297 |
| claude-haiku-4-5 | 0.197 | 0.793 |
| **this model (beam 12)** | 0.132 | 0.529 |
| **this model (greedy)** | 0.131 | 0.536 |

Best configuration: **this model (beam 12 + char LM)** at 0.229 word coverage.

Segmentation recall (fraction of gold lines the segmenter finds, at 50% area
coverage) is **96.0%** — 216 of 225 — measured by `scripts/eval_segmentation.py`.
That is the ceiling any recognizer can reach through this pipeline.

## The blank-dropping caveat, applied to ourselves

The leaderboard drops blank outputs before taking the median. When that rule
flatters someone else it is worth pointing out; when it flatters us it is worth
pointing out twice. Both readings of the best configuration:

| metric | this model | nearest published model |
|---|---|---|
| leaderboard rule (blanks dropped) | 0.273 over 217 lines | gemini-flash-lite 0.280 over 225 |
| no-drop (a blank scores 1.0) | 0.286 | 0.280 |

So this model wins on the board's own metric and loses on the stricter one. gemini-flash-lite returned text for all 225 lines, so its no-drop median is also 0.280; ours is inflated by 8 dropped line(s). Treat the two as a tie until the blanks are recovered.

## Reading these numbers

The official scores are whatever the maintainers' harness produces: the
leaderboard is maintainer-run, and `ivrit-ai/ocr-eval` is private, so
`hebocr/metrics.py` is a reconstruction from the leaderboard's description of
the metric.

The line-mode median is a forgiving statistic. Blank outputs are dropped before
it is taken, which is how gemini-flash posts 0.119 while its micro CER on the
same run is 2.19. The no-drop column above is the honest counterpart.
