# Results

Checkpoint: `runs/stage_b/best.pt`. Benchmark: `ivrit-ai/hebrew-handwriting-ocr-benchmark`
(225 gold lines, 10 pages). Published comparisons are the leaderboard's own
figures, generated 2026-09-04.

The benchmark was never trained on and never used to select a checkpoint;
selection used synthetic validation CER only.

## Line mode

| model | CER median | lines scored |
|---|---|---|
| *human, 2nd read* | 0.000 | *225* |
| gemini-flash | 0.119 | 212 |
| **this model (beam 12 + char LM)** | 0.222 | 222 |
| **this model (beam 12)** | 0.254 | 225 |
| **this model (greedy)** | 0.261 | 225 |
| gemini-flash-lite | 0.280 | 225 |
| gpt-5.6-sol | 0.440 | 225 |
| gpt-5.6-terra | 0.585 | 225 |
| claude-sonnet-5 | 0.615 | 225 |
| claude-opus-5 | 0.692 | 225 |
| gpt-5.6-luna | 0.735 | 225 |
| claude-haiku-4-5 | 0.905 | 225 |

Best configuration: **this model (beam 12 + char LM)** at 0.222 median CER.

Full statistics, including the no-drop median that charges blank outputs the
full 1.0 rather than dropping them from the median as the leaderboard does:

| configuration | CER median | no-drop median | micro CER | word cov | scored |
|---|---|---|---|---|---|
| this model (greedy) | 0.261 | 0.261 | 0.306 | 0.245 | 225/225 |
| this model (beam 12) | 0.254 | 0.254 | 0.300 | 0.250 | 225/225 |
| this model (beam 12 + char LM) | 0.222 | 0.224 | 0.274 | 0.376 | 222/225 |

## Full-page mode

| model | word coverage | page CER |
|---|---|---|
| *human, 2nd read* | 0.890 | - |
| gpt-5.6-sol | 0.462 | 0.400 |
| gemini-flash-lite | 0.453 | 0.353 |
| claude-opus-5 | 0.335 | 0.532 |
| **this model (beam 12 + char LM)** | 0.333 | 0.400 |
| gpt-5.6-terra | 0.270 | 0.599 |
| **this model (beam 12)** | 0.217 | 0.434 |
| gemini-flash | 0.215 | 0.764 |
| claude-sonnet-5 | 0.210 | 0.619 |
| gpt-5.6-luna | 0.199 | 1.297 |
| **this model (greedy)** | 0.199 | 0.433 |
| claude-haiku-4-5 | 0.197 | 0.793 |

Best configuration: **this model (beam 12 + char LM)** at 0.333 word coverage.

Segmentation recall (fraction of gold lines the segmenter finds, at 50% area
coverage) is **96.0%**, 216 of 225, measured by `scripts/eval_segmentation.py`.
That is the ceiling any recognizer can reach through this pipeline.

## The blank-dropping caveat, applied to ourselves

The leaderboard drops blank outputs before taking the median. When that rule
flatters someone else it is worth pointing out; when it flatters us it is worth
pointing out twice. Both readings of the best configuration:

| metric | this model | nearest published model |
|---|---|---|
| leaderboard rule (blanks dropped) | 0.222 over 222 lines | gemini-flash-lite 0.280 over 225 |
| no-drop (a blank scores 1.0) | 0.224 | 0.280 |

This model is ahead of gemini-flash-lite on both readings, so the ranking does not depend on the blank-dropping rule.

## Reading these numbers

The official scores are whatever the maintainers' harness produces: the
leaderboard is maintainer-run, and `ivrit-ai/ocr-eval` is private, so
`hebocr/metrics.py` is a reconstruction from the leaderboard's description of
the metric.

The line-mode median is a forgiving statistic. Blank outputs are dropped before
it is taken, which is how gemini-flash posts 0.119 while its micro CER on the
same run is 2.19. The no-drop column above is the honest counterpart.

## Decode-time study, 2026-09-11

Three levers that need no retraining were measured against a held-out dev set
and then scored once on the benchmark over a configuration list fixed in
advance. `EXPERIMENTS.md` has the method, the failures and the reasoning;
`decode/benchmark_results.json` has every number.

**Line mode is now beam 12 + char LM + TTA 3**, at 0.213 median CER against
0.222 for the previous configuration, a 4.0 percent relative gain. It costs
three forward passes per line instead of one.

| configuration | CER median | no-drop | micro | word cov | scored |
|---|---|---|---|---|---|
| **beam 12 + char LM + TTA 3** | **0.213** | **0.214** | **0.253** | **0.408** | 223/225 |
| ROVER over three checkpoints | 0.219 | 0.222 | 0.265 | 0.385 | 220/225 |
| beam 12 + char LM (previous) | 0.222 | 0.224 | 0.274 | 0.376 | 222/225 |
| beam 12 + char LM + word prior | 0.222 | 0.222 | 0.272 | 0.398 | 223/225 |
| beam 12 | 0.254 | 0.254 | 0.300 | 0.250 | 225/225 |
| greedy | 0.261 | 0.261 | 0.306 | 0.245 | 225/225 |

The line board is unchanged by this: second of nine either way, with
gemini-flash at 0.119 ahead and gemini-flash-lite at 0.280 behind.

**Full-page mode is now beam 12 + char LM + word prior 0.2**, at 0.349 word
coverage against 0.333. The word prior is worth nothing to the line median and
something to word coverage, which is what the page board ranks on.

| configuration | word coverage | page CER |
|---|---|---|
| **beam 12 + char LM + word prior 0.2** | **0.349** | 0.413 |
| beam 12 + char LM (previous) | 0.333 | 0.400 |
| ROVER, + word prior | 0.334 | 0.402 |
| ROVER over three checkpoints | 0.317 | 0.391 |

That moves full-page from fourth to third, ahead of claude-opus-5 at 0.335. The
margin is 0.014 on a metric this repo reconstructs rather than runs, so it is
thin and the maintainers' harness is the one that decides it.

Updated page standing:

| model | word coverage | page CER |
|---|---|---|
| *human, 2nd read* | 0.890 | - |
| gpt-5.6-sol | 0.462 | 0.400 |
| gemini-flash-lite | 0.453 | 0.353 |
| **this model (beam 12 + char LM + word prior)** | **0.349** | 0.413 |
| claude-opus-5 | 0.335 | 0.532 |
| gpt-5.6-terra | 0.270 | 0.599 |
| gemini-flash | 0.215 | 0.764 |
| claude-sonnet-5 | 0.210 | 0.619 |
| gpt-5.6-luna | 0.199 | 1.297 |
| claude-haiku-4-5 | 0.197 | 0.793 |
