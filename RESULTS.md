# Results

Checkpoint: the 30.2M model from `runs/final`, fine-tuned on all 943 lines of
real Hebrew cursive (`runs/ft_allpinkas`). Benchmark:
`ivrit-ai/hebrew-handwriting-ocr-benchmark`, 225 gold lines and 10 pages.
Published comparisons are the leaderboard's own figures, generated 2026-09-04.

The benchmark was never trained on and never used to select a checkpoint. The
base model selected on synthetic validation CER; the fine-tune selected on the
Pinkas corpus's own held-out partition. Neither ever saw the benchmark.

## Line mode

| model | CER median | lines scored |
|---|---|---|
| *human, 2nd read* | 0.000 | *225* |
| gemini-flash | 0.119 | 212 |
| **this model** | **0.175** | 224 |
| gemini-flash-lite | 0.280 | 225 |
| gpt-5.6-sol | 0.440 | 225 |
| gpt-5.6-terra | 0.585 | 225 |
| claude-sonnet-5 | 0.615 | 225 |
| claude-opus-5 | 0.692 | 225 |
| gpt-5.6-luna | 0.735 | 225 |
| claude-haiku-4-5 | 0.905 | 225 |

Second of nine. Decoding is beam 12, character 6-gram LM at 0.4, three-scale
TTA and a word prior at 0.2.

Every decode configuration, on the same checkpoint:

| configuration | CER median | no-drop | micro | word cov | scored |
|---|---|---|---|---|---|
| greedy | 0.239 | 0.240 | 0.261 | 0.332 | 223/225 |
| beam 12 + char LM | 0.202 | 0.206 | 0.233 | 0.446 | 222/225 |
| **beam 12 + char LM + TTA 3 + word prior** | **0.175** | **0.177** | **0.215** | **0.467** | 224/225 |

## Full-page mode

| model | word coverage | page CER |
|---|---|---|
| *human, 2nd read* | 0.890 | - |
| **this model** | **0.463** | **0.331** |
| gpt-5.6-sol | 0.462 | 0.400 |
| gemini-flash-lite | 0.453 | 0.353 |
| claude-opus-5 | 0.335 | 0.532 |
| gpt-5.6-terra | 0.270 | 0.599 |
| gemini-flash | 0.215 | 0.764 |
| claude-sonnet-5 | 0.210 | 0.619 |
| gpt-5.6-luna | 0.199 | 1.297 |
| claude-haiku-4-5 | 0.197 | 0.793 |

**Page CER of 0.331 is the best figure on the board**, 6 percent clear of
gemini-flash-lite's 0.353. That margin is real.

**Word coverage is a tie, not a win.** 0.4630 against gpt-5.6-sol's 0.4619 is a
margin of 0.0011 on a metric `hebocr/metrics.py` reconstructs rather than runs,
because `ivrit-ai/ocr-eval` is private. The honest claim is that the two are
indistinguishable at the top of that board. A variant tuned for coverage
(`--hebrew-ratio 0.7`) reaches 0.4712, a more comfortable 0.0093 lead, at the
cost of 0.012 line CER and 0.039 page CER; it is not what ships.

## Reproducibility

Beam search follows slightly different paths under bf16 autocast, so a re-run of
the shipped configuration returned 0.1771 where the harness recorded 0.1752.
Treat the line figure as 0.175 plus or minus 0.002. Greedy decoding is stable.

## What moved the number

| change | line CER |
|---|---|
| corrected augmentation, from measured properties of real images | 0.327 greedy |
| plus lines composed from real handwritten Hebrew glyphs | 0.306 |
| plus 11k real Arabic and English lines | 0.273 |
| plus 45k more real lines (Norwegian, French) | 0.234 |
| plus pretraining on 551k real lines across 8 scripts | 0.222 |
| plus multi-scale reading and a word prior at decode time | 0.188 |
| plus 943 lines of real Hebrew cursive | **0.175** |

Every entry is data except the last decode row. `EXPERIMENTS.md` records what
did not work, which is the longer list.
