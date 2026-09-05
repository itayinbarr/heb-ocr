# heb-ocr

Hebrew handwriting recognition (HTR) for the
[ivrit.ai Hebrew Handwriting OCR leaderboard](https://huggingface.co/spaces/ivrit-ai/hebrew-handwriting-ocr-leaderboard).

A character-level CTC line recognizer (CNN + ViT encoder, after HTR-VT) trained
entirely on synthetic Hebrew handwriting, plus a classical segment-then-recognize
pipeline for full-page mode. Everything runs on one 8 GB GPU.

## What the target actually is

The leaderboard is maintainer-run: there is no submission flow, and the
maintainers score models with their own harness. Numbers below are from
`data/{line,page}.csv` on the leaderboard Space, generated **2026-09-04**.

**Line mode** — one line crop in, text out, ranked by *median* CER:

| model | CER median | lines scored |
|---|---|---|
| *human, 2nd independent read* | *0.000* | *225* |
| gemini-flash | 0.119 | 212 (13 blank) |
| gemini-flash-lite | 0.280 | 225 |
| gpt-5.6-sol | 0.440 | 225 |
| gpt-5.6-terra | 0.585 | 225 |
| claude-sonnet-5 | 0.615 | 225 |
| claude-opus-5 | 0.692 | 225 |
| gpt-5.6-luna | 0.735 | 225 |
| claude-haiku-4-5 | 0.905 | 225 |

**Full-page mode** — whole page in, transcript out, ranked by order-independent
word coverage:

| model | word coverage | page CER |
|---|---|---|
| *human, 2nd independent read* | *0.890* | *-* |
| gpt-5.6-sol | 0.462 | 0.400 |
| gemini-flash-lite | 0.453 | 0.353 |
| claude-opus-5 | 0.335 | 0.532 |
| gpt-5.6-terra | 0.270 | 0.599 |
| gemini-flash | 0.215 | 0.764 |
| claude-sonnet-5 | 0.210 | 0.619 |
| gpt-5.6-luna | 0.199 | 1.297 |
| claude-haiku-4-5 | 0.197 | 0.793 |

Three things follow from this table, and they shaped every decision here.

**The benchmark is nowhere near saturated.** A second human transcriber scores a
median CER of **0.000** against the adjudicated gold and agrees exactly on 59.6%
of lines (`scripts/human_floor.py`). The best model is at 0.119. This is a gap in
capability, not a metric that has bottomed out.

**Full-page is the softer target.** Best-in-class coverage is 0.462 against a
human 0.890. Word coverage is explicitly order-independent, so segmentation and
reading-order mistakes cost nothing there — which makes a classical
segment-then-recognize pipeline a real contender.

**The line-mode leader's 0.119 is flattered by the metric.** Blank outputs are
dropped before the median is taken, so gemini-flash's headline number is a median
over 212 of 225 lines, and its *micro* CER on the same run is 2.19 — meaning some
outputs came back many times longer than the truth. `gemini-pro` is unranked
because 187 of its 225 outputs were blank. Because of this, `hebocr.metrics` also
reports `cer_median_nodrop`, which charges a blank the full 1.0. Quote both.

## Corrections to the original research plan

This repo was built from a research brief whose key premises did not survive
contact with the actual leaderboard:

- **`ivrit-ai/ivrit-handwriting-collection` is not a dataset or an eval harness.**
  It is a Flask photo-upload app for collecting submissions. The real harness,
  `ivrit-ai/ocr-eval`, is **private** (404), so the metric here is reconstructed
  from the leaderboard's own description of it.
- **The target number was wrong.** The brief cited Gemini Flash-Lite at ~0.12;
  Flash-Lite is actually **0.280**, and the 0.119 belongs to gemini-flash.
- **Pseudo-labeling the ~2,700 untranscribed pages is impossible.** They are not
  published. The benchmark is test-only by design: *"There is no train split, by
  design — it exists to be held out."*
- **`sivan22/hebrew-handwritten-dataset` is not line-level HHD.** It is
  single-character classification, 28 classes, ~4k glyphs.
- **The Blackwell/PyTorch setup step is moot** — torch 2.11+cu130 already reports
  `sm_120`.
- **The biggest lever already existed.** `cyttic/diffusionpen-hebrew-handwriting`
  (CC-BY-4.0) is 149,952 synthetic Hebrew handwriting lines in 491 writer styles,
  writer-independent splits. That is the "build a data engine" stage, prebuilt.

## The central constraint: no in-domain data

There is no public corpus of real modern Hebrew cursive with line-level
transcriptions. So **every training image here is synthetic**, and the whole
modelling problem is the domain gap to photographs of real paper. Two measured
gaps drive the augmentation:

- **Line length.** DiffusionPen tops out at 72 characters; 20% of benchmark lines
  are longer, up to 111. Fixed by joining lines (`concat_rtl`) — *right-to-left*,
  because Hebrew renders logical-first at the right edge. Reversing this would
  train on backwards text against forward labels and the loss would still fall.
- **Line density.** At a 64 px height, DiffusionPen renders ~34.5 px per
  character; the benchmark's real hands average ~20. The stretch augmentation is
  therefore deliberately lopsided toward compression (0.5–1.3x).

The rest of `Augment` manufactures what a phone camera does to paper: ruled
lines, page texture and shadow gradients, stroke-width variation, blur, sensor
noise and JPEG blocking.

## Design

**Recognizer** (`hebocr/models/htr_vt.py`, ~14M params at `base`) — a CNN
front-end feeding a ViT encoder with a CTC head, following HTR-VT
(Pattern Recognition, 2025). CTC over an autoregressive decoder because it cannot
enter a repetition loop, which is a documented failure of the VLMs on this exact
board. Character-level output means no tokenizer to mishandle RTL or final
letter forms. The CNN front-end is not optional: HTR-VT's ablation removes it and
IAM CER goes from 3.3% to 26.6%.

**Segmentation** (`hebocr/page/segment.py`) — deskew, then remove ruled lines,
then smear ink into lines and split components that swallowed two. Every
threshold is a multiple of the page's own measured line pitch, so nothing is
fitted to the ten test pages. Measured line recall: **96.0%** (216/225).

Order matters and cost two rewrites to get right: rules must be removed *after*
deskewing, and with a fan of *oriented* kernels — a photographed page's rules sag
and tilt a degree or two, and a flat kernel finds none of them. Leaving them in
welds every line on the page into one component (recall 30.7% → 48.0% → 96.0%).

**Batching** (`PixelBudgetSampler`) — batches are capped by
`lines x width-of-widest`, not by a line count. Widths span 150–2400 px, so a
fixed batch size has no stable memory cost; that is what OOMed the 8 GB card
first time.

## Results

Trained on synthetic data only, evaluated on the held-out benchmark. Full tables
and every prediction are in [`RESULTS.md`](RESULTS.md) and `runs/results.json`.

**Line mode — 3rd of 9**, behind only two Gemini tiers:

| model | CER median |
|---|---|
| *human, 2nd read* | *0.000* |
| gemini-flash | 0.119 |
| gemini-flash-lite | 0.280 |
| **this model** (beam + char LM) | **0.306** |
| gpt-5.6-sol | 0.440 |
| claude-opus-5 | 0.692 |

**Full-page mode — 5th of 9**: word coverage 0.235, page CER 0.514.

Decoding, all on the same weights: greedy 0.327 → beam 0.312 → beam + char LM
0.306. The LM's effect on *word coverage* is much larger than on CER (0.191 →
0.290), which is what you would expect from a model that fixes nearly-right
strings into exactly-right words.

The model never returns a blank, so its no-drop median equals its median.
That is not true of the current leader.

What moved the number, in order: fixing the augmentation's stroke weight,
vertical fill and neighbour bleed; mixing in lines built from real handwritten
glyphs; and giving those glyphs correct per-letter proportions. Model and
optimizer changes did nothing by comparison. The training curve across runs:

| epoch | 2 | 5 | 8 | 12 | 17 |
|---|---|---|---|---|---|
| DiffusionPen only | 0.595 | 0.475 | - | - | - |
| + real glyph lines | 0.562 | 0.414 | 0.382 | 0.360 | **0.327** |

## Usage

```bash
uv venv --system-site-packages .venv
uv pip install --python .venv/bin/python datasets rapidfuzz

python scripts/download_data.py          # accept the benchmark license on HF first
python -m hebocr.train --out runs/base --size base --epochs 30
python -m hebocr.evaluate runs/base/best.pt --out results.json

python scripts/eval_segmentation.py      # segmentation recall, no model needed
python scripts/human_floor.py            # the human noise floor
pytest -q                                # 81 tests
```

The benchmark is gated: accept the license at
[the dataset page](https://huggingface.co/datasets/ivrit-ai/hebrew-handwriting-ocr-benchmark)
with your own account first.

## Honest limitations

- **No in-domain validation set.** Checkpoints are selected on synthetic
  validation CER, never on the benchmark. The benchmark is scored during training
  but only ever *logged* — selecting on it would make every number here inflated.
  The risk this leaves: the checkpoint best on synthetic data need not be best on
  real paper.
- **The scorer is a reconstruction.** `ivrit-ai/ocr-eval` is private. Scoring is
  maintainer-run, so the official number for any model is whatever their harness
  says, not what this repo prints.
- **The segmenter was developed against the only real pages that exist**, which
  are the test pages. Its parameters are derived from image statistics rather
  than hand-tuned constants specifically to limit that contamination, but the
  exposure is not zero and should be read as a caveat on the full-page number.
- **Not attempted:** the VLM/QLoRA second track and ROVER ensembling (an 8 GB
  card cannot hold a 3B VLM), and character n-gram LM shallow fusion.

## Data and licensing

| dataset | role | license |
|---|---|---|
| `ivrit-ai/hebrew-handwriting-ocr-benchmark` | test only, never trained on | ivrit.ai License (gated) |
| `cyttic/diffusionpen-hebrew-handwriting` | all training data | CC-BY-4.0 |
