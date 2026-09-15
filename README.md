# heb-ocr

Hebrew handwriting recognition (HTR) for the
[ivrit.ai Hebrew Handwriting OCR leaderboard](https://huggingface.co/spaces/ivrit-ai/hebrew-handwriting-ocr-leaderboard).

A character-level CTC line recognizer (CNN + ViT encoder, after HTR-VT) trained
entirely on synthetic Hebrew handwriting, plus a classical segment-then-recognize
pipeline for full-page mode. Everything runs on one 8 GB GPU.

**Released weights: [`itayinbar/Mishkefet-v1`](https://huggingface.co/itayinbar/Mishkefet-v1)**,
30.2M parameters. Second of nine on the leaderboard's line mode at 0.175 median
CER; on full-page mode it posts the best page CER on the board (0.331) and ties
gpt-5.6-sol at the top on word coverage (0.463 against 0.462). It runs on a CPU,
needs no API key, and costs nothing per line.

## What the target actually is

The leaderboard is maintainer-run: there is no submission flow, and the
maintainers score models with their own harness. Numbers below are from
`data/{line,page}.csv` on the leaderboard Space, generated **2026-09-04**.

**Line mode**, one line crop in, text out, ranked by *median* CER:

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

**Full-page mode**, whole page in, transcript out, ranked by order-independent
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
reading-order mistakes cost nothing there, which makes a classical
segment-then-recognize pipeline a real contender.

**The line-mode leader's 0.119 is flattered by the metric.** Blank outputs are
dropped before the median is taken, so gemini-flash's headline number is a median
over 212 of 225 lines, and its *micro* CER on the same run is 2.19, meaning some
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
  design, it exists to be held out."*
- **`sivan22/hebrew-handwritten-dataset` is not line-level HHD.** It is
  single-character classification, 28 classes, ~4k glyphs.
- **The Blackwell/PyTorch setup step is moot**, torch 2.11+cu130 already reports
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
  are longer, up to 111. Fixed by joining lines (`concat_rtl`), *right-to-left*,
  because Hebrew renders logical-first at the right edge. Reversing this would
  train on backwards text against forward labels and the loss would still fall.
- **Line density.** At a 64 px height, DiffusionPen renders ~34.5 px per
  character; the benchmark's real hands average ~20. The stretch augmentation is
  therefore deliberately lopsided toward compression (0.5–1.3x).

The rest of `Augment` manufactures what a phone camera does to paper: ruled
lines, page texture and shadow gradients, stroke-width variation, blur, sensor
noise and JPEG blocking.

## Design

**Recognizer** (`hebocr/models/htr_vt.py`, 13.8M params at `base` and 30.2M at
`large`, which is the size that ships), a CNN front-end feeding a ViT encoder
with a CTC head, following HTR-VT
(Pattern Recognition, 2025). CTC over an autoregressive decoder because it cannot
enter a repetition loop, which is a documented failure of the VLMs on this exact
board. Character-level output means no tokenizer to mishandle RTL or final
letter forms. The CNN front-end is not optional: HTR-VT's ablation removes it and
IAM CER goes from 3.3% to 26.6%.

**Segmentation** (`hebocr/page/segment.py`), deskew, then remove ruled lines,
then smear ink into lines and split components that swallowed two. Every
threshold is a multiple of the page's own measured line pitch, so nothing is
fitted to the ten test pages. Measured line recall: **96.0%** (216/225).

Order matters and cost two rewrites to get right: rules must be removed *after*
deskewing, and with a fan of *oriented* kernels, a photographed page's rules sag
and tilt a degree or two, and a flat kernel finds none of them. Leaving them in
welds every line on the page into one component (recall 30.7% → 48.0% → 96.0%).

**Batching** (`PixelBudgetSampler`), batches are capped by
`lines x width-of-widest`, not by a line count. Widths span 150–2400 px, so a
fixed batch size has no stable memory cost; that is what OOMed the 8 GB card
first time.

## Results

Trained on synthetic Hebrew plus real handwriting in eight other scripts,
evaluated on the held-out benchmark. Full tables and every prediction are in
[`RESULTS.md`](RESULTS.md); every run including the failures is in
[`EXPERIMENTS.md`](EXPERIMENTS.md).

**Line mode, 2nd of 9:**

| model | CER median | no-drop | lines scored |
|---|---|---|---|
| *human, 2nd read* | *0.000* | *0.000* | *225* |
| gemini-flash | 0.119 | - | 212 |
| **this model** | **0.175** | **0.177** | 224 |
| gemini-flash-lite | 0.280 | 0.280 | 225 |
| gpt-5.6-sol | 0.440 | - | 225 |
| claude-opus-5 | 0.692 | - | 225 |

Ahead of gemini-flash-lite by 0.105 on **both** readings, so the ranking does
not depend on the leaderboard's blank-dropping rule.

**Full-page mode**: page CER 0.331, which is the best figure on that board, 6
percent clear of gemini-flash-lite's 0.353. Word coverage 0.463 against
gpt-5.6-sol's 0.462 is a tie rather than a win: 0.0011 on a metric this repo
reconstructs is inside its own error, and the maintainers' harness decides it.

### How it was trained

Two stages, because no public corpus of real modern Hebrew cursive exists:

1. **Pretrain on ink.** 692k lines, of which 551k are real handwriting in
   Arabic, English, Norwegian, French, German, Latin and Spanish. Hebrew is
   only a fifth of this mixture. It reaches 0.327 on the Hebrew benchmark
   anyway, which is the entire result of an earlier Hebrew-specialised run.
2. **Specialise on Hebrew.** Initialize from those weights, then train on a
   Hebrew-heavy mixture. The encoder transfers whole; only the CTC head is
   rebuilt, because the alphabet differs.

### What actually moved the number

Every large gain came from making the training ink more like real ink.

| change | line CER |
|---|---|
| corrected augmentation, from measured properties of real images | 0.327 greedy |
| plus lines composed from real handwritten Hebrew glyphs | 0.306 |
| plus 11k real Arabic and English lines | 0.273 |
| plus 45k more real lines (Norwegian, French) | 0.234 |
| plus pretraining on 551k real lines across 8 scripts | 0.222 |
| plus multi-scale reading and a word prior at decode time | 0.188 |
| plus 943 lines of real Hebrew cursive | **0.175** |

**Real handwriting in languages the model cannot read improves Hebrew.** That
is the finding this project rests on, confirmed five times at increasing scale.
What transfers across scripts is not language but ink: stroke texture, pen
width, and how paper takes a pen.

The two-stage result also arrives in an unexpected way. Its greedy decoding is
*worse* than the single-stage model's (0.261 against 0.250), and it wins only
after decoding, where language-model fusion is worth 0.039 to it against 0.016
to the single-stage model. Pretraining on ink produced better calibrated
probabilities rather than better best-guesses, so the advantage lives in the
decoder rather than the argmax.

Two things were tried and lost, and are written up in `EXPERIMENTS.md`:
starting from TrOCR's IAM-pretrained encoder (0.466 against 0.250), and scaling
the model to 46.6M parameters, which was abandoned because on this GPU it cost
more in wall-clock than it returned.

## Usage

### Read Hebrew handwriting with the released model

Nothing here needs training. Pull the weights and the character LM from the
model repository and run them:

```bash
pip install -e .
huggingface-cli download itayinbar/Mishkefet-v1 \
    mishkefet-v1.pt hebrew_char6.pkl hebrew_words.pkl --local-dir .
```

```python
from PIL import Image
from hebocr.lm import CharNGramLM
from hebocr.recognize import Recognizer
from hebocr.wordlm import WordUnigramLM

# Line mode: multi-scale reading is worth 4 percent and costs 3 forward passes.
lines = Recognizer("mishkefet-v1.pt", lm=CharNGramLM.load("hebrew_char6.pkl"),
                   lm_weight=0.4, beam_width=12, tta=3)
print(lines.read_tta([Image.open("line.jpg")])[0])

# Full-page mode ranks on word coverage, where the word prior is the better
# trade and TTA is not. Different metric, different configuration.
pages = Recognizer("mishkefet-v1.pt", lm=CharNGramLM.load("hebrew_char6.pkl"),
                   lm_weight=0.4, beam_width=12,
                   word_lm=WordUnigramLM.load("hebrew_words.pkl"), word_weight=0.2)
transcript, records = pages.read_page(Image.open("page.jpg"))
```

### Reproduce the training

```bash
uv venv --system-site-packages .venv
uv pip install --python .venv/bin/python datasets rapidfuzz

python scripts/download_data.py          # accept the benchmark license on HF first

# Stage A, pretrain on ink: 692k lines, 551k of them real, 8 scripts besides Hebrew.
python -m hebocr.train --out runs/stage_a --size large --epochs 2 \
    --real-ink all --lr 3e-4 --glyph-lines 25000 --ema 0.9995

# Stage B, specialise on Hebrew from those weights. This is the shipped model.
scripts/run_stage_b.sh

python scripts/make_results.py runs/stage_b/best.pt --beam-width 12 --lm-weight 0.4
```

Decode-time settings are tuned against a held-out dev set, never the benchmark:

```bash
python scripts/build_wordlm.py --corpus hebrew.txt   # the word prior
python scripts/make_devset.py --strength 2.2         # the tuning set
python scripts/tune_decode.py --stage baseline       # grid-search a lever
python scripts/eval_configs.py --configs decode/benchmark_configs.json --pages
```

Stage A is about 6 hours on an RTX 5070 and stage B about 13.5, so the pipeline
scripts are built to be detached and to survive a dropped session.

```bash
python scripts/eval_segmentation.py      # segmentation recall, no model needed
python scripts/human_floor.py            # the human noise floor
pytest -q                                # 140 tests
```

The benchmark is gated: accept the license at
[the dataset page](https://huggingface.co/datasets/ivrit-ai/hebrew-handwriting-ocr-benchmark)
with your own account first.

## Honest limitations

- **No in-domain validation set.** Checkpoints are selected on synthetic
  validation CER, never on the benchmark. The benchmark is scored during training
  but only ever *logged*, selecting on it would make every number here inflated.
  The risk this leaves: the checkpoint best on synthetic data need not be best on
  real paper.
- **The scorer is a reconstruction.** `ivrit-ai/ocr-eval` is private. Scoring is
  maintainer-run, so the official number for any model is whatever their harness
  says, not what this repo prints.
- **The segmenter was developed against the only real pages that exist**, which
  are the test pages. Its parameters are derived from image statistics rather
  than hand-tuned constants specifically to limit that contamination, but the
  exposure is not zero and should be read as a caveat on the full-page number.
- **Not attempted:** the VLM/QLoRA second track, because an 8 GB card cannot hold
  a 3B VLM. ROVER ensembling *was* attempted and is worth only 1.7 percent here;
  `EXPERIMENTS.md` explains why the dev set promised six times that.

## Data and licensing

Two corpora of real Hebrew handwriting with line-level transcriptions are
published and both are used: 10,219 lines in total. That is the entire public
supply, and only 943 of it is cursive, the script the benchmark is written in.
The ivrit.ai benchmark itself is test-only by design and was never trained on.

| dataset | role | license |
|---|---|---|
| [`ivrit-ai/hebrew-handwriting-ocr-benchmark`](https://huggingface.co/datasets/ivrit-ai/hebrew-handwriting-ocr-benchmark) | test only, never trained on | ivrit.ai License (gated) |
| [Pinkas](https://zenodo.org/records/3569694) | 943 lines of real Hebrew cursive, the only such corpus published | CC-BY-4.0 |
| [BiblIA](https://zenodo.org/records/5167263) | 9,276 lines of real medieval Hebrew square script | CC-BY-NC-SA-4.0 |
| [`cyttic/diffusionpen-hebrew-handwriting`](https://huggingface.co/datasets/cyttic/diffusionpen-hebrew-handwriting) | 116k synthetic Hebrew lines, 491 writer styles | CC-BY-4.0 |
| [`sivan22/hebrew-handwritten-dataset`](https://huggingface.co/datasets/sivan22/hebrew-handwritten-dataset) | 25k lines composed from real handwritten Hebrew glyphs | CC-BY-3.0 |
| [`johnlockejrr/KHATT_v1.0_dataset`](https://huggingface.co/datasets/johnlockejrr/KHATT_v1.0_dataset) | 4,672 real Arabic lines | MIT (as published) |
| [`Teklia/IAM-line`](https://huggingface.co/datasets/Teklia/IAM-line) | 6,482 real English lines | MIT (as published) |
| [`Teklia/NorHand-v3-line`](https://huggingface.co/datasets/Teklia/NorHand-v3-line) | 30,000 real Norwegian lines | MIT |
| [`Teklia/Belfort-line`](https://huggingface.co/datasets/Teklia/Belfort-line) | real French lines | MIT |
| Teklia NorHand-v2, HOME-Alcar, NewsEye, Himanis, RIMES, Esposalles, POPP | the rest of the 551k real pretraining lines | MIT |
| Hebrew Wikipedia | 30M characters for the character 6-gram LM | CC-BY-SA |

**Licensing caution.** The KHATT and IAM mirrors above are labelled MIT on the
Hub, but the original KHATT and IAM-DB corpora carry their own terms, and IAM-DB
has historically been restricted to non-commercial research use. Anyone
intending commercial use should verify those terms upstream rather than relying
on the mirrors' labels.

**This repository.** The code is MIT (see [`LICENSE`](LICENSE)). The trained
weights are released separately under **CC-BY-NC-SA-4.0** at
[`itayinbar/Mishkefet-v1`](https://huggingface.co/itayinbar/Mishkefet-v1).

That is a change from the CC-BY-4.0 of earlier releases, and it is forced by the
data. BiblIA is CC-BY-NC-SA-4.0, so a model trained on it inherits NonCommercial
and ShareAlike. Training with `--real-hebrew pinkas` keeps the permissive terms
at the cost of 9,276 of the 10,219 real Hebrew lines.

## Citation

```bibtex
@misc{mishkefet2026,
  title  = {Mishkefet-v1: a compact CTC recognizer for modern Hebrew handwriting},
  author = {Itay Inbar},
  year   = {2026},
  howpublished = {\url{https://huggingface.co/itayinbar/Mishkefet-v1}},
}
```
