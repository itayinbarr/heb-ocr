# Submitting to the ivrit.ai leaderboard

The leaderboard is **maintainer-run**: its README states "Maintainer-run — no
submission flow; regenerated and redeployed when a provider ships a newer
model." There is no upload endpoint and no automated scoring. So getting listed
means contacting the maintainers and making it trivial for them to score this
model with *their* harness.

## Who to contact

Via the [Hugging Face org](https://huggingface.co/ivrit-ai) or a GitHub issue on
[`ivrit-ai/ocr-eval`](https://github.com/ivrit-ai/ocr-eval) (private at the time
of writing, so the org page or the ivrit.ai site is the reliable route):

- Yanir Marmor — HF `yanirmr`
- Kinneret Misgav — HF `Kinneret`
- Yoad Snapir — HF `yoad`
- Yair Lifshitz — HF `benderrodriguez`

## What to send

Everything they need is in this repo. The two commands that reproduce every
number:

```bash
python -m hebocr.evaluate <checkpoint> --mode both        # both leaderboard modes
python scripts/eval_segmentation.py                       # full-page line recall
```

The model is a single self-contained checkpoint: weights, charset, and the
config needed to rebuild the architecture. It runs on CPU. There is no API key,
no external service and no per-line cost, which is a meaningful difference from
every model currently on the board.

## What to say plainly

Three things are worth stating up front, because a maintainer will find them
anyway and it is better they come from us:

1. **`hebocr/metrics.py` is a reconstruction of their metric**, not their code —
   `ocr-eval` is private. They should score with their own harness, and their
   number is the official one. If it disagrees with ours, theirs is right.
2. **The benchmark was never trained on and never used for checkpoint
   selection.** Selection used synthetic validation CER only; the benchmark was
   logged during training purely for visibility. This matters because the
   benchmark is explicitly test-only by design.
3. **The full-page segmenter was necessarily developed against the only real
   pages that exist, which are the test pages.** Its thresholds derive from each
   image's own measured line pitch rather than hand-tuned constants,
   specifically to limit that exposure — but the exposure is not zero, and the
   full-page number should be read with that caveat.

## Worth offering them

- The **human floor** measured from their own data (`scripts/human_floor.py`):
  a second independent volunteer read scores a median CER of 0.000 against the
  adjudicated gold, with exact agreement on 59.6% of lines, and word coverage
  0.890. That is useful context for their leaderboard page: it shows the
  benchmark is far from saturated and gives readers a ceiling to compare against.
- The observation that the **line-mode median is gameable by blanking**. Blank
  outputs are dropped before the median is taken, so gemini-flash's 0.119 is a
  median over 212 of 225 lines while its micro CER is 2.19, and gemini-pro is
  unranked only because 187 of its 225 outputs were blank. Reporting a
  no-drop median alongside (as `hebocr.metrics` does) would make the board
  harder to game.
- Training data provenance, in case they want to publish a baseline:
  `cyttic/diffusionpen-hebrew-handwriting` (CC-BY-4.0) is the entire training
  set; no ivrit.ai data was used for training.
