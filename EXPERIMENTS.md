# Experiment log

Every training run, including the ones that failed. Benchmark CER is greedy
decoding on the held-out ivrit.ai line benchmark, logged during training and
never used to select a checkpoint.

| run | change | final benchmark CER | outcome |
|---|---|---|---|
| v1 | DiffusionPen, first augmentation | abandoned at epoch 0 | augmentation produced blank images with full labels |
| v2 | corrected augmentation | 0.475 @ epoch 5 | baseline |
| v3 | + glyph lines, wrong letter proportions | neutral | yod drawn at twice its real size |
| v4 | + glyph lines, measured proportions | 0.327 (0.306 with beam + LM) | shipped |
| v5 | + EMA, perspective, show-through | superseded | folded into v6 |
| v6 | + 11k real Arabic and English lines | **0.303 (0.273 with beam + LM)** | **shipped, published** |
| v7 | TrOCR pretrained encoder, lr 1e-4 | stopped at epoch 2 | CTC collapsed to all-blank |
| v8 | same, lr 2e-4, head 5x | stopped at epoch 0 | collapse got worse |
| v9 | same, plus augmentation ramp | 0.466 | collapse fixed, but a clear transfer ceiling |

## The pretrained encoder did not work, and the reason is interesting

The reasoning going in was sound. Every gain this project made came from making
training ink look more like real ink, and TrOCR's encoder is a ViT pretrained on
IAM, real human handwriting. Rather than keep manufacturing that prior, start
from one that has it.

It lost to the from-scratch model by a wide margin: 0.466 against 0.303.

Getting there took three runs. The first two collapsed into CTC's degenerate
all-blank solution, 163 then 220 of 225 lines empty, and raising the learning
rate made it worse. An overfit test settled the cause: on clean crops the same
architecture reaches CER 0.000 with zero blanks, so nothing was structurally
broken. The augmentation was the problem. Tuned for a from-scratch CNN, it hands
a pretrained ViT with frozen patch embeddings inputs unlike anything in its
pretraining. Ramping the strength from a third to full over three epochs fixed
the collapse completely, 6 blanks instead of 220.

But the fixed run then plateaued. Across its last six epochs the benchmark sat
between 0.466 and 0.500 while synthetic validation kept improving from 0.100 to
0.050. **The model fit the training distribution as well as v6 did and
transferred worse.** That is the exact opposite of what a real-handwriting
prior was supposed to buy, and it is the useful finding.

The most likely explanation is a mismatch the "it saw real handwriting"
argument glosses over. That ViT was pretrained on 384x384 images at patch 16.
Our inputs are 64 px line crops upscaled horizontally, so a patch covers a
quarter of the line height and an eighth of a character. Whatever the encoder
learned about handwriting, it learned at a spatial scale we never present to it.
Freezing the patch embeddings, which the run did to fit the card and the clock,
removes the one component that could have adapted to that.

Worth trying if the path is revisited: unfreeze the patch embeddings while
keeping the transformer blocks frozen, which is the opposite of the usual recipe
and follows directly from the diagnosis above.

## What actually moved the number

Ranked by measured effect, largest first:

1. **Correcting the augmentation** to match measured properties of the real
   images: stroke weight (5 to 8 px synthetic against 1 to 2 real), vertical
   fill, and bleed from neighbouring lines.
2. **Lines composed from real handwritten glyphs**, once given each letter's
   measured proportions.
3. **Real handwriting in other scripts.** 11,154 Arabic and English lines,
   whose labels mean nothing to a Hebrew reader, improved Hebrew line CER by
   11 percent relative. What transfers is ink, not language.
4. **Decoding**: beam search plus a character 6-gram LM, 0.303 to 0.273.

Architecture and optimizer changes contributed almost nothing by comparison.
The single largest remaining lever is not code: it is the roughly 300
transcribed pages the ivrit.ai maintainers hold and have not published.
