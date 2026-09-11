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
| v6 | + 11k real Arabic and English lines | 0.303 (0.273 with beam + LM) | superseded by v10 |
| v7 | TrOCR pretrained encoder, lr 1e-4 | stopped at epoch 2 | CTC collapsed to all-blank |
| v8 | same, lr 2e-4, head 5x | stopped at epoch 0 | collapse got worse |
| v9 | same, plus augmentation ramp | 0.466 | collapse fixed, but a clear transfer ceiling |
| v10 | back to v6, real ink scaled 11k to 56k lines | 0.250 (0.234 with beam + LM) | superseded by v11 |
| v11a | pretrain 30.2M on 692k lines, 551k of them real ink in 8 scripts | 0.327 after 2 epochs | pretraining stage |
| v11b | specialise v11a on Hebrew, 10 epochs | **0.261 (0.222 with beam + LM)** | **best; shipped** |

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

## The two-stage result is stranger than the headline

Stage B is *worse* than the single-stage v10 at greedy decoding, 0.261 against
0.250, and better only after decoding, 0.222 against 0.234. The difference is
what language-model fusion is worth to each model: 0.039 to the pretrained one
and 0.016 to the other.

So pretraining on half a million lines of foreign handwriting did not make the
model's best guess better. It made its probability distribution better, which a
beam search and a character language model can exploit and an argmax cannot.
Reading only the greedy number, as this project did while the run was going,
made the experiment look like a failure until the final evaluation.

The practical lesson: when a change alters how a model is trained, compare it
under the decoder you actually ship with, not the cheap one used for progress
logging.

# Decode-time experiments

Three levers that need no retraining: test-time augmentation, combining several
checkpoints, and a word-level prior in the beam. One of them is now shipped for
line mode, one for full-page mode, and one failed for a reason worth keeping.

## A dev set, so the benchmark stays a test set

Beam width and LM weight were chosen once and never revisited, which is the only
reason the benchmark numbers in this repo mean anything. Tuning three more knobs
by watching the benchmark would have quietly undone that.

Synthetic validation cannot serve as the tuning set, because it is too easy. The
shipped model reads it at CER 0.036 and reads real paper at 0.26. At 0.036 the
first-choice character is nearly always right and a language model has nothing
left to fix, so a knob tuned there is tuned for a regime the decoder never
operates in.

`scripts/make_devset.py` builds a harder one instead: the same
writer-independent validation split, put through the training augmentation at a
strength chosen so the model reads it at roughly its benchmark error rate.
Writers never cross splits upstream, so nothing in it was trained on. At
strength 2.2 it reproduces the benchmark's operating point closely:

| configuration | dev CER | benchmark CER |
|---|---|---|
| greedy | 0.250 | 0.261 |
| beam 12 | 0.238 | 0.254 |
| beam 12 + char LM | 0.223 | 0.222 |

Every hyperparameter below was chosen on that dev set. The benchmark was then
run once, over a list of configurations fixed in advance
(`decode/benchmark_configs.json`), and every entry in that list is reported
here including the ones that lost.

## Results on the benchmark

| configuration | CER median | no-drop | micro | word cov | scored |
|---|---|---|---|---|---|
| greedy | 0.261 | 0.261 | 0.306 | 0.245 | 225/225 |
| beam 12 | 0.254 | 0.254 | 0.300 | 0.250 | 225/225 |
| beam 12 + char LM (previous ship) | 0.222 | 0.224 | 0.274 | 0.376 | 222/225 |
| beam 12 + char LM + word prior 0.2 | 0.222 | 0.222 | 0.272 | 0.398 | 223/225 |
| **beam 12 + char LM + TTA 3** | **0.213** | **0.214** | **0.253** | **0.408** | 223/225 |
| v10 alone, beam 12 + char LM | 0.234 | 0.240 | 0.263 | 0.377 | 214/225 |
| v6 alone, beam 12 + char LM | 0.273 | 0.286 | 0.324 | 0.316 | 217/225 |
| ROVER over the three, + char LM | 0.219 | 0.222 | 0.265 | 0.385 | 220/225 |
| ROVER over the three, + word prior | 0.230 | 0.232 | 0.266 | 0.398 | 223/225 |
| frame-averaged ensemble (control) | 0.484 | 0.500 | 0.497 | 0.150 | 211/225 |

Full-page mode, which ranks on word coverage:

| configuration | word coverage | page CER |
|---|---|---|
| beam 12 + char LM (previous ship) | 0.333 | 0.400 |
| **beam 12 + char LM + word prior 0.2** | **0.349** | 0.413 |
| ROVER over the three | 0.317 | 0.391 |
| ROVER over the three, + word prior | 0.334 | 0.402 |

## Test-time augmentation: the idea was fine, the criterion was broken

Reading each line at several horizontal scales and keeping the most confident
reading *lost* accuracy at first: dev CER 0.2421 at three views and 0.2429 at
five, against 0.2232 for a single view.

The fault was in how a view was chosen. `greedy_confidence` averaged the
best-path probability over every frame, and blanks are the easiest prediction a
CTC model ever makes. A wider rendering of the same line carries more frames,
nearly all of the extra ones blank and scoring near 1.0, so the score rose with
width for reasons having nothing to do with legibility. The rule was picking the
widest view, not the clearest one.

Averaging only over frames that emit a character makes the score comparable
across scales. That took dev CER from 0.2421 to 0.2222, and on the benchmark
three views give **0.213 against 0.222**, a 4.0 percent relative gain, with
micro CER 0.253 against 0.274 and word coverage 0.408 against 0.376. It is now
the shipped line-mode configuration. It costs three forward passes per line.

## Ensembling by averaging frames: fails, and CTC is the reason

The plan was to average several checkpoints' frame distributions before
decoding, on the grounds that every HTR-VT preset downsamples width by 4 and so
emits the same number of frames for a given image. The frame *counts* do match.
What the frames contain does not.

An equal-weight average of the two best checkpoints scored dev CER **0.697**
against 0.250 for the better one alone, word coverage 0.010. On the benchmark,
as the pre-registered negative control, 0.484. Skewing the weighting 2:1 toward
the stronger model recovered to 0.288, which is to say it improved only as the
second model was weighted into irrelevance.

CTC constrains the order in which a model emits characters and says nothing
about when. The alignment is latent and each run settles into its own, typically
a sharp spike inside the character with blanks either side. Measured between the
two best checkpoints over 40 dev lines:

- of the frames where either model emits a character, only **19.9%** are frames
  where both do
- for characters the two models agree on, **96%** sit at different frames,
  median offset one frame

The average therefore lays one model's spike over the other's blank and flattens
both. No weighting fixes this, because the defect is not in the weights. The
code is kept in `hebocr/ensemble.py` with the finding in its docstring, since
"just average the logits" is the obvious next thought for anyone reading this.

This is also exactly why ROVER votes on finished strings: in string space the
alignment is computed rather than assumed.

## ROVER: works, and transfers far worse than the dev set promised

`hebocr/rover.py` aligns each system's output to a pivot hypothesis by edit
distance and takes a weighted vote per slot. Two properties worth noting. With
only two systems it is a no-op, because ties go to the pivot and there is never
a majority against it, so three is the minimum. And all three weightings tried
gave identical output, because in each of them the two non-pivot models can
outvote the pivot; the result is insensitive to the weights inside that range.

On the dev set it was the clear winner, 0.2232 to 0.2000, a 10.4 percent
relative gain that held in sign at every difficulty tested (+4.2% at strength
1.8, +10.4% at 2.2, +4.2% at 3.2).

On the benchmark it is worth 0.219 against 0.222: **1.7 percent, not 10.4**.

The gap is explained by what the dev set does to the weaker members. There the
three checkpoints score 0.2232, 0.2222 and 0.2279, which is a three-way tie and
ideal conditions for voting. On the benchmark they score 0.222, 0.234 and 0.273.
The dev set is synthetic, and the two older checkpoints were trained on a
mixture closer to it, so it flatters them and makes the ensemble look more
balanced than it is. Voting gains come from members of comparable strength, and
on real paper these three are not.

## A word-level prior: two length biases, pointing opposite ways

`lm.py` argues against word-level modelling for Hebrew: particles fuse onto the
following word, spelling varies legitimately, and a lexicon rejects correct
tokens it has never seen. That rules out a hard constraint, not a soft prior,
and there was a reason to want one. At CER 0.222 the model's word coverage was
only 0.376, which is the signature of output that is character-plausible and not
word-plausible.

The first formulation scored each completed word by `weight * log P(word)` with
an OOV floor, and degraded monotonically: dev CER 0.2232 at weight 0, 0.2376 at
0.3, 0.3846 at 1.2, with word coverage collapsing to 0.10 and blank lines
appearing where the character model alone produced none. Every word costs at
least the OOV floor, so a hypothesis with fewer words wins on arithmetic rather
than on evidence. The prior was not preferring real words, it was preferring
silence.

Rebasing on the floor fixes that: an unknown word scores 0, a known word earns a
bonus growing with frequency, so the term can only promote a real word over a
non-word of equal acoustic score. The corrected form has the opposite bias at
the top end, being unbounded above, and at weight 4.0 the decoder starts
inserting common words that are not there and CER passes 1.0. Weight 0.2 sits
where it only breaks ties.

| word weight | dev CER | dev word coverage |
|---|---|---|
| 0 | 0.2232 | 0.4264 |
| 0.2 | 0.2236 | 0.4460 |
| 0.5 | 0.2290 | 0.4482 |
| 1.0 | 0.2609 | 0.4232 |
| 2.0 | 0.4599 | 0.2886 |
| 4.0 | 1.6742 | 0.1072 |

On the benchmark at weight 0.2 it does not move the median CER at all, 0.222
either way, but improves every secondary line-mode number: no-drop 0.222 against
0.224, micro 0.272 against 0.274, word coverage 0.398 against 0.376, one fewer
blank line. Line mode ranks on the median, so none of that counts there.

Full-page mode ranks on word coverage, and there it does count: **0.349 against
0.333**, which moves full-page from fourth to third, ahead of claude-opus-5 at
0.335. The margin is 0.014 on a reconstructed metric, so it is thin and should
be read that way. Page CER is slightly worse, 0.413 against 0.400, which is the
expected shape of a prior that buys whole words at the cost of a few characters.

ROVER and the word prior do not compose: together they score 0.230, worse than
either alone.

## The lesson about the dev set itself

The dev set ranked the three *existing* configurations perfectly and mis-ranked
both *new* levers, in opposite directions:

| lever | dev | benchmark |
|---|---|---|
| ROVER | 10.4% better | 1.7% better |
| TTA 3 | 0.4% better | 4.0% better |

It over-sold the one that depends on the members being equally good, because
being synthetic it flatters models trained on synthetic data. It under-sold the
one that depends on character density varying, because its own stretch was drawn
from a distribution the model already trains against, while real hands vary more
and in ways nothing sampled.

Both errors come from the same root: the dev set is made of the training
distribution, so it cannot measure anything whose value depends on the gap
between that distribution and real paper. It is still worth having, since it is
the only way to set hyperparameters without burning the test set, but a decode
change that looks neutral on it deserves a benchmark run anyway. TTA was only
included in the pre-registered list because it was cheap to add, and it turned
out to be the largest win of the three.

# SAM, the optimizer HTR-VT uses and this project never did

`hebocr/optim.py` has implemented Sharpness-Aware Minimization since early on
and its docstring notes that HTR-VT uses it, but `use_sam` was False in every
run this project ever did, including both stages of the shipped model. The
argument for trying it is that flat minima are associated with better
out-of-distribution transfer, and the gap to real paper is all of the remaining
error here. The argument against is that it costs two forward/backward passes
per step, so it has to beat twice as many ordinary steps.

Both arms ran from the stage A pretrain on the same mixture, which for the first
time included real Hebrew ink. Starting from the shipped checkpoint instead
would have measured whether SAM helps a short fine-tune escape an already-sharp
basin, which is a different question. Raw logs in `experiments/sam_ab/`.

| | epochs | hours | best benchmark CER |
|---|---|---|---|
| no SAM | 8 | 2.90 | **0.2653** |
| SAM | 4 | 2.87 | 0.2951 |

At matched wall clock SAM is 11.2 percent behind. That was the comparison the
experiment was designed to make, and on its own it only says the two extra
passes are not worth twice the steps.

The per-epoch comparison is stronger and says more. At four epochs each, where
SAM has had double the compute, no SAM is 0.2857 against SAM's 0.3026. SAM is
behind per step as well as per second.

## The interesting part is which way each metric moved

At epoch 3, SAM has the **better** synthetic validation CER and the **worse**
benchmark CER:

| epoch 3 | val CER | benchmark CER |
|---|---|---|
| no SAM | 0.0588 | 0.2857 |
| SAM | 0.0577 | 0.3026 |

It fit the training distribution slightly better and transferred slightly worse,
which is the precise opposite of the reason to reach for it.

That is now the third time this project has met that shape. The TrOCR
pretrained encoder fit as well as the from-scratch model and transferred far
worse. The frame-averaged ensemble is a different failure but the same lesson
about an argument that sounds right in the abstract. And now an optimizer chosen
specifically for generalization generalizes worse while optimizing better.

The common thread is worth stating plainly, because it has now cost three
experiments: on this problem, anything that improves the fit to synthetic Hebrew
should be assumed neutral-to-harmful on real paper until measured on real paper.
Synthetic validation CER is not a proxy for the thing being optimized. It is a
proxy for the thing that is already solved.

## What this does not establish

Four epochs is short and sharpness-aware methods are sometimes a long-horizon
effect. `rho` was left at HTR-VT's 0.05 and never tuned, and SAM is known to be
sensitive to it. The mixture had foreign ink capped to a quarter of its usual
volume to fit the comparison into an afternoon. None of that is evidence SAM is
useless; it is evidence that SAM does not pay at this budget on this problem,
which was the only question being asked.

One observation that mattered more for what came next: the no-SAM arm was still
improving when its budget ran out, sitting at 0.2857 for four consecutive epochs
and then dropping to 0.2653 on the last one. It had not converged, which is the
main reason the long run gets twelve epochs rather than eight.
