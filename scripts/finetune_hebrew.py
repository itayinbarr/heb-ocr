"""Fine-tune the finished model on real Hebrew, with rehearsal.

The long run mixes 10,219 real Hebrew lines into a pool of 1.27M, so they are
0.8 percent of every epoch. That is the price of scaling the synthetic corpus,
and it leaves the scarcest and most in-domain data this project has buried. This
is the controlled way to find out what it is worth.

Three things about the design are not optional.

**Selection never touches the benchmark, and cannot use synthetic validation
either.** Fine-tuning toward real Hebrew moves away from synthetic Hebrew, so
synthetic validation CER gets worse as the fine-tune does its job, and selecting
on it would pick zero fine-tuning every time. Both corpora publish their own
train/test partitions (BiblIA 183/19 pages, Pinkas two file lists), so selection
runs on held-out *real Hebrew* that was never trained on. The benchmark is
touched once, at the end, on the already-chosen checkpoint.

**Rehearsal, not a pure Hebrew fine-tune.** The real Hebrew is not the target
domain. BiblIA is medieval square script written by professional scribes and
Pinkas is early-modern quill cursive, while the benchmark is ballpoint on ruled
paper. They are closer to the target than synthetic ink, which is the whole
reason to use them, but training on them alone would drag 30M parameters toward
historical letterforms. Each step draws `--hebrew-ratio` of its batch from real
Hebrew and the rest from the mixture the model already knows.

**A low learning rate and a short schedule.** 10,219 lines against 30M
parameters overfits in minutes, which is also why this is cheap enough to sweep
rather than run once.

A caveat that has to be stated rather than buried. The long run was launched
with `--real-hebrew all`, before the published partitions were wired in, so the
base model has already seen the 1,219 lines used here for selection. They are
held out from the *fine-tune* but not from the *base*, which means the held-out
CER printed below is not an unbiased estimate of generalization: it is optimistic
for every checkpoint, including step 0.

What it is still good for is ranking, because every candidate carries the same
contamination and the fine-tune itself never trains on those lines. So use these
numbers to choose a checkpoint and not to make a claim. The claim comes from the
single benchmark pass at the end, which is genuinely untouched. Future training
runs should pass `--real-hebrew-split train` so this does not recur.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from hebocr.charset import Charset
from hebocr.data.hebrew_ink import HebrewInkSource, load_hebrew_ink
from hebocr.data.synth import LineDataset, collate, load_diffusionpen
from hebocr.metrics import line_report
from hebocr.models.htr_vt import build_model


def evaluate(model, charset, rows, device, batch_size=12):
    """Greedy CER on held-out real Hebrew. Never the benchmark."""
    from hebocr.data.transforms import preprocess
    from hebocr.decode import greedy_decode

    model.eval()
    preds, golds = [], []
    order = sorted(range(len(rows)), key=lambda i: rows[i]["image"].width)
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            chunk = order[start : start + batch_size]
            arrays = [torch.from_numpy(preprocess(rows[i]["image"])) for i in chunk]
            widths = torch.tensor([a.shape[-1] for a in arrays], dtype=torch.long)
            padded = torch.zeros(len(arrays), 1, arrays[0].shape[-2], int(widths.max()))
            for j, a in enumerate(arrays):
                padded[j, :, :, : a.shape[-1]] = a
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logprobs = model(padded.to(device))
            out = greedy_decode(logprobs.float(), model.output_lengths(widths), charset)
            preds.extend(out)
            golds.extend(rows[i]["text"] for i in chunk)
    model.train()
    return line_report(golds, preds)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", help="the finished model to fine-tune")
    ap.add_argument("--out", default="runs/finetune")
    ap.add_argument("--hebrew-ratio", type=float, default=0.3,
                    help="fraction of each batch drawn from real Hebrew")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--batch-lines", type=int, default=16,
                    help="upper bound on lines per step; the pixel budget usually "
                         "binds first")
    ap.add_argument("--pixel-budget", type=int, default=22000,
                    help="lines x width-of-widest, the same budget training uses")
    ap.add_argument("--corpora", default="all")
    ap.add_argument("--train-split", default="train", choices=["train", "all"],
                    help="'train' holds out the corpus test partition for selection. "
                         "'all' trains on every line, which is more data and leaves "
                         "nothing to select on, so it requires --fixed-steps")
    ap.add_argument("--fixed-steps", type=int, default=0,
                    help="stop at this step and keep that checkpoint, selecting "
                         "nothing. Required with --train-split all: the settings "
                         "must be carried over from a run that did have a held-out "
                         "set, the way you refit on train+val after choosing "
                         "hyperparameters on val")
    ap.add_argument("--heldout-limit", type=int, default=0,
                    help="score only this many held-out lines; 0 means all of them. "
                         "For smoke tests, not for a real selection")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    names = ("pinkas", "biblia") if args.corpora == "all" else tuple(args.corpora.split(","))
    if args.train_split == "all" and not args.fixed_steps:
        ap.error("--train-split all trains on the lines that would otherwise be "
                 "held out, so nothing is left to select on. Pass --fixed-steps "
                 "with the step count that won on a held-out run.")

    print(f"loading real Hebrew, {args.train_split} split", flush=True)
    heb_train = load_hebrew_ink(names, split=args.train_split, seed=args.seed)
    print("loading real Hebrew, test partition", flush=True)
    heb_val = load_hebrew_ink(names, split="test", seed=args.seed)
    if args.train_split == "all":
        print("  NOTE: these lines are IN the training set for this run. The CER "
              "printed below is a training-set number and selects nothing.", flush=True)
    if args.heldout_limit:
        heb_val = heb_val[: args.heldout_limit]
        print(f"  WARNING: scoring only {len(heb_val)} held-out lines", flush=True)
    print(f"  {len(heb_train)} train, {len(heb_val)} held out", flush=True)

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    charset = Charset(chars=state["charset"])
    model = build_model(charset.n_classes, state.get("config", {}).get("size", "large"),
                        mask_ratio=0.0)
    model.load_state_dict(state["model"])
    model.to(device).train()

    rehearsal = load_diffusionpen("train", max_cer=0.5, limit=40000)
    reh_ds = LineDataset(rehearsal, charset, train=True, seed=args.seed)
    heb_ds = LineDataset(HebrewInkSource(heb_train).rows, charset, train=True, seed=args.seed + 1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99),
                                  weight_decay=0.01)
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    rng = np.random.default_rng(args.seed)

    baseline = evaluate(model, charset, heb_val, device)
    print(f"before: held-out real Hebrew CER {baseline.cer_median:.4f}", flush=True)
    print("  (optimistic: the base model trained on these lines. Ranking only.)", flush=True)
    history = [{"step": 0, "heldout_cer": baseline.cer_median}]
    best, best_step = baseline.cer_median, 0
    torch.save({"model": model.state_dict(), "charset": charset.chars,
                "config": state.get("config", {}), "step": 0}, out / "best.pt")

    n_heb = max(1, int(round(args.batch_lines * args.hebrew_ratio)))
    n_reh = max(1, args.batch_lines - n_heb)
    print(f"each step: up to {n_heb} real Hebrew + {n_reh} rehearsal lines, "
          f"lr {args.lr}, pixel budget {args.pixel_budget}", flush=True)

    def take(dataset, count):
        """Draw lines, then drop whatever does not fit the pixel budget.

        Batching by line count is what OOMed the first attempt at this sweep,
        and it is the same mistake `PixelBudgetSampler` exists to prevent: these
        widths span 200 to 2560 px at a 64 px height, so sixteen short lines fit
        easily and sixteen long ones do not fit at all. The budget is
        lines x width-of-widest, which keeps peak memory roughly flat whatever
        the draw happens to contain.
        """
        picked, widest = [], 0
        for i in rng.integers(len(dataset), size=count * 2):
            item = dataset[int(i)]
            width = item[0].shape[-1]
            nxt = max(widest, width)
            if picked and (len(picked) + 1) * nxt > args.pixel_budget:
                continue
            picked.append(item)
            widest = nxt
            if len(picked) >= count:
                break
        return picked

    start = time.time()
    for step in range(1, args.steps + 1):
        picks = take(heb_ds, n_heb) + take(reh_ds, n_reh)
        if len(picks) < 2:
            continue
        batch = collate(picks)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logprobs = model(batch.images.to(device))
        loss = ctc(
            logprobs.float().permute(1, 0, 2),
            batch.targets.to(device),
            model.output_lengths(batch.widths).to(device),
            batch.target_lengths.to(device),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if args.fixed_steps and step == args.fixed_steps:
            # Carried-over settings: keep this checkpoint whatever it scores.
            torch.save({"model": model.state_dict(), "charset": charset.chars,
                        "config": state.get("config", {}), "step": step}, out / "best.pt")
            report = evaluate(model, charset, heb_val, device)
            print(f"  step {step:>5}  stopped at --fixed-steps, "
                  f"train-set CER {report.cer_median:.4f}", flush=True)
            best, best_step = report.cer_median, step
            break

        if step % args.eval_every == 0:
            report = evaluate(model, charset, heb_val, device)
            history.append({"step": step, "loss": float(loss.detach()),
                            "heldout_cer": report.cer_median})
            flag = ""
            if args.fixed_steps:
                pass
            elif report.cer_median < best:
                best, best_step = report.cer_median, step
                torch.save({"model": model.state_dict(), "charset": charset.chars,
                            "config": state.get("config", {}), "step": step}, out / "best.pt")
                flag = "  <- best"
            print(f"  step {step:>5}  loss {float(loss.detach()):.4f}  "
                  f"held-out CER {report.cer_median:.4f}{flag}", flush=True)
            (out / "history.json").write_text(json.dumps(history, indent=2))

    print(f"\nbest held-out real Hebrew CER {best:.4f} at step {best_step} "
          f"(was {baseline.cer_median:.4f}), {(time.time()-start)/60:.0f} min", flush=True)
    print(f"wrote {out}/best.pt", flush=True)
    print("\nthe benchmark has not been touched. Score the chosen checkpoint once:", flush=True)
    print(f"  python scripts/make_results.py {out}/best.pt --beam-width 12 --lm-weight 0.4",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
