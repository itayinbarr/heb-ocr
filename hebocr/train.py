"""Training loop for the CTC line recognizer.

Checkpoint selection uses the synthetic validation split, never the ivrit.ai
benchmark. The benchmark is scored during training too, but only ever logged --
if it picked the checkpoint it would stop being a held-out test set and every
number this repo reports would be inflated.
"""

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .charset import BLANK, Charset
from .data.synth import (
    LineDataset, MixedLineDataset, PixelBudgetSampler, build_glyph_lines,
    collate, image_widths, load_diffusionpen, load_real_ink,
)
from .decode import greedy_decode
from .metrics import line_report
from .models.htr_vt import build_model
from .models.trocr_ctc import build_trocr_ctc
from .optim import SAM, ModelEMA


@dataclass
class Config:
    arch: str = "htrvt"
    size: str = "base"
    freeze_layers: int = 0
    head_lr_mult: float = 1.0
    aug_warmup_epochs: int = 0
    epochs: int = 40
    pixel_budget: int = 20000
    max_batch: int = 64
    concat_prob: float = 0.35
    max_concat: int = 3
    glyph_lines: int = 0
    real_ink: tuple = ()
    eval_batch_size: int = 12
    lr: float = 3e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.05
    warmup_steps: int = 1500
    grad_clip: float = 1.0
    accum_steps: int = 1
    use_sam: bool = False
    ema_decay: float = 0.0
    sam_rho: float = 0.05
    mask_ratio: float = 0.4
    aug_strength: float = 1.0
    max_train_cer: float = 0.5
    num_workers: int = 8
    seed: int = 0
    train_limit: int | None = None
    val_limit: int | None = 4000
    eval_every: int = 1
    amp: bool = True


def _lr_at(step: int, cfg: Config, total_steps: int) -> float:
    """Linear warmup, then cosine decay to `min_lr`."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(cfg.warmup_steps, 1)
    progress = (step - cfg.warmup_steps) / max(total_steps - cfg.warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))


def ctc_loss(logprobs: torch.Tensor, batch, model) -> torch.Tensor:
    """CTC over the batch, with impossible samples masked out.

    A line whose transcription is longer than the number of CTC time steps
    cannot be aligned at all. `zero_infinity` would silently zero those, but
    they would still dilute the mean, so they are dropped explicitly.
    """
    input_lengths = model.output_lengths(batch.widths)
    keep = input_lengths >= batch.target_lengths
    if not keep.any():
        return logprobs.sum() * 0.0

    if not keep.all():
        # Rebuild the flat target buffer from only the alignable rows.
        offsets = torch.cat([torch.zeros(1, dtype=torch.long), batch.target_lengths.cumsum(0)])
        pieces = [
            batch.targets[offsets[i] : offsets[i + 1]]
            for i in range(len(batch.target_lengths))
            if keep[i]
        ]
        targets = torch.cat(pieces) if pieces else batch.targets[:0]
        target_lengths = batch.target_lengths[keep]
        logprobs = logprobs[keep]
        input_lengths = input_lengths[keep]
    else:
        targets, target_lengths = batch.targets, batch.target_lengths

    return F.ctc_loss(
        logprobs.permute(1, 0, 2).float(),
        targets,
        input_lengths.to(logprobs.device),
        target_lengths.to(logprobs.device),
        blank=BLANK,
        reduction="mean",
        zero_infinity=True,
    )


@torch.no_grad()
def evaluate_loader(model, loader, charset: Charset, device, amp: bool, limit: int | None = None):
    """Greedy-decode a loader and score it. Returns a `LineReport`."""
    model.eval()
    refs, hyps = [], []
    for batch in loader:
        images = batch.images.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            logprobs = model(images)
        lengths = model.output_lengths(batch.widths)
        hyps.extend(greedy_decode(logprobs.float(), lengths, charset))
        refs.extend(batch.texts)
        if limit is not None and len(refs) >= limit:
            break
    model.train()
    return line_report(refs, hyps)


@torch.no_grad()
def evaluate_benchmark(model, charset: Charset, device, amp: bool, batch_size: int = 16):
    """Score the held-out ivrit.ai line benchmark. Logged only, never selected on."""
    from .data.benchmark import load_lines
    from .data.transforms import preprocess

    model.eval()
    lines = load_lines()
    refs = [l.text for l in lines]
    order = sorted(range(len(lines)), key=lambda i: lines[i].image.size[0] / max(lines[i].image.size[1], 1))
    hyps: dict[int, str] = {}

    for start in range(0, len(order), batch_size):
        chunk = order[start : start + batch_size]
        arrays = [torch.from_numpy(preprocess(lines[i].image)) for i in chunk]
        widths = torch.tensor([a.shape[-1] for a in arrays], dtype=torch.long)
        padded = torch.zeros(len(arrays), 1, arrays[0].shape[-2], int(widths.max()))
        for j, a in enumerate(arrays):
            padded[j, :, :, : a.shape[-1]] = a
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            logprobs = model(padded.to(device))
        for j, text in enumerate(greedy_decode(logprobs.float(), model.output_lengths(widths), charset)):
            hyps[chunk[j]] = text

    model.train()
    return line_report(refs, [hyps[i] for i in range(len(refs))])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="runs/base", help="checkpoint and log directory")
    ap.add_argument("--size", default="base", choices=["small", "base", "large"])
    ap.add_argument("--arch", default="htrvt", choices=["htrvt", "trocr"],
                    help="htrvt: our from-scratch CNN+ViT. trocr: pretrained handwriting ViT encoder")
    ap.add_argument("--freeze-layers", type=int, default=0,
                    help="trocr only: freeze the lowest N encoder blocks")
    ap.add_argument("--head-lr-mult", type=float, default=1.0,
                    help="learning-rate multiplier for newly initialized layers")
    ap.add_argument("--aug-warmup-epochs", type=int, default=0,
                    help="ramp augmentation strength from a third to full over N epochs")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--pixel-budget", type=int, default=20000,
                    help="max sum of padded pixels per batch (lines x widest line)")
    ap.add_argument("--max-batch", type=int, default=64)
    ap.add_argument("--concat-prob", type=float, default=0.35,
                    help="probability of joining lines to match benchmark line lengths")
    ap.add_argument("--glyph-lines", type=int, default=0,
                    help="how many training lines to compose from real HHD glyphs")
    ap.add_argument("--real-ink", default="",
                    help="comma-separated real-handwriting sources to mix in (khatt,iam)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--accum-steps", type=int, default=1)
    ap.add_argument("--sam", action="store_true", help="use SAM (2x step cost)")
    ap.add_argument("--ema", type=float, default=0.0,
                    help="EMA decay for an averaged copy of the weights (0 disables)")
    ap.add_argument("--aug-strength", type=float, default=1.0)
    ap.add_argument("--mask-ratio", type=float, default=0.4)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--train-limit", type=int, default=None)
    ap.add_argument("--val-limit", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = Config(
        arch=args.arch, size=args.size, freeze_layers=args.freeze_layers,
        head_lr_mult=args.head_lr_mult, aug_warmup_epochs=args.aug_warmup_epochs,
        epochs=args.epochs, lr=args.lr,
        pixel_budget=args.pixel_budget, max_batch=args.max_batch, concat_prob=args.concat_prob,
        glyph_lines=args.glyph_lines,
        real_ink=tuple(x for x in args.real_ink.split(",") if x),
        accum_steps=args.accum_steps, use_sam=args.sam, aug_strength=args.aug_strength,
        mask_ratio=args.mask_ratio, num_workers=args.num_workers, ema_decay=args.ema,
        train_limit=args.train_limit, val_limit=args.val_limit, seed=args.seed,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  torch {torch.__version__}", flush=True)

    charset = Charset.default()

    print("loading data...", flush=True)
    train_rows = load_diffusionpen("train", max_cer=cfg.max_train_cer, limit=cfg.train_limit)
    val_rows = load_diffusionpen("validation", max_cer=cfg.max_train_cer, limit=cfg.val_limit)
    print(f"train {len(train_rows)}  val {len(val_rows)}", flush=True)

    val_ds = LineDataset(val_rows, charset, train=False)

    print("measuring image widths...", flush=True)
    train_widths = image_widths(train_rows, cache=str(out / "train_widths.npy"))

    extra_sources, extra_widths = [], []
    if cfg.real_ink:
        print(f"loading real handwriting: {', '.join(cfg.real_ink)}", flush=True)
        for name, ds in load_real_ink(cfg.real_ink):
            extra_sources.append(ds)
            extra_widths.append(image_widths(ds, cache=str(out / f"widths_{name}.npy")))
            print(f"  {name}: {len(ds)} real lines", flush=True)
        if extra_sources:
            # These lines carry their own alphabets. Hebrew keeps its class
            # indices; the foreign letters are appended after them.
            before = charset.n_classes
            for ds in extra_sources:
                charset = charset.extend(ds["text"], min_count=20)
            print(f"  charset {before} -> {charset.n_classes} classes", flush=True)

    charset.save(out / "charset.json")
    print(f"charset: {charset.n_classes} classes", flush=True)

    glyph_items = []
    if cfg.glyph_lines:
        # Real handwritten ink. DiffusionPen's generated strokes are the one
        # thing the model never sees a real version of, and that shows up as the
        # gap between synthetic validation CER and benchmark CER.
        print(f"composing {cfg.glyph_lines} lines from real HHD glyphs...", flush=True)
        glyph_items = build_glyph_lines(train_rows["text"], cfg.glyph_lines, seed=cfg.seed)
        print(f"  got {len(glyph_items)} glyph lines", flush=True)

    train_ds = MixedLineDataset(
        train_rows, glyph_items, charset, train=True,
        aug_strength=cfg.aug_strength * (0.33 if cfg.aug_warmup_epochs else 1.0),
        seed=cfg.seed, extra_sources=extra_sources,
    )
    train_widths = train_ds.widths(train_widths, extra_widths)
    print(
        f"train items: {len(train_ds)} "
        f"({len(glyph_items)} glyph-composed, {sum(len(s) for s in extra_sources)} real-ink)",
        flush=True,
    )
    print(
        f"widths: mean {train_widths.mean():.0f}  p95 {np.percentile(train_widths, 95):.0f}"
        f"  max {train_widths.max()}",
        flush=True,
    )
    sampler = PixelBudgetSampler(
        train_widths,
        budget=cfg.pixel_budget,
        max_batch=cfg.max_batch,
        concat_prob=cfg.concat_prob,
        max_concat=cfg.max_concat,
        shuffle=True,
        seed=cfg.seed,
        # Hebrew sources occupy the leading indices; the real-ink lines that
        # follow are left out of concatenation because it assumes RTL.
        concat_max_index=len(train_rows) + len(glyph_items),
    )

    train_loader = DataLoader(
        train_ds, batch_sampler=sampler, collate_fn=collate,
        num_workers=cfg.num_workers, pin_memory=True,
        # Not persistent: workers are re-forked each epoch so they pick up a
        # changed augmentation strength.
        persistent_workers=False,
        prefetch_factor=4 if cfg.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.eval_batch_size, shuffle=False, collate_fn=collate,
        num_workers=max(2, cfg.num_workers // 2), pin_memory=True,
    )

    if cfg.arch == "trocr":
        model = build_trocr_ctc(charset.n_classes, freeze_layers=cfg.freeze_layers).to(device)
        label = f"trocr-ctc (frozen {cfg.freeze_layers} layers)"
    else:
        model = build_model(charset.n_classes, cfg.size, mask_ratio=cfg.mask_ratio).to(device)
        label = f"htrvt {cfg.size}"
    n_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"model {label}: {n_params/1e6:.1f}M params "
        f"({trainable/1e6:.1f}M trainable)  SAM={cfg.use_sam}",
        flush=True,
    )

    # Split by weight decay, and separately by whether a parameter is
    # pretrained. A freshly initialized head has to move much further than a
    # pretrained encoder does, and holding both at one learning rate leaves the
    # head crawling: with a random CTC head at the encoder's rate the model sits
    # in CTC's all-blank solution for epochs before it escapes.
    NEW_MODULE_PREFIXES = ("proj", "norm", "head")

    buckets: dict[tuple[bool, bool], list] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_new = name.startswith(NEW_MODULE_PREFIXES)
        buckets.setdefault((is_new, param.ndim <= 1), []).append(param)

    groups = []
    for (is_new, is_bias), params in buckets.items():
        groups.append({
            "params": params,
            "weight_decay": 0.0 if is_bias else cfg.weight_decay,
            "lr_mult": cfg.head_lr_mult if is_new else 1.0,
        })
    n_new = sum(len(v) for (is_new, _), v in buckets.items() if is_new)
    if cfg.head_lr_mult != 1.0:
        print(
            f"learning rate: {n_new} newly initialized tensors at "
            f"{cfg.head_lr_mult}x the encoder rate",
            flush=True,
        )
    if cfg.use_sam:
        optimizer = SAM(groups, torch.optim.AdamW, rho=cfg.sam_rho, lr=cfg.lr, betas=(0.9, 0.99))
    else:
        optimizer = torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.99))

    ema = ModelEMA(model, decay=cfg.ema_decay) if cfg.ema_decay else None
    if ema is not None:
        print(f"tracking an EMA of the weights (decay {cfg.ema_decay})", flush=True)

    steps_per_epoch = math.ceil(len(sampler) / cfg.accum_steps)
    total_steps = steps_per_epoch * cfg.epochs
    start_epoch, step, best_val = 0, 0, float("inf")

    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch, step, best_val = ck["epoch"] + 1, ck["step"], ck.get("best_val", float("inf"))
        print(f"resumed from {args.resume} at epoch {start_epoch}", flush=True)

    log_path = out / "log.jsonl"
    amp = cfg.amp and device.type == "cuda"

    def set_lr(value: float) -> None:
        for g in optimizer.param_groups:
            g["lr"] = value * g.get("lr_mult", 1.0)

    def log(record: dict) -> None:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"training {cfg.epochs} epochs, {steps_per_epoch} steps/epoch", flush=True)
    model.train()

    for epoch in range(start_epoch, cfg.epochs):
        sampler.set_epoch(epoch)

        if cfg.aug_warmup_epochs:
            # Ease the model into the distortions rather than opening at full
            # strength. Verified separately: this architecture reaches CER 0.000
            # overfitting clean crops, so early failure was the augmentation,
            # not the model.
            ramp = min(1.0, (epoch + 1) / (cfg.aug_warmup_epochs + 1))
            strength = cfg.aug_strength * (0.33 + 0.67 * ramp)
            if abs(strength - train_ds.aug_strength) > 1e-6:
                train_ds.aug_strength = strength
                print(f"  augmentation strength {strength:.2f}", flush=True)
        epoch_loss, n_batches = 0.0, 0
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(train_loader):
            images = batch.images.to(device, non_blocking=True)
            lr = _lr_at(step, cfg, total_steps)
            set_lr(lr)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                loss = ctc_loss(model(images), batch, model)
            (loss / cfg.accum_steps).backward()

            if (i + 1) % cfg.accum_steps == 0:
                if cfg.use_sam:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    optimizer.first_step(zero_grad=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                        ctc_loss(model(images), batch, model).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    optimizer.second_step(zero_grad=True)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                step += 1
                if ema is not None:
                    ema.update(model)

            epoch_loss += float(loss.detach())
            n_batches += 1

            if n_batches % 200 == 0:
                rate = n_batches / (time.time() - t0)
                print(
                    f"  epoch {epoch} [{n_batches}/{len(sampler)}] "
                    f"loss {epoch_loss/n_batches:.4f}  lr {lr:.2e}  {rate:.1f} it/s",
                    flush=True,
                )

        train_loss = epoch_loss / max(n_batches, 1)
        record = {
            "epoch": epoch, "step": step, "train_loss": train_loss,
            "lr": lr, "seconds": round(time.time() - t0, 1),
        }

        if (epoch + 1) % cfg.eval_every == 0 or epoch == cfg.epochs - 1:
            # Score the raw weights and the EMA separately and keep whichever is
            # better. An EMA starts from the random initialization and needs
            # roughly 1/(1-decay) steps to shed it, so early on it is far worse
            # than the live weights -- scoring only the EMA hides the model's
            # real trajectory for the first few epochs.
            val = evaluate_loader(model, val_loader, charset, device, amp, limit=cfg.val_limit)
            bench = evaluate_benchmark(model, charset, device, amp, batch_size=cfg.eval_batch_size)
            use_ema = False

            if ema is not None:
                shadow = ema.shadow.to(device)
                ema_val = evaluate_loader(shadow, val_loader, charset, device, amp, limit=cfg.val_limit)
                ema_bench = evaluate_benchmark(shadow, charset, device, amp, batch_size=cfg.eval_batch_size)
                record["ema_val"] = ema_val.as_dict()
                record["ema_benchmark"] = ema_bench.as_dict()
                print(
                    f"           ema        val {ema_val.summary()}\n"
                    f"           ema        bench {ema_bench.summary()}",
                    flush=True,
                )
                if ema_val.cer_micro < val.cer_micro:
                    val, bench, use_ema = ema_val, ema_bench, True
            record["val"] = val.as_dict()
            record["benchmark"] = bench.as_dict()
            print(
                f"epoch {epoch}  loss {train_loss:.4f}  "
                f"val {val.summary()}\n"
                f"           benchmark(held out, not selected on)  {bench.summary()}",
                flush=True,
            )

            # Selection metric: micro CER on synthetic validation. Micro rather
            # than median because median saturates at 0 on easy synthetic lines
            # long before the model is actually good.
            score = val.cer_micro
            if score < best_val:
                best_val = score
                torch.save(
                    {
                        "model": (ema.state_dict() if use_ema else model.state_dict()),
                        "used_ema": use_ema,
                        "raw_model": model.state_dict() if use_ema else None,
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch, "step": step, "best_val": best_val,
                        "config": asdict(cfg), "charset": charset.chars,
                        "val": val.as_dict(), "benchmark": bench.as_dict(),
                    },
                    out / "best.pt",
                )
                print(f"           new best val CER {best_val:.4f} -> {out/'best.pt'}", flush=True)
        else:
            print(f"epoch {epoch}  loss {train_loss:.4f}", flush=True)

        torch.save(
            {
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": epoch, "step": step, "best_val": best_val,
                "config": asdict(cfg), "charset": charset.chars,
            },
            out / "last.pt",
        )
        log(record)

    print(f"done. best synthetic-val CER {best_val:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
