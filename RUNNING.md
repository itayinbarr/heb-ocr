# What is running, and what to do with it

Started 2026-09-11 20:53. Expected to finish about 2026-09-15 04:00.

## The run

`scripts/run_final.sh`, detached in its own session, 16 epochs over 1,273,522
items an epoch, which is 20.4M samples at a measured 4.99 hours an epoch. No
SAM, because the A/B said so (`EXPERIMENTS.md`, and the raw logs are in
`experiments/sam_ab/`).

The mixture, for the first time, contains real Hebrew handwriting: all 10,219
lines of it, from Pinkas and BiblIA. Alongside that, 1M glyph-composed lines
generated on demand rather than stored, the full DiffusionPen train split, and
foreign real ink at the same proportion the shipped stage B used.

```bash
tail -f runs/final.log        # per-epoch progress
cat runs/final_run.log        # stage transitions, and the A/B verdict it read
```

When training ends the script scores the checkpoint itself and writes
`RESULTS_final.md` and `runs/results_final.json`. Nothing else is automatic.

## If it is still running and you want it to stop

```bash
pkill -f 'bash scripts/run_final'   # the supervisor
pkill -f 'hebocr.train --out runs/final'
```

Checkpoints are written every epoch, so `runs/final/best.pt` is usable whenever
you stop it. `best.pt` is selected on synthetic validation CER, never on the
benchmark, which is the rule the whole project's numbers depend on.

## If it died

The two ways it can, and both are guarded:

- **Something took the GPU.** `mogli-asr.service` has `Restart=always` and this
  is exactly how the first A/B attempt died forty seconds in. The script now
  refuses to start if the card is busy and names what holds it. If you see that
  message, `systemctl --user stop mogli-asr.service` and relaunch.
- **The link dropped.** It cannot take the run with it: the run is detached and
  needs no network. `wifi-keepalive.service` reconnects to `Itay` every 60s.

Relaunching starts from scratch, since `--resume` takes a checkpoint path and
the script does not pass one. To continue instead, run `hebocr.train` directly
with `--resume runs/final/last.pt`.

## What to do when it finishes

1. Read `RESULTS_final.md`. The number to compare against is the shipped model's
   **0.213 line CER** and **0.349 full-page word coverage**, not its 0.222,
   because the decode-time study moved it.
2. If it is better, the decode study's conclusions are worth rechecking rather
   than assumed: TTA and the word prior were tuned against the old checkpoint
   and both are decode-time, so they cost nothing to re-measure.
   `scripts/eval_configs.py --configs decode/benchmark_configs.json --pages`
   does it in one pass.
3. **The licence changes if this model ships.** It is trained on BiblIA, which
   is CC-BY-NC-SA-4.0, so the weights cannot go out as CC-BY-4.0 the way
   Mishkefet-v1 did. `LICENSE`, `README.md` and the Hugging Face model card all
   state CC-BY-4.0 today and all three would need updating together. Training
   with `--real-hebrew pinkas` instead keeps the old terms at the cost of 9,276
   of the 10,219 real Hebrew lines.

## Things that are switched off

Both were stopped to free the 8 GB card, with permission:

```bash
systemctl --user start mogli-asr.service   # mogli's speech-to-text, currently down
```

llama.cpp is also down; its exact command line is in
`~/.claude/jobs/a1692ecb/tmp/llama-server-restart.txt`. Starting either before
the run finishes will not kill it, but will slow it and risk an out-of-memory
error, since the pixel budget assumes a free card.
