"""Strip a training checkpoint down to what inference actually needs.

A training checkpoint carries AdamW's two moment buffers and, when an EMA is
tracked, a second copy of the weights kept only for diagnosis. None of that is
needed to read a line of Hebrew, and it roughly quadruples the file. This drops
everything except the weights, the charset and the architecture config, which
is what you hand to someone else.
"""

import argparse
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint")
    ap.add_argument("--out", default=None, help="default: <checkpoint stem>-release.pt")
    ap.add_argument("--half", action="store_true",
                    help="store weights as float16 (halves the file; CTC decoding is unaffected)")
    args = ap.parse_args()

    source = Path(args.checkpoint)
    target = Path(args.out) if args.out else source.with_name(source.stem + "-release.pt")

    state = torch.load(source, map_location="cpu", weights_only=False)
    weights = state["model"]
    if args.half:
        weights = {k: (v.half() if v.is_floating_point() else v) for k, v in weights.items()}

    torch.save(
        {
            "model": weights,
            "charset": state["charset"],
            # Only the fields build_model needs, so the release file does not
            # carry training-only settings that would confuse a reader.
            "config": {"size": state.get("config", {}).get("size", "base")},
            "epoch": state.get("epoch"),
            "used_ema": state.get("used_ema"),
            "val": state.get("val"),
            "benchmark": state.get("benchmark"),
        },
        target,
    )

    before = source.stat().st_size / 2**20
    after = target.stat().st_size / 2**20
    print(f"{source}  {before:.1f} MB")
    print(f"{target}  {after:.1f} MB  ({before/max(after, 1e-9):.1f}x smaller)")
    print(f"  parameters: {sum(v.numel() for v in weights.values())/1e6:.1f}M")
    print(f"  charset:    {len(state['charset'])} classes")
    print(f"  from epoch {state.get('epoch')}, used_ema={state.get('used_ema')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
