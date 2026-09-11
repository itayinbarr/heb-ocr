"""Read both A/B arms and say whether SAM earned its two passes per step."""

import json
from pathlib import Path


def load(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def table(name: str, rows):
    if not rows:
        print(f"{name}: no epochs logged")
        return None
    print(f"\n{name}")
    print(f"  {'epoch':>5} {'hours':>7} {'val CER':>9} {'bench CER':>10} {'bench wcov':>11}")
    hours = 0.0
    for r in rows:
        hours += r.get("seconds", 0) / 3600
        print(f"  {r['epoch']:>5} {hours:>7.2f} {r['val']['cer_median']:>9.4f} "
              f"{r['benchmark']['cer_median']:>10.4f} {r['benchmark']['wcov_micro']:>11.4f}")
    return hours


def main() -> int:
    nosam = load(Path("runs/ab_nosam/log.jsonl"))
    sam = load(Path("runs/ab_sam/log.jsonl"))

    h_nosam = table("no SAM", nosam)
    h_sam = table("SAM", sam)

    if not nosam or not sam:
        print("\nboth arms must finish before there is anything to compare")
        return 1

    # The comparison that matters: the best each arm reached, and at what cost.
    best_nosam = min(r["benchmark"]["cer_median"] for r in nosam)
    best_sam = min(r["benchmark"]["cer_median"] for r in sam)

    print("\n" + "=" * 62)
    print(f"no SAM: best benchmark CER {best_nosam:.4f} in {h_nosam:.2f}h over {len(nosam)} epochs")
    print(f"SAM:    best benchmark CER {best_sam:.4f} in {h_sam:.2f}h over {len(sam)} epochs")

    # Benchmark CER here is greedy decoding, logged for visibility and never
    # used to select anything. It is the comparable number across arms.
    delta = best_nosam - best_sam
    relative = 100 * delta / best_nosam if best_nosam else 0.0
    if delta > 0:
        print(f"\nSAM is ahead by {delta:.4f} ({relative:+.1f}% relative) at comparable wall clock.")
        print("Take it into the long run.")
    else:
        print(f"\nSAM is behind by {-delta:.4f} ({relative:+.1f}% relative) at comparable wall clock.")
        print("The two extra passes per step buy less than twice the ordinary steps do.")
        print("Run the long job without it.")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
