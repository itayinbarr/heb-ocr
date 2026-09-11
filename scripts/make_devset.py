"""Build a held-out Hebrew dev set for tuning decode-time hyperparameters.

The benchmark is test-only by design, and this project's numbers are worth
quoting only because nothing has ever been selected on it. Decode-time knobs
(beam width, LM weights, ensemble weights, how many TTA views) are still
hyperparameters, so they need somewhere to be tuned that is not the benchmark.

The synthetic validation split is the obvious candidate and is too easy: the
shipped model reads it at CER 0.036 while it reads real paper at 0.26. Tuning at
0.036 optimizes for a regime the decoder will never operate in, where the
first-choice character is almost always right and an LM has nothing to fix.

So this builds a *hard* dev set: the same writer-independent validation split,
put through the training augmentation at a strength chosen so the model reads it
at roughly the error rate it shows on the benchmark. Same difficulty, same kinds
of degradation, no contact with the test set.

Writers never cross splits upstream, so no writer here was trained on.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
from PIL import Image

from hebocr.data.synth import load_diffusionpen
from hebocr.data.transforms import Augment


def build(n: int, strength: float, seed: int, max_cer: float | None):
    rows = load_diffusionpen("validation", max_cer=max_cer, limit=None)
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(rows), size=min(n, len(rows)), replace=False)

    augment = Augment(np.random.default_rng(seed + 1), strength=strength)
    images, texts = [], []
    for i, index in enumerate(picks):
        row = rows[int(index)]
        gray = augment.augment_gray(row["image"])
        images.append(Image.fromarray(gray, mode="L"))
        texts.append(row["text"])
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(picks)}", flush=True)
    return images, texts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="runs/dev/devset.pkl")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--strength", type=float, default=1.0,
                    help="augmentation strength; raise it until the model reads "
                         "this set at about its benchmark error rate")
    ap.add_argument("--seed", type=int, default=20260911)
    ap.add_argument("--max-cer", type=float, default=0.5)
    args = ap.parse_args()

    images, texts = build(args.n, args.strength, args.seed, args.max_cer)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump(
            {
                "images": [(im.tobytes(), im.size) for im in images],
                "texts": texts,
                "strength": args.strength,
                "seed": args.seed,
            },
            fh, protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"wrote {out}: {len(images)} lines at strength {args.strength}")
    return 0


def load_devset(path: str | Path = "runs/dev/devset.pkl"):
    """Images and gold texts, as `Recognizer.read` wants them."""
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    images = [Image.frombytes("L", size, blob) for blob, size in data["images"]]
    return images, data["texts"]


if __name__ == "__main__":
    raise SystemExit(main())
