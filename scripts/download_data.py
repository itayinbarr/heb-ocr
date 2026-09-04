"""Fetch every dataset the pipeline needs into the local HF cache.

Run once before training. Everything here is either public or gated behind a
license checkbox you accept on the dataset page with your own HF account.
"""

import argparse
import sys

from huggingface_hub import snapshot_download

# repo_id -> why we need it
REPOS = {
    "ivrit-ai/hebrew-handwriting-ocr-benchmark": "the 225-line / 10-page gold test set (test only, never trained on)",
    "cyttic/diffusionpen-hebrew-handwriting": "150k synthetic Hebrew handwriting lines, 491 writer styles",
}

OPTIONAL = {
    "cyttic/diffusionpen-hebrew-handwriting-cer0": "the clean-only subset of the above",
    "sivan22/hebrew-handwritten-dataset": "HHD single-character glyphs (real handwriting, weak signal)",
}


def fetch(repo_id: str, why: str) -> bool:
    print(f"\n=== {repo_id}\n    {why}", flush=True)
    try:
        path = snapshot_download(repo_id=repo_id, repo_type="dataset")
    except Exception as exc:  # noqa: BLE001 - we want the reason, not a traceback
        print(f"    FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        if "restricted" in str(exc) or "gated" in str(exc).lower():
            print(
                f"    -> accept the license at https://huggingface.co/datasets/{repo_id}",
                file=sys.stderr,
                flush=True,
            )
        return False
    print(f"    ok: {path}", flush=True)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--optional", action="store_true", help="also fetch the optional extras")
    args = ap.parse_args()

    repos = dict(REPOS)
    if args.optional:
        repos.update(OPTIONAL)

    failed = [r for r, why in repos.items() if not fetch(r, why)]
    if failed:
        print(f"\n{len(failed)} repo(s) failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    print("\nall datasets present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
