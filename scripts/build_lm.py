"""Train the character n-gram LM used for shallow fusion at decode time.

Trained on Hebrew Wikipedia. The corpus is capped rather than exhaustive: an
unbounded character 6-gram count table over hundreds of millions of characters
does not fit comfortably in memory, and the returns past a few tens of millions
of characters are small for a model this simple.
"""

import argparse
import re

from hebocr.lm import CharNGramLM

SENTENCE_SPLIT = re.compile(r"[\n.!?;]+")
# Keep Hebrew, digits, and the punctuation the charset knows. Wikipedia is full
# of Latin script, wiki markup and CJK, none of which the recognizer can emit,
# and all of which would distort the letter statistics.
KEEP = re.compile(r"[^֐-׿0-9 '\"\-.,:;!?()\[\]/]+")


def lines_from_wikipedia(max_chars: int, config: str):
    from datasets import load_dataset

    stream = load_dataset("wikimedia/wikipedia", config, split="train", streaming=True)
    seen = 0
    for article in stream:
        for chunk in SENTENCE_SPLIT.split(article["text"]):
            cleaned = KEEP.sub(" ", chunk).strip()
            cleaned = re.sub(r"\s+", " ", cleaned)
            # Require real Hebrew content, not a stub of digits and brackets.
            if len(cleaned) < 12:
                continue
            hebrew = sum("֐" <= c <= "׿" for c in cleaned)
            if hebrew < len(cleaned) * 0.5:
                continue
            yield cleaned
            seen += len(cleaned)
            if seen >= max_chars:
                return


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="runs/lm/hebrew_char6.pkl")
    ap.add_argument("--order", type=int, default=6)
    ap.add_argument("--max-chars", type=int, default=30_000_000)
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--config", default="20231101.he")
    args = ap.parse_args()

    print(f"training a char {args.order}-gram LM on up to {args.max_chars:,} characters", flush=True)
    lm = CharNGramLM(order=args.order)
    lm.train(lines_from_wikipedia(args.max_chars, args.config), progress_every=200_000)

    print("before pruning:", lm.stats(), flush=True)
    lm.prune(min_count=args.min_count)
    print("after pruning: ", lm.stats(), flush=True)

    lm.save(args.out)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
