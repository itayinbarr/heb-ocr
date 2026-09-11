"""Train the word unigram model used for word-boundary fusion at decode time.

Same corpus as the character LM, counted differently. Reads either a local text
file (one passage per line) or streams Hebrew Wikipedia, so the build does not
depend on network access when a corpus is already on disk.
"""

import argparse
import re

from hebocr.wordlm import WordUnigramLM

SENTENCE_SPLIT = re.compile(r"[\n.!?;]+")
KEEP = re.compile(r"[^֐-׿0-9 '\"\-.,:;!?()\[\]/]+")


def _clean(chunk: str) -> str | None:
    cleaned = re.sub(r"\s+", " ", KEEP.sub(" ", chunk)).strip()
    if len(cleaned) < 12:
        return None
    hebrew = sum("֐" <= c <= "׿" for c in cleaned)
    return cleaned if hebrew >= len(cleaned) * 0.5 else None


def lines_from_file(path: str, max_chars: int):
    seen = 0
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            for chunk in SENTENCE_SPLIT.split(raw):
                cleaned = _clean(chunk)
                if not cleaned:
                    continue
                yield cleaned
                seen += len(cleaned)
                if seen >= max_chars:
                    return


def lines_from_wikipedia(max_chars: int, config: str):
    from datasets import load_dataset

    stream = load_dataset("wikimedia/wikipedia", config, split="train", streaming=True)
    seen = 0
    for article in stream:
        for chunk in SENTENCE_SPLIT.split(article["text"]):
            cleaned = _clean(chunk)
            if not cleaned:
                continue
            yield cleaned
            seen += len(cleaned)
            if seen >= max_chars:
                return


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="runs/lm/hebrew_words.pkl")
    ap.add_argument("--corpus", default=None, help="local text file; else stream Wikipedia")
    ap.add_argument("--max-chars", type=int, default=120_000_000)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--config", default="20231101.he")
    args = ap.parse_args()

    source = (
        lines_from_file(args.corpus, args.max_chars) if args.corpus
        else lines_from_wikipedia(args.max_chars, args.config)
    )
    lm = WordUnigramLM()
    lm.train(source, progress_every=500_000)
    print("before pruning:", lm.stats(), flush=True)
    lm.prune(min_count=args.min_count)
    print("after pruning: ", lm.stats(), flush=True)
    lm.save(args.out)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
