"""What CER does a careful human score on this benchmark?

Every gold line was transcribed independently by at least two volunteers and
then adjudicated. Scoring one volunteer's read against the adjudicated gold
gives the noise floor: roughly the best any model could post, and the honest
context for reading a leaderboard number.
"""

from datasets import load_dataset

from hebocr.metrics import cer, line_report


def main() -> int:
    rows = load_dataset("ivrit-ai/hebrew-handwriting-ocr-benchmark", "lines", split="test")

    pairs = []
    for row in rows:
        reads, chosen = list(row["reads"]), row["chosen_read_index"]
        others = [r for i, r in enumerate(reads) if i != chosen]
        if others:
            pairs.append((row["text"], others[0]))

    report = line_report([g for g, _ in pairs], [h for _, h in pairs])
    exact = sum(1 for g, h in pairs if cer(g, h) == 0)

    print(f"independent human read vs adjudicated gold ({len(pairs)}/{len(rows)} lines)")
    print(f"  {report.summary()}")
    print(f"  exact agreement: {exact}/{len(pairs)} = {exact/len(pairs):.1%}")
    print("\nFor comparison, the best published model (gemini-flash) scores 0.119 median.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
