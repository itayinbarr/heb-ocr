"""Combining several recognizers by voting on their output strings.

The obvious way to combine CTC models is to average their frame distributions
before decoding, which keeps more information than voting on finished strings.
It does not work here, and the reason is worth stating because it is a property
of CTC rather than an implementation detail.

CTC constrains the *order* of the characters a model emits and says nothing
about *when* it emits them. The alignment is latent, and each training run
settles into its own, typically a sharp spike somewhere inside the character
with blanks on either side. Measured between this project's two best
checkpoints, on 40 dev lines: of the frames where either model emits a
character, only 19.9% are frames where both do, and for characters the two
models agree on, 96% sit at different frames, median offset one frame.

So frame-averaging lines up one model's spike against the other model's blank
and averages both away. It is not a small effect: an equal-weight frame average
of the two scored CER 0.697 against 0.250 for the better model alone, and it
recovered toward the single model (0.288) only as the weighting was skewed far
enough to make the second model irrelevant.

Voting on strings has no alignment to get wrong, which is exactly why ROVER is
the standard method for combining independently trained transcription systems.
Characters are aligned to a pivot hypothesis by edit distance, and each slot is
decided by weighted vote.
"""

from dataclasses import dataclass

from rapidfuzz.distance import Levenshtein


@dataclass
class _Alignment:
    """One hypothesis, expressed against the pivot's coordinate system."""

    # aligned[i] is what this hypothesis puts opposite pivot[i]: a character, or
    # "" when it deletes that position.
    aligned: list[str]
    # inserts[i] is what it adds *before* pivot[i]; inserts[len(pivot)] is the tail.
    inserts: list[str]


def _align(pivot: str, hypothesis: str) -> _Alignment:
    aligned = [""] * len(pivot)
    inserts = [""] * (len(pivot) + 1)

    for op in Levenshtein.opcodes(pivot, hypothesis):
        n_src = op.src_end - op.src_start
        n_dst = op.dest_end - op.dest_start
        if op.tag == "equal":
            for k in range(n_src):
                aligned[op.src_start + k] = hypothesis[op.dest_start + k]
        elif op.tag == "replace":
            for k in range(n_src):
                aligned[op.src_start + k] = (
                    hypothesis[op.dest_start + k] if k < n_dst else ""
                )
            if n_dst > n_src:
                inserts[op.src_end] += hypothesis[op.dest_start + n_src : op.dest_end]
        elif op.tag == "delete":
            for k in range(op.src_start, op.src_end):
                aligned[k] = ""
        elif op.tag == "insert":
            inserts[op.src_start] += hypothesis[op.dest_start : op.dest_end]
    return _Alignment(aligned=aligned, inserts=inserts)


def rover(hypotheses: list[str], weights: list[float] | None = None) -> str:
    """Weighted character-level vote over transcriptions of one line.

    `hypotheses[0]` is the pivot and should be the strongest system: it defines
    the coordinate system every other hypothesis is expressed in, and it wins
    ties. That asymmetry is deliberate. With a small number of systems of
    visibly different quality, a symmetric vote lets two weaker systems that
    share a mistake outvote the stronger one that did not make it.

    An insertion is only accepted when the systems proposing it outweigh those
    that do not, so a single system cannot add text the others never saw.
    """
    if not hypotheses:
        return ""
    if len(hypotheses) == 1:
        return hypotheses[0]

    weights = list(weights) if weights else [1.0] * len(hypotheses)
    if len(weights) != len(hypotheses):
        raise ValueError("one weight per hypothesis, or none at all")

    pivot, pivot_weight = hypotheses[0], weights[0]
    others = [(_align(pivot, h), w) for h, w in zip(hypotheses[1:], weights[1:])]
    total = sum(weights)

    out = []
    for i in range(len(pivot) + 1):
        # Insertions before this position, then the position itself.
        insert_votes: dict[str, float] = {}
        for alignment, weight in others:
            text = alignment.inserts[i]
            if text:
                insert_votes[text] = insert_votes.get(text, 0.0) + weight
        if insert_votes:
            text, weight = max(insert_votes.items(), key=lambda kv: kv[1])
            if weight > total / 2:
                out.append(text)

        if i == len(pivot):
            break

        votes: dict[str, float] = {pivot[i]: pivot_weight}
        for alignment, weight in others:
            char = alignment.aligned[i]
            votes[char] = votes.get(char, 0.0) + weight
        # Ties go to the pivot, which is why its own vote is seeded first and
        # max() over an insertion-ordered dict keeps it.
        best = max(votes.items(), key=lambda kv: kv[1])[0]
        out.append(best)

    return "".join(out)


def rover_batch(
    hypothesis_sets: list[list[str]], weights: list[float] | None = None
) -> list[str]:
    """`hypothesis_sets[i]` is every system's reading of line i."""
    return [rover(h, weights) for h in hypothesis_sets]
