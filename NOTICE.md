# What each piece is licensed under

Three different things ship from this project and they do not share terms.

## The code: MIT

Everything in this repository. See [`LICENSE`](LICENSE).

## The trained weights: CC-BY-NC-SA-4.0

Published at
[`itayinbar/Mishkefet-v1`](https://huggingface.co/itayinbar/Mishkefet-v1), not
here, and **not under the MIT licence that covers the code**.

The data requires it. The model is trained on
[BiblIA](https://zenodo.org/records/5167263), which is CC-BY-NC-SA-4.0, and
those terms propagate to anything trained on it: NonCommercial means the weights
may not be used commercially, and ShareAlike means derivatives carry the same
licence.

Training with `--real-hebrew pinkas` produces CC-BY-4.0 weights instead, using
only the permissively licensed Hebrew corpus. That costs 9,276 of the 10,219
real Hebrew lines and scores worse.

## The training data: various, and one of them needs care

| dataset | licence |
|---|---|
| [`ivrit-ai/hebrew-handwriting-ocr-benchmark`](https://huggingface.co/datasets/ivrit-ai/hebrew-handwriting-ocr-benchmark) | ivrit.ai License (gated). Test only, never trained on |
| [`cyttic/diffusionpen-hebrew-handwriting`](https://huggingface.co/datasets/cyttic/diffusionpen-hebrew-handwriting) | CC-BY-4.0 |
| [`sivan22/hebrew-handwritten-dataset`](https://huggingface.co/datasets/sivan22/hebrew-handwritten-dataset) | CC-BY-3.0 |
| [Pinkas](https://zenodo.org/records/3569694) | CC-BY-4.0 |
| [BiblIA](https://zenodo.org/records/5167263) | **CC-BY-NC-SA-4.0** |
| KHATT, IAM, NorHand, Belfort, HOME-Alcar, NewsEye, Himanis, RIMES, Esposalles, POPP | MIT as published on the Hub |
| Hebrew Wikipedia | CC-BY-SA |

**The KHATT and IAM mirrors are labelled MIT on the Hub, and the original
corpora are not.** IAM-DB has historically been restricted to non-commercial
research use, and KHATT carries its own terms. Anyone intending commercial use
should verify upstream rather than relying on a mirror's label, which is a
separate concern from the BiblIA clause above and points the same way.
