"""Training data: synthetic Hebrew handwriting lines.

Every training image in this project is synthetic. The ivrit.ai benchmark is
test-only by design ("There is no train split, by design -- it exists to be
held out"), and no public corpus of real modern Hebrew cursive with line-level
transcriptions exists. So the training set is DiffusionPen output and the
entire modelling problem is the domain gap to photographed paper, which is why
`transforms.Augment` is as aggressive as it is.

DiffusionPen Hebrew (cyttic/diffusionpen-hebrew-handwriting, CC-BY-4.0):
149,952 lines in 491 writer styles, split so a style never crosses splits.
Each row carries a `cer` from an independent OCR pass, used here to drop
samples whose rendering is too degraded to read back.
"""

from dataclasses import dataclass

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from PIL import Image

from ..charset import Charset
from .transforms import LINE_HEIGHT, Augment, preprocess

DIFFUSIONPEN = "cyttic/diffusionpen-hebrew-handwriting"


@dataclass
class Batch:
    """A padded batch ready for `torch.nn.functional.ctc_loss`."""

    images: torch.Tensor        # (B, 1, H, W_max), zero-padded on the right
    widths: torch.Tensor        # (B,) true pixel width of each line
    targets: torch.Tensor       # (sum(target_lengths),) concatenated labels
    target_lengths: torch.Tensor
    texts: list[str]

    def to(self, device, non_blocking: bool = False) -> "Batch":
        return Batch(
            images=self.images.to(device, non_blocking=non_blocking),
            widths=self.widths,
            targets=self.targets.to(device, non_blocking=non_blocking),
            target_lengths=self.target_lengths,
            texts=self.texts,
        )


class LineDataset(Dataset):
    """Line images plus transcriptions, augmented when `train` is set."""

    def __init__(
        self,
        rows,
        charset: Charset,
        train: bool = True,
        aug_strength: float = 1.0,
        seed: int = 0,
        concat_prob: float = 0.45,
        max_concat: int = 3,
    ):
        self.rows = rows
        self.charset = charset
        self.train = train
        self.aug_strength = aug_strength
        self.seed = seed
        self.concat_prob = concat_prob if train else 0.0
        self.max_concat = max_concat
        self._augment: Augment | None = None
        self._rng: np.random.Generator | None = None

    def __len__(self) -> int:
        return len(self.rows)

    def _aug_for_worker(self) -> Augment:
        # Built lazily so each dataloader worker gets its own generator; sharing
        # one across forked workers would replay identical augmentations.
        if self._augment is None:
            info = torch.utils.data.get_worker_info()
            wid = info.id if info is not None else 0
            self._augment = Augment(
                np.random.default_rng(self.seed + 9973 * (wid + 1)),
                strength=self.aug_strength,
            )
        return self._augment

    def _rng_for_worker(self) -> np.random.Generator:
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            wid = info.id if info is not None else 0
            self._rng = np.random.default_rng(self.seed + 7919 * (wid + 1))
        return self._rng

    def __getitem__(self, index):
        """`index` is an int, or a group of ints to join into one long line.

        Grouping is decided by the sampler rather than here, because the batch's
        memory cost depends on the joined width and the sampler has to know it
        before it commits to a batch.
        """
        idxs = [int(index)] if isinstance(index, (int, np.integer)) else [int(i) for i in index]
        rows = [self.rows[i] for i in idxs]

        if len(rows) == 1:
            image, text = rows[0]["image"], rows[0]["text"]
        else:
            image, text = concat_rtl([r["image"] for r in rows], [r["text"] for r in rows])

        arr = self._aug_for_worker()(image) if self.train else preprocess(image)
        return torch.from_numpy(arr), self.charset.encode(text), text


def collate(items) -> Batch:
    """Pad a list of variable-width lines into one batch."""
    images, labels, texts = zip(*items)
    widths = torch.tensor([im.shape[-1] for im in images], dtype=torch.long)
    height = images[0].shape[-2]
    padded = torch.zeros(len(images), 1, height, int(widths.max()), dtype=torch.float32)
    for i, im in enumerate(images):
        padded[i, :, :, : im.shape[-1]] = im
    lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    flat = torch.tensor([i for l in labels for i in l], dtype=torch.long)
    return Batch(padded, widths, flat, lengths, list(texts))


class PixelBudgetSampler(torch.utils.data.Sampler):
    """Batch by total pixel area, not by a fixed line count.

    Line widths here span more than an order of magnitude -- 200 px to 2269 px
    at a 64 px height, before concatenation. A fixed batch size therefore has no
    stable memory cost: 32 short lines fit easily and 32 long ones exhaust an
    8 GB card, which is exactly how this first OOMed. Budgeting on
    `len(batch) * width_of_longest` instead keeps peak memory roughly flat and
    lets short-line batches be large, which is free throughput.

    The sampler also decides line concatenation, because the joined width has to
    be known before the batch is committed to.
    """

    def __init__(
        self,
        widths,
        budget: int = 20000,
        max_batch: int = 64,
        concat_prob: float = 0.0,
        max_concat: int = 3,
        gap: int = 24,
        max_width: int = 2400,
        shuffle: bool = True,
        seed: int = 0,
        pool: int = 4096,
        concat_max_index: int | None = None,
    ):
        self.widths = np.asarray(widths, dtype=np.int64)
        self.budget = budget
        self.max_batch = max_batch
        self.concat_prob = concat_prob
        self.max_concat = max_concat
        self.gap = gap
        self.max_width = max_width
        self.shuffle = shuffle
        self.seed = seed
        self.pool = pool
        # Only items below this index may be joined together. Concatenation
        # places the next line to the LEFT, which is right for Hebrew and wrong
        # for left-to-right scripts, so English real-ink lines must never be
        # joined to anything.
        self.concat_max_index = len(self.widths) if concat_max_index is None else concat_max_index
        self.epoch = 0
        self._cached_len: int | None = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _groups(self, rng):
        """Walk the shuffled indices, occasionally joining neighbours."""
        order = rng.permutation(len(self.widths)) if self.shuffle else np.arange(len(self.widths))
        groups, i = [], 0
        while i < len(order):
            n = 1
            if (
                self.concat_prob
                and order[i] < self.concat_max_index
                and rng.random() < self.concat_prob
            ):
                n = int(rng.integers(2, self.max_concat + 1))
            picks = order[i : i + n]
            # Drop any member that is not concat-eligible, rather than reordering
            # a script that does not read right to left.
            if len(picks) > 1:
                picks = picks[picks < self.concat_max_index]
                if len(picks) == 0:
                    picks = order[i : i + 1]
            width = int(self.widths[picks].sum() + self.gap * (len(picks) - 1))
            # Too wide once joined: fall back to the single line.
            if width > self.max_width and len(picks) > 1:
                picks, width = order[i : i + 1], int(self.widths[order[i]])
            groups.append((tuple(int(x) for x in picks), min(width, self.max_width)))
            i += len(picks)
        return groups

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        groups = self._groups(rng)

        batches = []
        for start in range(0, len(groups), self.pool):
            window = sorted(groups[start : start + self.pool], key=lambda g: g[1])
            batch, widest = [], 0
            for picks, width in window:
                nxt = max(widest, width)
                if batch and ((len(batch) + 1) * nxt > self.budget or len(batch) >= self.max_batch):
                    batches.append(batch)
                    batch, widest = [picks], width
                else:
                    batch.append(picks)
                    widest = nxt
            if batch:
                batches.append(batch)

        if self.shuffle:
            rng.shuffle(batches)
        self._cached_len = len(batches)
        yield from batches

    def __len__(self) -> int:
        if self._cached_len is not None:
            return self._cached_len
        # Estimate before the first pass: average width against the budget.
        mean_w = float(self.widths.mean()) * (1 + self.concat_prob)
        per_batch = max(1, min(self.max_batch, int(self.budget / max(mean_w, 1))))
        return max(1, len(self.widths) // per_batch)


def load_diffusionpen(split: str, max_cer: float | None = 0.5, limit: int | None = None):
    """Load a DiffusionPen split, dropping rows the quality gate flags.

    `max_cer` filters on the dataset's own per-row OCR check. The published set
    already drops anything above 0.5; tightening it trades volume for legibility.
    """
    from datasets import load_dataset

    ds = load_dataset(DIFFUSIONPEN, split=split)
    if max_cer is not None and "cer" in ds.column_names:
        ds = ds.filter(lambda c: c <= max_cer, input_columns="cer", num_proc=8)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def concat_rtl(images: list, texts: list[str], gap: int = 24, max_width: int = 1792):
    """Join short lines into one long line, right-to-left.

    DiffusionPen tops out at 72 characters per line, but 20% of the benchmark's
    lines are longer than that and the longest is 111 characters. Without this
    the model never sees a line the length of the ones it will be tested on.

    Hebrew is right-to-left, and these renders follow it: the logically first
    word sits at the *right* edge of the image. So joining "A" then "B" places
    B's pixels to the LEFT of A's. Getting this backwards would train the model
    on text in reverse order while the labels stayed forward.
    """
    from PIL import Image

    keep_imgs, keep_txts, total = [], [], 0
    for im, txt in zip(images, texts):
        width = im.width + (gap if keep_imgs else 0)
        if total + width > max_width and keep_imgs:
            break
        keep_imgs.append(im)
        keep_txts.append(txt)
        total += width

    if len(keep_imgs) == 1:
        return keep_imgs[0], keep_txts[0]

    height = max(im.height for im in keep_imgs)
    canvas = Image.new("L", (total, height), 255)
    # Paste in reverse: the last line logically is the leftmost visually.
    x = 0
    for im in reversed(keep_imgs):
        canvas.paste(im.convert("L"), (x, (height - im.height) // 2))
        x += im.width + gap
    return canvas, " ".join(keep_txts)


def image_widths(ds, cache: str | None = None, target_height: int | None = None) -> np.ndarray:
    """Width every image will have *after* preprocessing, without decoding pixels.

    Scaled to the model's line height rather than reported raw, because that is
    the width the batch actually pads to. The sources differ: DiffusionPen
    renders at 64 px tall, KHATT and IAM at 128, so their raw widths overstate
    the real cost by a factor of two and would push the sampler into batches
    half the size it could afford.

    The sampler budgets on width, so it needs real numbers rather than a
    characters-times-a-constant guess: measured px-per-character ranges from 20
    to 67 across this corpus, and under-estimating a batch's width is what
    turns a planned batch into an out-of-memory error. PIL reads only the
    header when asked for `.size`, so this is cheap; the result is cached
    because it never changes for a given split.
    """
    import io
    from pathlib import Path

    import datasets as hfds
    from PIL import Image

    from .transforms import LINE_HEIGHT, MAX_WIDTH, MIN_WIDTH

    target_height = target_height or LINE_HEIGHT

    if cache and Path(cache).exists():
        widths = np.load(cache)
        if len(widths) == len(ds):
            return widths

    raw = ds.cast_column("image", hfds.Image(decode=False))

    def width_of(record) -> int:
        # An undecoded image arrives either as embedded bytes or as a path on
        # disk, depending on how the dataset was built. Parquet-packed sets
        # (DiffusionPen) carry bytes; sets published as individual image files
        # (KHATT, IAM) carry only a path, and reading `bytes` there yields None.
        data = record.get("bytes")
        source = io.BytesIO(data) if data else record.get("path")
        if source is None:
            return MIN_WIDTH
        with Image.open(source) as im:
            w, h = im.size
        scaled = int(round(w * target_height / max(h, 1)))
        return max(MIN_WIDTH, min(scaled, MAX_WIDTH))

    widths = np.fromiter(
        (width_of(rec) for rec in raw["image"]), dtype=np.int64, count=len(ds)
    )
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, widths)
    return widths


class MixedLineDataset(Dataset):
    """DiffusionPen lines plus lines composed from real handwritten glyphs.

    The two sources are complementary and each is deficient alone. DiffusionPen
    has believable spacing, slant and letter joins but generated ink;
    glyph-composed lines have real ballpoint texture but mechanical spacing and
    no consistent hand within a line. Mixing exposes the model to both, so
    neither deficiency is a constant it can learn to rely on.

    Glyph lines are pre-composed at startup rather than on demand so their true
    widths are known before batching -- the pixel-budget sampler needs a real
    width per item, not an estimate.
    """

    def __init__(
        self,
        rows,
        glyph_items: list[tuple[np.ndarray, str]],
        charset: Charset,
        train: bool = True,
        aug_strength: float = 1.0,
        seed: int = 0,
        extra_sources: list | None = None,
    ):
        self.rows = rows
        self.glyph_items = glyph_items
        self.charset = charset
        self.train = train
        self.aug_strength = aug_strength
        self.seed = seed
        # Real handwriting in other scripts. Their labels mean nothing to a
        # Hebrew reader; the ink is the point, because stroke texture and how
        # ink sits on paper are what no synthetic source supplies.
        self.extra_sources = list(extra_sources or [])
        self.n_rows = len(rows)
        self.n_glyphs = len(glyph_items)
        self._augment: Augment | None = None

    def __len__(self) -> int:
        return self.n_rows + self.n_glyphs + sum(len(s) for s in self.extra_sources)

    def widths(self, row_widths: np.ndarray, extra_widths: list | None = None) -> np.ndarray:
        """Widths for every item, in the order `__getitem__` indexes them."""
        # An endless generator knows its widths without drawing anything; a
        # pre-composed list has to be measured. Asking the generator to
        # materialize every line here would defeat the point of having one.
        glyph_widths = getattr(self.glyph_items, "widths", None)
        if glyph_widths is None:
            glyph_widths = np.array(
                [g.shape[1] for g, _ in self.glyph_items], dtype=np.int64
            )
        parts = [
            np.asarray(row_widths, dtype=np.int64),
            np.asarray(glyph_widths, dtype=np.int64),
        ]
        parts.extend(np.asarray(w, dtype=np.int64) for w in (extra_widths or []))
        return np.concatenate(parts)

    def _aug_for_worker(self) -> Augment:
        # Rebuilt when the strength changes, so an augmentation ramp actually
        # reaches the workers. A pretrained encoder meets inputs far outside its
        # pretraining distribution if full-strength augmentation starts on step
        # one, and CTC answers that by collapsing to all-blank output.
        if self._augment is None or self._augment.strength != self.aug_strength:
            info = torch.utils.data.get_worker_info()
            wid = info.id if info is not None else 0
            self._augment = Augment(
                np.random.default_rng(self.seed + 9973 * (wid + 1)), strength=self.aug_strength
            )
        return self._augment

    def _fetch(self, index: int):
        if index < self.n_rows:
            row = self.rows[index]
            return row["image"], row["text"]

        index -= self.n_rows
        if index < self.n_glyphs:
            from PIL import Image as _Image

            image, text = self.glyph_items[index]
            return _Image.fromarray(image), text

        index -= self.n_glyphs
        for source in self.extra_sources:
            if index < len(source):
                row = source[int(index)]
                return row["image"], row["text"]
            index -= len(source)
        raise IndexError("index past the end of every source")

    def __getitem__(self, index):
        idxs = [int(index)] if isinstance(index, (int, np.integer)) else [int(i) for i in index]
        fetched = [self._fetch(i) for i in idxs]

        if len(fetched) == 1:
            image, text = fetched[0]
        else:
            image, text = concat_rtl([f[0] for f in fetched], [f[1] for f in fetched])

        arr = self._aug_for_worker()(image) if self.train else preprocess(image)
        return torch.from_numpy(arr), self.charset.encode(text), text


def build_glyph_lines(texts, count: int, seed: int = 0, split: str = "train"):
    """Pre-compose `count` lines from real HHD glyphs. Returns (array, text) pairs."""
    from .glyphs import GlyphBank, GlyphLineDataset

    bank = GlyphBank.load(split)
    source = GlyphLineDataset(bank, texts, seed=seed)
    rng = np.random.default_rng(seed)

    items: list[tuple[np.ndarray, str]] = []
    attempts = 0
    while len(items) < count and attempts < count * 4:
        attempts += 1
        image, text = source.sample(rng)
        if image is None or not text:
            continue
        items.append((np.asarray(image.convert("L"), dtype=np.uint8), text))
    return items


class EndlessGlyphLines:
    """Glyph-composed lines generated on demand, so the supply is unbounded.

    `build_glyph_lines` pre-composes into memory, which is why the count has
    always been small: 20,000 lines already costs gigabytes, and the arrays are
    inherited by every dataloader worker. Scaling the way TrOCR did, where the
    synthetic corpus is three orders of magnitude larger than this one, cannot
    be done by pre-composing anything. Twenty million lines at this height is
    roughly a terabyte.

    Composing per access removes the storage entirely and, as a side effect,
    removes repetition: a line is drawn fresh every time it is asked for, so
    the model never sees the same arrangement of glyphs twice no matter how
    many epochs it runs. The glyph inventory is still finite, so what grows
    without bound is the number of *arrangements*, not the number of hands.
    That is worth being precise about: this buys coverage of spacing, sequence
    and juxtaposition, and buys nothing at all in letterform diversity.

    The one thing the sampler will not tolerate is an unknown width. It budgets
    a batch by lines times the width of the widest, so it needs a real width per
    item before any pixels exist, and a guess that comes in low is what turns a
    planned batch into an out-of-memory error. So the widths are decided first,
    drawn from the distribution of genuinely composed lines, and each line is
    then rendered and resized to the width already promised for it. Resizing
    changes character density, which is not a side effect worth avoiding: it is
    the same axis the stretch augmentation already varies on purpose.
    """

    def __init__(self, bank, texts, count: int, seed: int = 0, calibration: int = 512):
        from .glyphs import GlyphLineDataset

        self.source = GlyphLineDataset(bank, texts, seed=seed)
        if not len(self.source):
            # Every text was rejected: too short, or written in characters the
            # glyph bank has no ink for. Saying so here beats the "high <= 0"
            # a sampler raises several frames later.
            raise RuntimeError(
                "no renderable texts: the glyph bank cannot draw any of them"
            )
        self.count = int(count)
        self.seed = seed
        self._rng = None

        # Measure the natural width distribution once, then sample from it.
        rng = np.random.default_rng(seed)
        measured = []
        for _ in range(max(32, calibration)):
            image, text = self.source.sample(rng)
            if image is not None and text:
                measured.append(image.width)
        if not measured:
            raise RuntimeError("the glyph bank produced no lines to calibrate on")
        self._widths = rng.choice(
            np.asarray(measured, dtype=np.int64), size=self.count, replace=True
        )

    def __len__(self) -> int:
        return self.count

    @property
    def widths(self) -> np.ndarray:
        """The widths promised to the sampler, before anything is drawn."""
        return self._widths

    def _rng_for_worker(self) -> np.random.Generator:
        # Fresh content every access, but seeded per worker so two workers never
        # walk the same sequence. Entropy is mixed in deliberately: the width is
        # already fixed, so the content is free to differ between epochs, and
        # that is the whole point of generating rather than storing.
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            wid = info.id if info is not None else 0
            self._rng = np.random.default_rng([self.seed, wid, os.getpid()])
        return self._rng

    def __getitem__(self, index):
        rng = self._rng_for_worker()
        target = int(self._widths[int(index) % self.count])
        image, text = self.source.sample(rng)
        if image is None or not text:
            # A bad draw must still honour its width, or the batch overruns.
            return np.full((LINE_HEIGHT, target), 255, dtype=np.uint8), ""
        if image.width != target:
            image = image.resize((max(8, target), image.height), Image.BICUBIC)
        return np.asarray(image.convert("L"), dtype=np.uint8), text


# Real handwriting in other scripts, used to teach the encoder what pen on
# paper looks like. Both are photographed/scanned modern handwriting rather
# than manuscript facsimiles, which is the closer match to this benchmark.
REAL_INK_SOURCES = {
    # Arabic: right-to-left and cursive, structurally the nearest script to
    # Hebrew among the large public handwriting corpora.
    "khatt": ("johnlockejrr/KHATT_v1.0_dataset", "train", 6000),
    "iam": ("Teklia/IAM-line", "train", 7000),
    "norhand3": ("Teklia/NorHand-v3-line", "train", 30000),
    "norhand2": ("Teklia/NorHand-v2-line", "train", 20000),
    "belfort": ("Teklia/Belfort-line", "train", 15000),
    "alcar": ("Teklia/HOME-Alcar-line", "train", 20000),
    "newseye": ("Teklia/NewsEye-Austrian-line", "train", 20000),
    # Chinese. Registered, but deliberately left out of ALL_REAL_INK below.
    # Including it takes the charset from 197 classes to 1,837, so 6 percent of
    # the data would claim 87 percent of the output layer. Capping the alphabet
    # instead is worse: CASIA's 300 most frequent characters cover only 68
    # percent of its text, so a third of every line would go unlabelled, and
    # partial labels are precisely what CTC cannot tolerate. Pass it explicitly
    # to --real-ink if you want it.
    "casia": ("Teklia/CASIA-HWDB2-line", "train", 20000),
    "himanis": ("Teklia/Himanis-line", "train", 15000),
    "rimes": ("Teklia/RIMES-2011-line", "train", 10000),
    "esposalles": ("Teklia/Esposalles-line", "train", 2328),
    "popp": ("Teklia/POPP-line", "train", 3835),
}

# Every source worth pretraining on. Excludes CASIA for the alphabet reason
# above; the claim that ink transfers across scripts is already carried by the
# Latin and Arabic corpora, which share no letters with Hebrew either.
ALL_REAL_INK = tuple(n for n in REAL_INK_SOURCES if n != "casia")


def load_real_ink(names=("khatt", "iam"), cap_override: int | None = None):
    """Load real-handwriting line datasets. Returns [(name, dataset), ...].

    A source that fails to load is skipped with a warning rather than killing
    the run: these are a supplement, and a night of training should not be lost
    to one unavailable repo.
    """
    from datasets import load_dataset

    loaded = []
    for name in names:
        if name not in REAL_INK_SOURCES:
            raise ValueError(f"unknown real-ink source {name!r}")
        repo, split, cap = REAL_INK_SOURCES[name]
        if cap_override is not None:
            cap = None if cap_override <= 0 else cap_override
        try:
            ds = load_dataset(repo, split=split)
        except Exception as exc:  # noqa: BLE001
            print(f"  skipping {name} ({repo}): {type(exc).__name__}: {exc}", flush=True)
            continue
        if cap is not None and len(ds) > cap:
            # Deterministic subsample, so a rerun trains on the same lines.
            ds = ds.shuffle(seed=0).select(range(cap))
        loaded.append((name, ds))
    return loaded
