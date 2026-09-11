"""Real Hebrew handwriting, cut into lines from the two public annotated corpora.

This project's central constraint has been that no public corpus of real modern
Hebrew cursive exists with line-level transcriptions, so every training image is
synthetic and the whole problem is the domain gap to photographs of real paper.
That statement is still true of *modern cursive*. It is not true of Hebrew
script in general, and the distinction turned out to matter.

Two annotated corpora of real Hebrew handwriting are openly published:

`pinkas`
    30 pages of community records from early modern Europe, c. 1500-1800,
    written by non-professional scribes in a mixture of Hebrew and Yiddish.
    PAGE XML with a line-level transcription per text line. CC-BY-4.0.
    https://zenodo.org/records/3569694

`biblia`
    202 pages of medieval Hebrew and Aramaic manuscripts from the BnF and the
    Vatican Library, in six script families (Ashkenazi, Byzantine, Italian,
    Oriental, Sephardi, Yemenite). ALTO 4.2 with baselines and transcriptions.
    CC-BY-NC-SA-4.0, which is *not* the licence the rest of this project uses.
    https://zenodo.org/records/5167263

Neither is the target script. Both are centuries older than the benchmark's
ballpoint-on-ruled-paper, and BiblIA in particular is formal square script
written by trained scribes, which is about as far from a modern hand as Hebrew
gets. They are included anyway, and the reason is the finding this project
already rests on: 56k lines of Arabic, English, Norwegian and French, in
alphabets the model cannot read, improved Hebrew line CER from 0.306 to 0.234.
What transferred there was ink rather than language. Real Hebrew ink should
transfer at least as well as real Latin ink, and it carries the letterforms too.

**Licence warning.** BiblIA is NonCommercial-ShareAlike. A model trained on it
cannot be released under CC-BY-4.0 without thinking about what the ShareAlike
term means for the weights. `--real-hebrew pinkas` selects only the CC-BY-4.0
corpus, and that is the default for anything intended for release.
"""

import io
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..normalize import normalize
from .transforms import LINE_HEIGHT, MAX_WIDTH as MAX_LINE_WIDTH

DATA_ROOT = Path("data/hebrew_real")

# Crop padding around a line polygon, as a fraction of the line's own height.
# Real hands overshoot their baselines and the annotation polygons are tight.
PAD = 0.12

# Odd, and small. See the note in _crop_polygon.
MASK_DILATION = 5

# Median pixels per character the benchmark presents at a 64 px line height,
# measured from its 225 gold lines. Every real-ink source is matched to it.
TARGET_PX_PER_CHAR = 22.1

MIN_TEXT = 2
MIN_WIDTH = 24
MIN_HEIGHT = 8


def _polygon(points: str) -> list[tuple[int, int]] | None:
    """PAGE gives "x,y x,y ...", ALTO gives "x y x y ...". Accept both."""
    raw = points.split()
    coords: list[tuple[int, int]] = []
    if raw and "," in raw[0]:
        for pair in raw:
            x, _, y = pair.partition(",")
            try:
                coords.append((int(float(x)), int(float(y))))
            except ValueError:
                return None
    else:
        if len(raw) < 6 or len(raw) % 2:
            return None
        try:
            values = [float(v) for v in raw]
        except ValueError:
            return None
        coords = [(int(values[i]), int(values[i + 1])) for i in range(0, len(values), 2)]
    return coords if len(coords) >= 3 else None


def _crop_polygon(image: Image.Image, polygon: list[tuple[int, int]]) -> Image.Image | None:
    """Cut one text line out, erasing whatever else falls in its bounding box.

    Cropping the bounding box alone is not enough and the failure is not subtle.
    These lines slant and curve, so the box around a line's polygon routinely
    contains a third of the line above and a third of the line below. Training
    on that pairs an image holding three lines of ink with a label describing
    one of them, which is the same defect as an augmentation that blanks the
    image and keeps the transcription: the loss still falls, and the model
    learns that most of the ink in front of it is not to be read.

    So everything outside the polygon is filled with the page's own background
    level rather than a constant, which keeps the paper tone continuous and
    leaves the augmentation free to add its own neighbour bleed at a strength
    it controls.
    """
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    height = y1 - y0
    pad = int(round(height * PAD))
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(image.width, x1 + pad), min(image.height, y1 + pad)
    if x1 - x0 < MIN_WIDTH or y1 - y0 < MIN_HEIGHT:
        return None

    box = (x0, y0, x1, y1)
    crop = image.crop(box)

    mask = Image.new("L", crop.size, 0)
    shifted = [(px - x0, py - y0) for px, py in polygon]
    ImageDraw.Draw(mask).polygon(shifted, fill=255)
    # Let the mask breathe by a few pixels, so ascenders and descenders the
    # annotator clipped are not sliced off mid-stroke. Deliberately small:
    # dilating by the full crop padding reaches far enough to pull the
    # neighbouring line back in, which is the thing the mask exists to remove.
    mask = mask.filter(ImageFilter.MaxFilter(size=MASK_DILATION))

    array = np.asarray(crop, dtype=np.uint8)
    keep = np.asarray(mask, dtype=np.uint8) > 0
    if keep.sum() < 0.05 * keep.size:
        return None
    # Background level taken from the page inside this crop, not a constant.
    background = int(np.percentile(array, 85))
    out = np.where(keep, array, np.uint8(background))

    # Crop again to what the mask actually kept. The polygon's bounding box is
    # much taller than the line, because a slanted line's box spans its whole
    # rise, and `preprocess` normalizes *image* height to 64 px rather than
    # letter height. Left as-is the line would occupy half the strip and be
    # squeezed to half the character size the model is used to.
    rows = np.flatnonzero(keep.any(axis=1))
    cols = np.flatnonzero(keep.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    out = out[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    if out.shape[1] < MIN_WIDTH or out.shape[0] < MIN_HEIGHT:
        return None
    return Image.fromarray(out, mode="L")


def _match_density(image: Image.Image, text: str, target: float,
                   rng: np.random.Generator | None = None) -> Image.Image:
    """Stretch a line horizontally so it carries `target` pixels per character.

    Sources disagree wildly on how tightly they pack characters once height is
    normalized to 64 px. Measured medians: the benchmark 22.1, DiffusionPen
    33.5, Pinkas 8.4. The augmentation's stretch is deliberately lopsided
    toward compression because it was tuned against DiffusionPen being too
    sparse; applied unchanged to a source that is already 2.6x too dense it
    pushes it further from the target rather than toward it.

    Correcting per source first means the augmentation varies around the right
    centre for all of them, which is the same reasoning that made stroke weight
    and vertical fill worth measuring rather than guessing.
    """
    if not text or image.width < 2:
        return image
    scaled_width = image.width * (LINE_HEIGHT / max(image.height, 1))
    current = scaled_width / len(text)
    if current <= 0:
        return image
    factor = target / current
    if rng is not None:
        factor *= float(rng.uniform(0.85, 1.18))
    factor = float(np.clip(factor, 0.4, 8.0))
    if abs(factor - 1.0) < 0.02:
        return image
    width = max(MIN_WIDTH, int(round(image.width * factor)))
    if width > MAX_LINE_WIDTH:
        width = MAX_LINE_WIDTH
    return image.resize((width, image.height), Image.BICUBIC)


def _usable(text: str) -> str | None:
    """Normalize and reject lines with nothing a Hebrew recognizer can learn."""
    text = normalize(re.sub(r"\s+", " ", text).strip())
    if len(text) < MIN_TEXT:
        return None
    hebrew = sum("א" <= c <= "ת" for c in text)
    return text if hebrew >= max(1, len(text.replace(" ", "")) * 0.5) else None


def _read_pinkas(archive: Path):
    """PAGE XML: every TextLine carries its own TextEquiv/Unicode."""
    with zipfile.ZipFile(archive) as z:
        pages = sorted(n for n in z.namelist() if n.lower().endswith(".xml"))
        for xml_name in pages:
            image_name = xml_name[: -len(".xml")] + ".jpg"
            if image_name not in z.namelist():
                continue
            root = ET.fromstring(z.read(xml_name))
            ns = {"p": root.tag.split("}")[0].strip("{")}
            image = Image.open(io.BytesIO(z.read(image_name))).convert("L")

            for line in root.findall(".//p:TextLine", ns):
                unicode_el = line.find("p:TextEquiv/p:Unicode", ns)
                if unicode_el is None or not unicode_el.text:
                    continue
                text = _usable(unicode_el.text)
                coords = line.find("p:Coords", ns)
                if text is None or coords is None:
                    continue
                polygon = _polygon(coords.get("points", ""))
                if polygon is None:
                    continue
                crop = _crop_polygon(image, polygon)
                if crop is not None:
                    yield crop, text


def _read_biblia(archive: Path):
    """ALTO 4.2: a TextLine carries a Shape/Polygon and its String elements."""
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        images = {
            Path(n).stem: n for n in names
            if n.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))
        }
        for xml_name in sorted(n for n in names if n.lower().endswith(".xml")):
            stem = Path(xml_name).stem
            if stem not in images:
                continue
            try:
                root = ET.fromstring(z.read(xml_name))
            except ET.ParseError:
                continue
            namespace = root.tag.split("}")[0].strip("{") if "}" in root.tag else ""
            tag = (lambda name: f"{{{namespace}}}{name}") if namespace else (lambda name: name)

            try:
                image = Image.open(io.BytesIO(z.read(images[stem]))).convert("L")
            except Exception:
                continue

            for line in root.iter(tag("TextLine")):
                words = [
                    el.get("CONTENT", "") for el in line.iter(tag("String"))
                ]
                text = _usable(" ".join(w for w in words if w))
                if text is None:
                    continue

                polygon = None
                for shape in line.iter(tag("Polygon")):
                    polygon = _polygon(shape.get("POINTS", ""))
                    if polygon:
                        break
                if polygon is None and line.get("WIDTH"):
                    x = int(float(line.get("HPOS", 0)))
                    y = int(float(line.get("VPOS", 0)))
                    w = int(float(line.get("WIDTH", 0)))
                    h = int(float(line.get("HEIGHT", 0)))
                    polygon = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
                if polygon is None:
                    continue

                crop = _crop_polygon(image, polygon)
                if crop is not None:
                    yield crop, text


SOURCES = {
    "pinkas": ("pinkas.zip", _read_pinkas, "CC-BY-4.0"),
    "biblia": ("biblia.zip", _read_biblia, "CC-BY-NC-SA-4.0"),
}

# Everything whose licence allows the same release terms as the rest of the
# project. `--real-hebrew all` adds BiblIA and its NonCommercial clause with it.
RELEASE_SAFE = ("pinkas",)


def _cache_path(root: Path, name: str, match_density: bool, seed: int) -> Path:
    tag = f"{name}-h{LINE_HEIGHT}-{'d' if match_density else 'raw'}-s{seed}"
    return root / "cache" / f"{tag}.npz"


def _store(rows: list[dict], path: Path) -> None:
    """Cache crops as one compressed archive of 64 px strips."""
    path.parent.mkdir(parents=True, exist_ok=True)
    flat, widths, texts = [], [], []
    for row in rows:
        arr = np.asarray(row["image"], dtype=np.uint8)
        flat.append(arr.reshape(-1))
        widths.append(arr.shape[1])
        texts.append(row["text"])
    np.savez_compressed(
        path,
        pixels=np.concatenate(flat) if flat else np.zeros(0, np.uint8),
        widths=np.asarray(widths, dtype=np.int64),
        texts=np.asarray(texts, dtype=object),
    )


def _restore(path: Path) -> list[dict]:
    data = np.load(path, allow_pickle=True)
    pixels, widths, texts = data["pixels"], data["widths"], data["texts"]
    rows, offset = [], 0
    for width, text in zip(widths, texts):
        size = LINE_HEIGHT * int(width)
        arr = pixels[offset : offset + size].reshape(LINE_HEIGHT, int(width))
        offset += size
        rows.append({"image": Image.fromarray(arr, mode="L"), "text": str(text)})
    return rows


def load_hebrew_ink(
    sources=RELEASE_SAFE,
    root: Path | str = DATA_ROOT,
    limit: int | None = None,
    match_density: bool = True,
    seed: int = 0,
    cache: bool = True,
):
    """Line crops and transcriptions from the real Hebrew corpora.

    Returns rows shaped like the synthetic ones, `{"image": ..., "text": ...}`,
    so they drop straight into the training mixture. With `match_density` the
    crops are stretched to the benchmark's characters-per-pixel first; see
    `_match_density` for why that is not optional.

    Crops are stored at the model's own line height. Cutting 10,219 polygons out
    of 162 full-resolution manuscript photographs takes minutes and the result
    holds gigabytes at source resolution, which every training run would pay
    again and every dataloader worker would inherit. `preprocess` normalizes to
    64 px anyway, so nothing is lost by doing it once and caching the result.
    """
    root = Path(root)
    rng = np.random.default_rng(seed)
    rows = []
    for name in sources:
        if name not in SOURCES:
            raise ValueError(f"unknown Hebrew corpus {name!r}, expected {sorted(SOURCES)}")
        filename, reader, licence = SOURCES[name]
        archive = root / filename
        if not archive.exists():
            print(f"  {name}: {archive} not present, skipping", flush=True)
            continue

        cached = _cache_path(root, name, match_density, seed)
        if cache and cached.exists():
            found = _restore(cached)
            rows.extend(found)
            print(f"  {name}: {len(found)} real Hebrew lines ({licence}, cached)", flush=True)
            if limit is not None and len(rows) >= limit:
                return rows[:limit]
            continue

        produced = []
        for image, text in reader(archive):
            if match_density:
                image = _match_density(image, text, TARGET_PX_PER_CHAR, rng)
            scale = LINE_HEIGHT / max(image.height, 1)
            width = max(MIN_WIDTH, min(int(round(image.width * scale)), MAX_LINE_WIDTH))
            produced.append({
                "image": image.resize((width, LINE_HEIGHT), Image.BICUBIC),
                "text": text,
            })
        if cache:
            _store(produced, cached)
        rows.extend(produced)
        print(f"  {name}: {len(produced)} real Hebrew lines ({licence})", flush=True)
        if limit is not None and len(rows) >= limit:
            return rows[:limit]
    return rows


class HebrewInkSource:
    """The real Hebrew rows, presented the way the training loop expects.

    `MixedLineDataset` indexes its extra sources positionally and the charset
    builder asks them for a `text` column, which is the Hugging Face dataset
    interface. This is the smallest object that satisfies both, so real Hebrew
    drops into the mixture beside the foreign real-ink corpora without either
    side needing to know the difference.
    """

    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index):
        if index == "text":
            return [r["text"] for r in self.rows]
        if index == "image":
            return [r["image"] for r in self.rows]
        return self.rows[int(index)]

    @property
    def column_names(self):
        return ["image", "text"]

    def widths(self, target_height: int = LINE_HEIGHT) -> "np.ndarray":
        """Post-preprocess widths, which is what the pixel budget is spent on."""
        out = []
        for row in self.rows:
            image = row["image"]
            scale = target_height / max(image.height, 1)
            out.append(min(int(round(image.width * scale)), MAX_LINE_WIDTH))
        return np.asarray(out, dtype=np.int64)
