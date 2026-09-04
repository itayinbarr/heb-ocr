"""Line segmentation for full-page mode.

The benchmark's ten pages are photographs of handwritten notebooks: ruled
paper, spiral bindings, torn edges, dark table backgrounds, perspective skew,
mixed pen colours, and on three pages a layout that is not a column of text at
all (a mind map, a boxed diagram, a two-column table).

The design follows from how the mode is scored. The leaderboard ranks full-page
by *word coverage*, which is order-independent -- "segmentation and
reading-order differences don't count against it". So this segmenter is tuned
for recall of text regions, not for a perfect reading order. Missing a line
costs coverage; ordering it wrongly costs only the secondary page-CER number.

Every threshold here is derived from the image's own measured stroke size
rather than being a pixel constant, so nothing is fitted to these ten pages.
That matters: they are the test set, and hand-tuning constants against them
would quietly turn the benchmark into a training set.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass
class LineBox:
    """One detected text line, in page pixel coordinates."""

    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return self.w * self.h

    def crop(self, image: Image.Image, pad: float = 0.12) -> Image.Image:
        """Crop with a little vertical padding for ascenders and descenders."""
        py = int(self.h * pad)
        px = int(self.h * 0.05)
        return image.crop(
            (
                max(self.x - px, 0),
                max(self.y - py, 0),
                min(self.x + self.w + px, image.width),
                min(self.y + self.h + py, image.height),
            )
        )


def _flatten(gray: np.ndarray) -> np.ndarray:
    """Remove the lighting gradient a phone camera leaves across a page."""
    sigma = max(gray.shape) / 30.0
    background = cv2.GaussianBlur(gray, (0, 0), sigma)
    flat = gray.astype(np.float32) / np.maximum(background.astype(np.float32), 1.0)
    return np.clip(flat * 128.0, 0, 255).astype(np.uint8)


def binarize(image: Image.Image, max_side: int = 2000) -> tuple[np.ndarray, float]:
    """Page -> binary ink mask (255 = ink), plus the scale it was worked at.

    Downscaling first is not just for speed: at full resolution the adaptive
    threshold window sits inside single pen strokes and shatters them.
    """
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    scale = min(1.0, max_side / max(gray.shape))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    flat = _flatten(gray)
    block = max(15, (min(flat.shape) // 40) | 1)  # odd, and scaled to the page
    binary = cv2.adaptiveThreshold(
        flat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 12
    )
    # Speckle removal: a lone pixel is sensor noise, never ink.
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return binary, scale


def _oriented_line_kernel(length: int, angle_deg: float) -> np.ndarray:
    """A one-pixel-wide straight line of `length`, drawn at `angle_deg`."""
    drop = int(round(np.tan(np.radians(angle_deg)) * (length - 1)))
    height = abs(drop) + 1
    kernel = np.zeros((height, length), np.uint8)
    y0 = 0 if drop >= 0 else height - 1
    cv2.line(kernel, (0, y0), (length - 1, y0 + drop), 1, 1)
    return kernel


def remove_rules(binary: np.ndarray, pitch: float, angles=(-2.0, -1.0, -0.4, 0.0, 0.4, 1.0, 2.0)) -> np.ndarray:
    """Erase printed rules -- both the horizontal lines and the margin rules.

    Must run *after* deskewing. A ruled line drifts vertically across the page
    when the photo is even slightly rotated, so a long flat kernel does not fit
    inside it and the opening finds nothing; that failure is what fused every
    text line into one blob on the first attempt.

    Only *thin* long runs are removed. A long run that is also thick is a
    deliberate pen stroke -- a table border or an underline -- and erasing it
    would punch a hole through the writing that sits on it.
    """
    length = max(40, int(pitch * 2.0))
    thickness = max(3, int(pitch * 0.28))

    # A photographed page is never perfectly flat or perfectly square to the
    # lens, so its rules sag and tilt by a degree or two. A single flat kernel
    # finds none of them -- measured on this benchmark, it removed 0.0% of the
    # ink on page 1 -- so the opening is repeated over a small fan of angles and
    # the results unioned.
    horizontal = np.zeros_like(binary)
    for angle in angles:
        horizontal = cv2.bitwise_or(
            horizontal, cv2.morphologyEx(binary, cv2.MORPH_OPEN, _oriented_line_kernel(length, angle))
        )
    vertical = np.zeros_like(binary)
    for angle in angles:
        kernel = np.ascontiguousarray(_oriented_line_kernel(length, angle).T)
        vertical = cv2.bitwise_or(vertical, cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel))

    out = binary
    for runs, thin_kernel in ((horizontal, (1, thickness)), (vertical, (thickness, 1))):
        # A long run that is also thick is a pen stroke -- a table border or an
        # underline -- and cutting it would punch through the writing on it.
        thick = cv2.morphologyEx(
            runs, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, thin_kernel)
        )
        rules = cv2.subtract(runs, thick)
        out = cv2.subtract(out, cv2.dilate(rules, np.ones((3, 3), np.uint8)))
    return out


def estimate_line_pitch(binary: np.ndarray) -> float:
    """Distance from one text line to the next, by autocorrelating row ink.

    Far more robust than a median connected-component height, which on these
    pages is dominated by speckles, vowel-sized marks and rule fragments and
    came out at 13 px when the true line spacing was 73. Ruled writing is
    strongly periodic vertically, so the first real peak of the projection
    profile's autocorrelation is the line pitch.
    """
    profile = binary.sum(axis=1, dtype=np.float64)
    profile = profile - profile.mean()
    if not np.any(profile):
        return 30.0

    corr = np.correlate(profile, profile, mode="full")[len(profile) - 1 :]
    lo = max(8, int(len(profile) * 0.004))       # below this is stroke texture
    hi = min(len(corr) - 1, int(len(profile) * 0.15))  # above this is page structure
    if hi <= lo + 2:
        return 30.0

    window = corr[lo:hi]
    # First local maximum that stands above its neighbours, not just the argmax:
    # the argmax can land on a harmonic when a page has paragraph gaps.
    peaks = [
        i for i in range(1, len(window) - 1)
        if window[i] > window[i - 1] and window[i] >= window[i + 1] and window[i] > 0
    ]
    return float(lo + (peaks[0] if peaks else int(np.argmax(window))))


def estimate_skew(binary: np.ndarray, limit: float = 6.0, step: float = 0.5) -> float:
    """Find the rotation that makes horizontal projection most peaked.

    When lines are level, the row-sum profile alternates hard between text rows
    and blank gaps, so its variance is maximal. Searching that variance is more
    robust on handwriting than Hough lines, which lock onto the ruled paper.
    """
    small = cv2.resize(binary, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    best_angle, best_score = 0.0, -1.0
    h, w = small.shape
    center = (w / 2, h / 2)
    for angle in np.arange(-limit, limit + step, step):
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(small, m, (w, h), flags=cv2.INTER_NEAREST)
        profile = rotated.sum(axis=1, dtype=np.float64)
        score = float(np.var(np.diff(profile)))
        if score > best_score:
            best_angle, best_score = float(angle), score
    return best_angle


def _rotate(image: np.ndarray, angle: float, fill: int = 0) -> np.ndarray:
    if abs(angle) < 1e-3:
        return image
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        image, m, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=fill
    )


def find_lines(binary: np.ndarray, pitch: float) -> list[LineBox]:
    """Group ink into text lines by run-length smearing.

    Smearing horizontally by roughly a word-gap joins the characters of a line
    into one blob while leaving separate lines apart; a small vertical smear
    then bridges the gap between a letter body and its dots. Connected
    components of the result are the lines. Both extents are fractions of the
    line pitch, so the same code works on a page shot from 20 cm and one shot
    from a metre.
    """
    smear_x = max(8, int(pitch * 0.55))
    # Kept deliberately small: vertical smearing is what welds a line to the one
    # below it. `split_tall` recovers lines that are joined by their descenders.
    smear_y = max(1, int(pitch * 0.05))
    smeared = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (smear_x, smear_y)))
    smeared = cv2.morphologyEx(
        smeared, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (smear_x, 1))
    )

    n, _, stats, _ = cv2.connectedComponentsWithStats(smeared, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = (
            stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
            stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT],
            stats[i, cv2.CC_STAT_AREA],
        )
        # A line is wider than one character and not a blot or a page edge.
        if w < pitch * 0.8 or h < pitch * 0.25:
            continue
        if h > pitch * 12.0:  # spans a whole column: a box outline, not text
            continue
        if area < pitch * pitch * 0.12:
            continue
        boxes.append(LineBox(int(x), int(y), int(w), int(h)))
    return boxes


def split_tall(binary: np.ndarray, boxes: list[LineBox], pitch: float) -> list[LineBox]:
    """Cut components that swallowed more than one text line.

    In dense cursive the descenders of one line reach into the next, so the two
    are genuinely connected ink and no amount of careful smearing separates
    them. But the *amount* of ink per row still dips between them, so a
    component roughly N pitches tall is split at the N-1 quietest rows near
    where the boundaries are expected. Searching only a narrow window around
    each expected boundary stops the cut from wandering into the middle of a
    line, where a short row of text would also look like a minimum.
    """
    out: list[LineBox] = []
    for box in boxes:
        n = int(round(box.h / pitch))
        if n <= 1:
            out.append(box)
            continue

        roi = binary[box.y : box.y + box.h, box.x : box.x + box.w]
        profile = roi.sum(axis=1, dtype=np.float64)
        smooth = cv2.GaussianBlur(profile.reshape(-1, 1), (1, 5), 0).ravel()

        cuts = [0]
        for k in range(1, n):
            centre = int(k * box.h / n)
            lo = max(cuts[-1] + 2, centre - int(pitch * 0.35))
            hi = min(box.h - 2, centre + int(pitch * 0.35))
            if hi > lo:
                cuts.append(lo + int(np.argmin(smooth[lo:hi])))
        cuts.append(box.h)

        for top, bottom in zip(cuts, cuts[1:]):
            band = roi[top:bottom]
            rows = np.where(band.sum(axis=1) > 0)[0]
            cols = np.where(band.sum(axis=0) > 0)[0]
            if len(rows) == 0 or len(cols) == 0:
                continue
            # Tighten onto the actual ink so the crop is not mostly blank paper.
            out.append(
                LineBox(
                    x=box.x + int(cols[0]),
                    y=box.y + top + int(rows[0]),
                    w=int(cols[-1] - cols[0] + 1),
                    h=int(rows[-1] - rows[0] + 1),
                )
            )
    return out


def merge_overlapping(boxes: list[LineBox], pitch: float) -> list[LineBox]:
    """Join fragments of one line that the smear left separate.

    Two boxes belong together when they occupy the same band vertically and sit
    within a word-gap horizontally -- the usual case being a line broken across
    a wide space or interrupted by a ruled line the cleanup missed.
    """
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: (b.y, b.x))
    merged: list[LineBox] = []
    for box in boxes:
        placed = False
        for i, other in enumerate(merged):
            v_overlap = min(box.y + box.h, other.y + other.h) - max(box.y, other.y)
            shorter = min(box.h, other.h)
            h_gap = max(box.x, other.x) - min(box.x + box.w, other.x + other.w)
            if v_overlap > shorter * 0.5 and h_gap < pitch * 1.2:
                x0 = min(box.x, other.x)
                y0 = min(box.y, other.y)
                x1 = max(box.x + box.w, other.x + other.w)
                y1 = max(box.y + box.h, other.y + other.h)
                merged[i] = LineBox(x0, y0, x1 - x0, y1 - y0)
                placed = True
                break
        if not placed:
            merged.append(box)
    return merged


def reading_order(boxes: list[LineBox], pitch: float) -> list[LineBox]:
    """Sort into Hebrew reading order: rows top to bottom, then right to left.

    Boxes whose vertical centres fall within roughly one line height are treated
    as the same row -- side-by-side cells of a table, or a heading beside a
    marginal note -- and within a row the rightmost comes first, because Hebrew
    is read right to left.
    """
    if not boxes:
        return []
    ordered = sorted(boxes, key=lambda b: b.y + b.h / 2)
    rows: list[list[LineBox]] = [[ordered[0]]]
    for box in ordered[1:]:
        centre = box.y + box.h / 2
        row_centre = np.mean([b.y + b.h / 2 for b in rows[-1]])
        if abs(centre - row_centre) < pitch * 0.45:
            rows[-1].append(box)
        else:
            rows.append([box])
    return [b for row in rows for b in sorted(row, key=lambda b: -(b.x + b.w))]


def segment_page(image: Image.Image, max_side: int = 2000, deskew: bool = True):
    """Page image -> (line crops in reading order, their boxes, skew angle).

    Order matters. Deskew comes first, because rule removal needs the rules to
    be genuinely horizontal, and line-pitch estimation needs the rows of text to
    stack cleanly in the projection profile. Doing it the other way round leaves
    the rules in place, and the smear then welds every line on the page into one
    component.

    Crops come from the *deskewed original* rather than the binary mask, so the
    recognizer sees real greyscale ink and applies its own preprocessing.
    """
    binary, scale = binarize(image, max_side=max_side)

    angle = estimate_skew(binary) if deskew else 0.0
    if abs(angle) > 1e-3:
        binary = _rotate(binary, angle)
        work_image = image.rotate(angle, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
    else:
        work_image = image

    pitch = estimate_line_pitch(binary)
    binary = remove_rules(binary, pitch)
    pitch = estimate_line_pitch(binary)

    boxes = find_lines(binary, pitch)
    boxes = split_tall(binary, boxes, pitch)
    boxes = merge_overlapping(boxes, pitch)
    boxes = reading_order(boxes, pitch)

    # Map boxes from the downscaled working image back to page coordinates.
    inv = 1.0 / scale if scale > 0 else 1.0
    full = [
        LineBox(int(b.x * inv), int(b.y * inv), int(b.w * inv), int(b.h * inv))
        for b in boxes
    ]
    return [b.crop(work_image) for b in full], full, angle
