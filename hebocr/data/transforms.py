"""Line-image preprocessing and augmentation.

The whole model trains on synthetic handwriting (clean, evenly lit, white
background, rendered at a fixed 64 px) and is evaluated on photographs of real
paper (shadows, page texture, JPEG artifacts, ruled lines, phone-camera
white balance, line crops from 41 to 325 px tall). Nothing bridges that gap
except preprocessing that erases the easy differences and augmentation that
manufactures the hard ones.

`preprocess` runs identically at train and test time. `Augment` runs at train
time only.
"""

import cv2
import numpy as np
from PIL import Image

LINE_HEIGHT = 64

# The benchmark's longest line needs ~2433 px once scaled to a 64 px height
# (111 characters, aspect ratio 38:1). A tighter cap silently truncates the
# right-hand end of a fifth of the test set, so the ceiling is set above the
# real maximum rather than at a round number.
MAX_WIDTH = 2560
MIN_WIDTH = 16


def _to_gray(image) -> np.ndarray:
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("L"), dtype=np.uint8)
    else:
        arr = np.asarray(image)
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        arr = arr.astype(np.uint8)
    return arr


def ink_fraction(gray: np.ndarray) -> float:
    """Share of pixels meaningfully darker than the page.

    Used as a safety check on augmentation. An augmented image that has lost
    almost all its ink still carries its full transcription as a label, and
    training a CTC model to read text out of blank paper actively teaches it to
    hallucinate.
    """
    background = float(np.median(np.concatenate([gray[0], gray[-1]])))
    return float((gray.astype(np.float32) < background - 25).mean())


def _contrast_normalize(gray: np.ndarray) -> np.ndarray:
    """Flatten uneven lighting, then stretch to full range.

    Synthetic lines already sit on a clean white ground; photographed lines do
    not. Dividing by a heavily blurred copy of the image removes the low
    frequency lighting gradient and leaves the ink, which puts both domains on
    comparable footing before the model ever sees them.
    """
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(gray.shape[0] / 8, 3))
    flat = np.divide(
        gray.astype(np.float32),
        np.maximum(blur.astype(np.float32), 1e-3),
        dtype=np.float32,
    )
    lo, hi = np.percentile(flat, 2), np.percentile(flat, 98)
    if hi - lo < 1e-6:
        return np.full_like(gray, 255)
    return np.clip((flat - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def preprocess(image, height: int = LINE_HEIGHT, max_width: int = MAX_WIDTH) -> np.ndarray:
    """Line image -> float32 array in [0, 1], shape (1, height, width).

    Ink is high-valued (near 1) and background is near 0, so zero padding a
    batch to a common width adds background rather than ink.
    """
    gray = _to_gray(image)
    gray = _contrast_normalize(gray)

    h, w = gray.shape
    scale = height / max(h, 1)
    new_w = int(round(w * scale))
    new_w = max(MIN_WIDTH, min(new_w, max_width))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    gray = cv2.resize(gray, (new_w, height), interpolation=interp)

    arr = 1.0 - gray.astype(np.float32) / 255.0  # invert: ink -> high
    return arr[None, :, :]


class Augment:
    """Randomized geometry, stroke and photometric distortion for training.

    Ordered deliberately: geometry first (on the raw crop, where interpolation
    is cheapest), then stroke thickness, then the photometric effects that
    imitate a camera.
    """

    def __init__(self, rng: np.random.Generator | None = None, strength: float = 1.0):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.strength = strength

    def _maybe(self, p: float) -> bool:
        return self.rng.random() < p * self.strength

    def _shear(self, gray: np.ndarray) -> np.ndarray:
        """Slant, imitating writer-to-writer differences in hand angle."""
        h, w = gray.shape
        k = self.rng.uniform(-0.35, 0.35) * self.strength
        pad = int(abs(k) * h) + 1
        m = np.float32([[1, k, -k * h / 2 if k > 0 else 0], [0, 1, 0]])
        return cv2.warpAffine(
            gray, m, (w + pad, h), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def _rotate(self, gray: np.ndarray) -> np.ndarray:
        """Small rotation: the benchmark's own crops are visibly non-level."""
        h, w = gray.shape
        angle = self.rng.uniform(-2.5, 2.5) * self.strength
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(
            gray, m, (w, h), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def _perspective(self, gray: np.ndarray) -> np.ndarray:
        """Projective warp: the page photographed from an angle.

        Every benchmark page is a phone photo of paper lying on a desk, so the
        text plane is never parallel to the sensor. Shear and rotation are both
        affine and cannot produce the converging baselines and varying letter
        height across a line that a real oblique photo does.
        """
        h, w = gray.shape
        dy = h * 0.12 * self.strength
        dx = w * 0.01 * self.strength
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([
            [self.rng.uniform(-dx, dx), self.rng.uniform(-dy, dy)],
            [w + self.rng.uniform(-dx, dx), self.rng.uniform(-dy, dy)],
            [w + self.rng.uniform(-dx, dx), h + self.rng.uniform(-dy, dy)],
            [self.rng.uniform(-dx, dx), h + self.rng.uniform(-dy, dy)],
        ])
        matrix = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(
            gray, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )

    def _bleed_through(self, gray: np.ndarray) -> np.ndarray:
        """Faint ink showing through from the reverse side of the page.

        Visible on several benchmark pages, where the paper is thin enough that
        writing on the back shows as a pale mirrored ghost. Mirrored because
        that is what seeing ink through paper does, and faint so it reads as
        background the model must ignore rather than text it should transcribe.
        """
        ghost = np.fliplr(gray)
        shift = int(self.rng.integers(-gray.shape[0] // 3, gray.shape[0] // 3 + 1))
        ghost = np.roll(ghost, shift, axis=0)
        # Subtract a fraction of the ghost's *ink*, rather than blending the two
        # images. Blending scales the background down as well, which darkens the
        # whole crop and makes the ghost read as genuine ink -- measured, it
        # nearly doubled the ink fraction. Ink showing through paper darkens the
        # page by only a few tens of grey levels, so the strength is capped well
        # below the threshold at which anything counts as writing.
        strength = self.rng.uniform(0.03, 0.09)
        ghost_ink = (255.0 - ghost.astype(np.float32)) * strength
        return np.clip(gray.astype(np.float32) - ghost_ink, 0, 255).astype(np.uint8)

    def _elastic(self, gray: np.ndarray) -> np.ndarray:
        """Local warping -- the standard stand-in for handwriting variability."""
        h, w = gray.shape
        alpha = self.rng.uniform(3.0, 9.0) * self.strength
        sigma = self.rng.uniform(4.0, 7.0)
        dx = cv2.GaussianBlur(self.rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur(self.rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigma) * alpha
        yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
        return cv2.remap(
            gray, (xx + dx).astype(np.float32), (yy + dy).astype(np.float32),
            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )

    def _stretch(self, gray: np.ndarray) -> np.ndarray:
        """Horizontal scaling: how tightly or loosely a writer spaces letters.

        The range is deliberately lopsided toward compression. Measured at a
        64 px line height, DiffusionPen renders about 34.5 px per character
        while the benchmark's real hands average about 20 -- real writing is
        roughly 0.6x as wide per character. A symmetric range around 1.0 would
        never show the model text at the density it is actually tested on.
        """
        h, w = gray.shape
        f = self.rng.uniform(0.5, 1.3)
        return cv2.resize(gray, (max(MIN_WIDTH, int(w * f)), h), interpolation=cv2.INTER_LINEAR)

    def _stroke(self, gray: np.ndarray) -> np.ndarray:
        """Thicken or thin the ink -- pen width and writing pressure.

        Biased hard toward thinning. Measured on 64 px-high lines, DiffusionPen
        draws 5-8 px strokes while the benchmark's ballpoint hands draw 1-2 px;
        a symmetric jitter of a pixel or two never crosses that distance.
        """
        before = ink_fraction(gray)
        if self.rng.random() >= 0.72:
            # Thicken: safe, it can only add ink.
            return cv2.erode(gray, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))

        # Thin, largest kernel first, and stop at the first one that leaves the
        # writing legible. A 6 px kernel erases a 2 px stroke completely.
        for k in sorted(self.rng.integers(2, 7, size=1).tolist(), reverse=True):
            for size in range(k, 1, -1):
                thinned = cv2.dilate(
                    gray, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
                )
                if ink_fraction(thinned) >= before * 0.25:
                    return thinned
        return gray

    def _vertical_fit(self, gray: np.ndarray) -> np.ndarray:
        """Shrink the writing inside its crop and pad with paper.

        A synthetic line is rendered edge to edge; a real crop is a bounding box
        drawn around handwriting with slack for ascenders and descenders, so the
        ink covers roughly half the crop's height. Since `preprocess` rescales
        everything to 64 px, that difference becomes a difference in apparent
        letter size -- which the model would otherwise never have seen.
        """
        h, w = gray.shape
        scale = self.rng.uniform(0.45, 1.0)
        new_h = max(8, int(h * scale))
        shrunk = cv2.resize(gray, (w, new_h), interpolation=cv2.INTER_AREA)

        background = int(np.median(np.concatenate([gray[0], gray[-1]])))
        pad = h - new_h
        top = int(self.rng.uniform(0.25, 0.75) * pad)
        return cv2.copyMakeBorder(
            shrunk, top, pad - top, 0, 0, cv2.BORDER_CONSTANT, value=background
        )

    def _neighbour_bleed(self, gray: np.ndarray) -> np.ndarray:
        """Bleed a fragment of an adjacent text line into the top or bottom edge.

        Real line crops on a densely written page clip the descenders of the line
        above and the ascenders of the line below -- visible on most of the
        benchmark's crops. Trained without it, the model has never had to ignore
        ink that is not part of its line.
        """
        h, w = gray.shape
        band = int(self.rng.uniform(0.10, 0.22) * h)
        if band < 3:
            return gray

        # A shifted, mirrored slice of this same line stands in for the
        # neighbour: the right stroke statistics, none of the real transcription.
        source = gray[int(h * 0.3) : int(h * 0.3) + band, :]
        if source.shape[0] < band:
            return gray
        fragment = np.fliplr(source)
        shift = int(self.rng.integers(0, max(1, w // 3)))
        fragment = np.roll(fragment, shift, axis=1)

        out = gray.copy()
        if self.rng.random() < 0.5:
            out[:band, :] = np.minimum(out[:band, :], fragment)
        else:
            out[h - band :, :] = np.minimum(out[h - band :, :], fragment)
        return out

    def _paper(self, gray: np.ndarray) -> np.ndarray:
        """Impose a low-frequency background: page texture and shadow gradients."""
        h, w = gray.shape
        noise = self.rng.normal(0, 1, (h, w)).astype(np.float32)
        texture = cv2.GaussianBlur(noise, (0, 0), self.rng.uniform(6, 20))
        texture /= max(np.abs(texture).max(), 1e-6)
        amount = self.rng.uniform(10, 45) * self.strength
        return np.clip(gray.astype(np.float32) + texture * amount, 0, 255).astype(np.uint8)

    def _ruled_line(self, gray: np.ndarray) -> np.ndarray:
        """Draw a faint ruled line -- most real pages in this corpus have them."""
        h, w = gray.shape
        out = gray.copy()
        y = int(self.rng.uniform(0.6, 0.95) * h)
        thickness = int(self.rng.integers(1, 3))
        value = int(self.rng.integers(120, 200))
        y2 = int(np.clip(y + self.rng.normal(0, 2), 0, h - 1))
        cv2.line(out, (0, y), (w - 1, y2), value, thickness)
        return cv2.addWeighted(gray, 0.35, out, 0.65, 0)

    def _camera(self, gray: np.ndarray) -> np.ndarray:
        """Blur, sensor noise and JPEG blocking, as a phone would add them."""
        out = gray
        if self._maybe(0.4):
            out = cv2.GaussianBlur(out, (0, 0), self.rng.uniform(0.4, 1.4))
        if self._maybe(0.4):
            out = np.clip(
                out.astype(np.float32) + self.rng.normal(0, self.rng.uniform(2, 10), out.shape),
                0, 255,
            ).astype(np.uint8)
        if self._maybe(0.35):
            quality = int(self.rng.integers(25, 80))
            ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if ok:
                out = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        return out

    def _levels(self, gray: np.ndarray) -> np.ndarray:
        """Contrast and brightness shifts, plus occasional faded ink."""
        a = self.rng.uniform(0.65, 1.35)
        b = self.rng.uniform(-35, 35) * self.strength
        return np.clip(gray.astype(np.float32) * a + b, 0, 255).astype(np.uint8)

    def __call__(self, image) -> np.ndarray:
        """Augmented line -> the same (1, H, W) float array `preprocess` yields."""
        return preprocess(self.augment_gray(image))

    def augment_gray(self, image) -> np.ndarray:
        """The augmentation chain alone, returning a grayscale image.

        Split out of `__call__` so an evaluation set can be built from the same
        pipeline the model trains against and still be handled as an image,
        rather than as an already-normalized tensor. Test-time augmentation and
        segmentation both need to resize, and neither can do that to a
        normalized array.
        """
        gray = _to_gray(image)
        original_ink = ink_fraction(gray)

        if self._maybe(0.6):
            gray = self._shear(gray)
        if self._maybe(0.5):
            gray = self._rotate(gray)
        if self._maybe(0.45):
            gray = self._perspective(gray)
        if self._maybe(0.5):
            gray = self._elastic(gray)
        if self._maybe(0.65):
            gray = self._stretch(gray)
        if self._maybe(0.75):
            gray = self._stroke(gray)
        if self._maybe(0.7):
            gray = self._vertical_fit(gray)
        if self._maybe(0.35):
            gray = self._neighbour_bleed(gray)
        if self._maybe(0.28):
            gray = self._bleed_through(gray)
        if self._maybe(0.55):
            gray = self._paper(gray)
        if self._maybe(0.25):
            gray = self._ruled_line(gray)
        if self._maybe(0.5):
            gray = self._levels(gray)
        gray = self._camera(gray)

        # Last line of defence: if the pipeline as a whole destroyed the
        # writing, fall back to the untouched crop rather than hand the model a
        # blank image paired with a full transcription.
        if ink_fraction(gray) < original_ink * 0.15:
            return _to_gray(image)
        return gray
