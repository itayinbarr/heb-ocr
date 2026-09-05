"""Guards on the repository itself, not the model.

These exist because `.gitignore` contained a bare `data/`, which matches
`hebocr/data/` as well as the top-level data directory. The entire data package
-- augmentation, datasets, glyph composition -- was silently never committed,
and the published repository could not even be imported. Tests passed the whole
time, because they ran against the working tree.
"""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def tracked_files() -> set[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return set(out.stdout.split())


def test_every_package_module_is_committed():
    """A module that exists only in the working tree is a module nobody else has."""
    tracked = tracked_files()
    on_disk = {
        str(p.relative_to(ROOT))
        for p in (ROOT / "hebocr").rglob("*.py")
        if "__pycache__" not in p.parts
    }
    missing = sorted(on_disk - tracked)
    assert not missing, f"untracked package modules: {missing}"


def test_every_test_module_is_committed():
    tracked = tracked_files()
    on_disk = {
        str(p.relative_to(ROOT))
        for p in (ROOT / "tests").rglob("*.py")
        if "__pycache__" not in p.parts
    }
    assert not sorted(on_disk - tracked)


def test_scripts_are_committed():
    tracked = tracked_files()
    on_disk = {
        str(p.relative_to(ROOT))
        for p in (ROOT / "scripts").rglob("*.py")
        if "__pycache__" not in p.parts
    }
    assert not sorted(on_disk - tracked)


def test_the_package_imports_from_a_clean_checkout():
    """Import every module, so a missing dependency between them is caught."""
    import importlib

    for name in (
        "hebocr.normalize", "hebocr.metrics", "hebocr.charset", "hebocr.decode",
        "hebocr.lm", "hebocr.optim", "hebocr.recognize", "hebocr.train",
        "hebocr.evaluate", "hebocr.models.htr_vt", "hebocr.page.segment",
        "hebocr.data.transforms", "hebocr.data.synth", "hebocr.data.glyphs",
        "hebocr.data.benchmark",
    ):
        importlib.import_module(name)
