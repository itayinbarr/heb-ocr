import pytest

from hebocr.normalize import is_blank, normalize, tokens


def test_gershayim_variants_collapse():
    """The gold's own two volunteers wrote this line both ways."""
    assert normalize('בס"ד') == normalize("בס''ד") == normalize("בס״ד")


def test_geresh_variants_collapse():
    assert normalize("צה׳ל") == normalize("צה'ל") == normalize("צה’ל")


def test_bidi_marks_are_stripped():
    """Invisible marks would otherwise count as edits against the gold."""
    assert normalize("‏שלום‎") == "שלום"
    assert normalize("﻿א​ב") == "אב"


def test_whitespace_collapses_and_trims():
    assert normalize("  a \t\n b  ") == "a b"
    assert normalize(" a b") == "a b"


def test_hebrew_letters_are_untouched():
    """Final forms are distinct letters and must survive scoring."""
    for word in ("שלום", "מים", "ארץ", "כסף", "ילדן"):
        assert normalize(word) == word


def test_niqqud_is_not_stripped():
    """'Hebrew text itself is untouched' -- vowel points are Hebrew text."""
    assert normalize("שָׁלוֹם") != "שלום"


def test_blank_detection():
    assert is_blank(None) and is_blank("") and is_blank("   ") and is_blank("‎")
    assert not is_blank("א")


def test_dashes_and_ellipsis_canonicalize():
    """Maqaf, en/em dashes and hyphen all mean the same thing to a scorer."""
    assert normalize("a—b") == normalize("a-b") == normalize("a־b") == "a-b"
    assert normalize("a…") == "a..."


@pytest.mark.parametrize("value", [None, "", "   "])
def test_tokens_of_blank_is_empty(value):
    assert tokens(value) == []


def test_tokens_split_on_whitespace():
    assert tokens(" שלום   עולם ") == ["שלום", "עולם"]
