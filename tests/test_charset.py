import pytest

from hebocr.charset import BLANK, Charset


def test_blank_is_index_zero():
    assert Charset.default().chars[BLANK] == ""


def test_roundtrip_preserves_text():
    charset = Charset.default()
    text = 'שלום, בס"ד 123 (א)'
    assert charset.decode(charset.encode(text)) == text


def test_final_forms_are_distinct_classes():
    """Folding ם onto מ would make two different words score as one."""
    charset = Charset.default()
    assert charset.encode("ם") != charset.encode("מ")
    for medial, final in (("כ", "ך"), ("מ", "ם"), ("נ", "ן"), ("פ", "ף"), ("צ", "ץ")):
        assert charset.encode(medial) != charset.encode(final)


def test_default_covers_every_gold_character():
    """Any character the charset lacks is a guaranteed error on that line."""
    from hebocr.data.benchmark import load_lines

    charset = Charset.default()
    uncovered = [l.text for l in load_lines() if charset.coverage(l.text) < 1.0]
    assert uncovered == []


def test_out_of_vocabulary_characters_are_dropped_not_fatal():
    charset = Charset.default()
    assert charset.encode("א☃ב") == charset.encode("אב")


def test_from_texts_honours_min_count():
    charset = Charset.from_texts(["אאא", "ב"], min_count=2)
    assert "א" in charset.chars and "ב" not in charset.chars


def test_save_and_load_roundtrip(tmp_path):
    charset = Charset.default()
    path = tmp_path / "charset.json"
    charset.save(path)
    assert Charset.load(path).chars == charset.chars


def test_rejects_charset_without_blank_slot():
    with pytest.raises(ValueError):
        Charset(chars=["א", "ב"])
