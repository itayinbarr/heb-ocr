import math

import numpy as np
import pytest

from hebocr.metrics import cer, line_report, page_report, word_coverage, _quartiles


def test_cer_identical_is_zero():
    assert cer("שלום עולם", "שלום עולם") == 0.0


def test_cer_is_edit_distance_over_reference_length():
    assert cer("abcd", "abed") == pytest.approx(0.25)
    assert cer("abcd", "") == pytest.approx(1.0)


def test_cer_can_exceed_one():
    """A runaway output is worse than silence, and must score worse than 1.0."""
    assert cer("ab", "abababab") > 1.0


def test_cer_empty_reference_is_degenerate_not_a_crash():
    assert cer("", "") == 0.0
    assert cer("", "x") == 1.0


def test_word_coverage_ignores_order():
    assert word_coverage("a b c", "c a b") == 1.0


def test_word_coverage_is_a_multiset():
    """A word needed twice but produced once is only half covered."""
    assert word_coverage("a a", "a") == pytest.approx(0.5)
    assert word_coverage("a a", "a a a") == 1.0


def test_quartiles_match_numpy():
    for n in (1, 2, 5, 9, 20, 225):
        values = list(np.random.default_rng(n).random(n))
        q1, q3 = _quartiles(values)
        assert q1 == pytest.approx(float(np.percentile(values, 25)))
        assert q3 == pytest.approx(float(np.percentile(values, 75)))


def test_line_report_drops_blanks_from_the_median():
    """The leaderboard's rule: 'Outputs with no text are dropped.'"""
    report = line_report(["aaaa", "bbbb", "cccc"], ["aaaa", "", "cccc"])
    assert report.n_scored == 2 and report.n_blank == 1
    assert report.cer_median == 0.0


def test_line_report_nodrop_charges_blanks():
    """The honest counterpart: a blank is a total miss, not a free pass."""
    report = line_report(["aaaa", "bbbb", "cccc"], ["aaaa", "", "cccc"])
    assert report.cer_median_nodrop == pytest.approx(0.0)
    report2 = line_report(["aaaa", "bbbb"], ["", ""])
    assert report2.cer_median_nodrop == 1.0
    assert math.isnan(report2.cer_median)


def test_line_report_blank_gaming_is_visible():
    """Answering only the easy line looks perfect on median, awful on no-drop."""
    refs = [f"line{i}" for i in range(10)]
    hyps = ["line0"] + [""] * 9
    report = line_report(refs, hyps)
    assert report.cer_median == 0.0
    assert report.cer_median_nodrop == 1.0
    assert report.n_blank == 9


def test_line_report_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        line_report(["a"], ["a", "b"])


def test_page_report_order_independence():
    """Word coverage must not punish a reversed reading order."""
    gold = "שורה אחת\nשורה שתיים"
    forward = page_report([gold], ["שורה אחת שורה שתיים"])
    reversed_ = page_report([gold], ["שורה שתיים שורה אחת"])
    assert forward.wcov_micro == reversed_.wcov_micro == 1.0
    # Page CER is the strict view and *should* punish it.
    assert reversed_.cer_micro > forward.cer_micro


def test_page_report_counts_blank_pages():
    report = page_report(["a b", "c d"], ["a b", ""])
    assert report.n_blank == 1 and report.n_scored == 1


def test_language_model_prefers_seen_sequences():
    from hebocr.lm import CharNGramLM

    lm = CharNGramLM(order=5).train(["שלום עולם"] * 100)
    assert lm.logprob("שלו", "ם") > lm.logprob("שלו", "ז")


def test_language_model_backs_off_instead_of_returning_negative_infinity():
    from hebocr.lm import CharNGramLM

    lm = CharNGramLM(order=5).train(["אבג"] * 10)
    assert lm.logprob("zzz", "ת") > float("-inf")


def test_language_model_survives_a_save_load_roundtrip(tmp_path):
    from hebocr.lm import CharNGramLM

    lm = CharNGramLM(order=4).train(["שלום עולם"] * 50)
    path = tmp_path / "lm.pkl"
    lm.save(path)
    loaded = CharNGramLM.load(path)
    assert loaded.logprob("של", "ו") == lm.logprob("של", "ו")
    assert loaded.stats() == lm.stats()
