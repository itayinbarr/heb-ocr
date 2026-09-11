"""Voting over several systems' transcriptions of the same line."""

import pytest

from hebocr.rover import rover, rover_batch


def test_unanimous_output_is_unchanged():
    assert rover(["שלום עולם"] * 3) == "שלום עולם"


def test_single_hypothesis_passes_through():
    assert rover(["שלום"]) == "שלום"
    assert rover([]) == ""


def test_majority_overrules_the_pivot():
    """Two systems agreeing on a character the pivot got wrong should win."""
    assert rover(["שלוח", "שלום", "שלום"]) == "שלום"


def test_pivot_wins_a_tie():
    """With one vote each, the strongest system decides."""
    assert rover(["שלום", "שלוח"]) == "שלום"


def test_weights_can_outvote_a_numeric_majority():
    """A heavily weighted pivot holds against two lighter disagreeing systems."""
    assert rover(["שלום", "שלוח", "שלוח"], weights=[3.0, 1.0, 1.0]) == "שלום"
    assert rover(["שלום", "שלוח", "שלוח"], weights=[1.0, 1.0, 1.0]) == "שלוח"


def test_a_lone_insertion_is_rejected():
    """One system adding text the others never saw must not get it through."""
    assert rover(["שלום", "שלום רב", "שלום"]) == "שלום"


def test_an_agreed_insertion_is_accepted():
    assert rover(["שלום", "שלום רב", "שלום רב"]) == "שלום רב"


def test_a_majority_deletion_removes_a_character():
    assert rover(["שלוםם", "שלום", "שלום"]) == "שלום"


def test_empty_hypotheses_do_not_erase_a_good_pivot():
    """A blank from one weak system should not empty the output."""
    assert rover(["שלום", "", ""], weights=[3.0, 1.0, 1.0]) == "שלום"


def test_length_differences_are_handled_without_crashing():
    out = rover(["אבג", "אבגדהו", "א"])
    assert isinstance(out, str)


def test_batch_maps_over_lines():
    sets = [["אבג", "אבג"], ["דהו", "דהז"]]
    assert rover_batch(sets) == ["אבג", "דהו"]


def test_weight_count_must_match():
    with pytest.raises(ValueError):
        rover(["א", "ב"], weights=[1.0])
