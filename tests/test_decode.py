import torch

from hebocr.charset import BLANK, Charset
from hebocr.decode import greedy_confidence, greedy_decode


def _logprobs(sequence, n_classes):
    tensor = torch.full((1, len(sequence), n_classes), -20.0)
    for t, index in enumerate(sequence):
        tensor[0, t, index] = 0.0
    return torch.log_softmax(tensor, dim=-1)


def test_repeats_collapse_and_blanks_vanish():
    charset = Charset.default()
    a, b = charset.encode("אב")
    decoded = greedy_decode(
        _logprobs([a, a, a, BLANK, b, b], charset.n_classes), torch.tensor([6]), charset
    )
    assert decoded == ["אב"]


def test_blank_separates_a_doubled_letter():
    """Without the blank between them, 'אא' would collapse to 'א'."""
    charset = Charset.default()
    a = charset.encode("א")[0]
    assert greedy_decode(_logprobs([a, BLANK, a], charset.n_classes), torch.tensor([3]), charset) == ["אא"]
    assert greedy_decode(_logprobs([a, a, a], charset.n_classes), torch.tensor([3]), charset) == ["א"]


def test_padding_beyond_the_valid_length_is_ignored():
    """Batch padding must never contribute characters."""
    charset = Charset.default()
    a, b = charset.encode("אב")
    logprobs = _logprobs([a, BLANK, b, b, b], charset.n_classes)
    assert greedy_decode(logprobs, torch.tensor([1]), charset) == ["א"]


def test_confidence_is_between_zero_and_one():
    charset = Charset.default()
    a = charset.encode("א")[0]
    score = greedy_confidence(_logprobs([a, a], charset.n_classes), torch.tensor([2]))[0]
    assert 0.0 <= score <= 1.0


def test_beam_decode_matches_greedy_on_an_unambiguous_signal():
    """With one dominant path and no LM, beam search must agree with greedy."""
    from hebocr.decode import beam_decode

    charset = Charset.default()
    a, b = charset.encode("אב")
    logprobs = _logprobs([a, a, BLANK, b, b], charset.n_classes)
    lengths = torch.tensor([5])
    assert beam_decode(logprobs, lengths, charset, beam_width=8) == greedy_decode(
        logprobs, lengths, charset
    )


def test_beam_decode_respects_valid_lengths():
    from hebocr.decode import beam_decode

    charset = Charset.default()
    a, b = charset.encode("אב")
    logprobs = _logprobs([a, BLANK, b], charset.n_classes)
    assert beam_decode(logprobs, torch.tensor([1]), charset, beam_width=4) == ["א"]


def test_language_model_can_change_the_chosen_path():
    """A tie the visual model cannot break, broken by Hebrew letter statistics."""
    from hebocr.decode import beam_decode
    from hebocr.lm import CharNGramLM

    charset = Charset.default()
    good, bad = charset.encode("םך")  # both final forms, equally likely visually
    prefix = charset.encode("של")

    logprobs = torch.full((1, 4, charset.n_classes), -30.0)
    for t, index in enumerate(prefix):
        logprobs[0, t, index] = 0.0
    logprobs[0, 2, good] = logprobs[0, 2, bad] = 0.0  # exact tie
    logprobs[0, 3, BLANK] = 0.0
    logprobs = torch.log_softmax(logprobs, dim=-1)

    lm = CharNGramLM(order=4).train(["שלם"] * 200)
    decoded = beam_decode(
        logprobs, torch.tensor([4]), charset, beam_width=8, lm=lm, lm_weight=2.0, length_bonus=0.0
    )
    assert decoded[0].endswith("ם")
