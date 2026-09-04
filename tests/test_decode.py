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
