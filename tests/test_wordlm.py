"""The word unigram prior used for word-boundary fusion."""

from hebocr.wordlm import WordUnigramLM


def build() -> WordUnigramLM:
    lm = WordUnigramLM()
    lm.train(["שלום עולם שלום", "שלום לכם", "בית ספר גדול", "הבית הגדול"])
    return lm


def test_frequent_words_score_above_rare_ones():
    lm = build()
    assert lm.logprob("שלום") > lm.logprob("עולם")


def test_an_unknown_word_gets_the_floor_not_negative_infinity():
    lm = build()
    score = lm.logprob("קסדראותכ")
    assert score == lm.oov_logprob
    assert score > float("-inf")


def test_a_prefixed_form_backs_off_to_its_stem():
    """ktiv and particle fusion mean the corpus cannot hold every surface form."""
    lm = build()
    assert lm.logprob("ושלום") > lm.oov_logprob
    assert lm.logprob("ושלום") < lm.logprob("שלום")


def test_bonus_is_never_negative():
    """The fusion term must not punish, or the decoder learns to say less."""
    lm = build()
    for word in ("שלום", "עולם", "קסדראותכ", ""):
        assert lm.bonus(word) >= 0.0


def test_an_unknown_word_earns_no_bonus():
    lm = build()
    assert lm.bonus("קסדראותכ") == 0.0


def test_a_known_word_earns_a_bonus_over_an_unknown_one():
    lm = build()
    assert lm.bonus("שלום") > lm.bonus("קסדראותכ")


def test_bonus_ranks_the_same_way_logprob_does():
    lm = build()
    assert lm.bonus("שלום") > lm.bonus("עולם") > lm.bonus("זזזזזז")


def test_pruning_drops_singletons():
    lm = build()
    before = len(lm.counts)
    lm.prune(min_count=2)
    assert len(lm.counts) < before
    assert "שלום" in lm.counts


def test_round_trips_through_disk(tmp_path):
    lm = build()
    path = tmp_path / "words.pkl"
    lm.save(path)
    loaded = WordUnigramLM.load(path)
    assert loaded.total == lm.total
    assert loaded.logprob("שלום") == lm.logprob("שלום")
    assert loaded.bonus("שלום") == lm.bonus("שלום")


def test_known_matches_the_scoring_path():
    lm = build()
    assert lm.known("שלום")
    assert lm.known("ושלום")
    assert not lm.known("קסדראותכ")
