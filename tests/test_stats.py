import numpy as np
import pytest

from torahcodes.corpus import Corpus, Reference
from torahcodes.search.els import ELSMatch
from torahcodes.stats.compactness import compactness
from torahcodes.stats.permutations import shuffle_letters, shuffle_words
from torahcodes.stats.significance import minimal_skip_significance, pair_compactness_significance


def make_word_corpus(words):
    """Build a Corpus (with a word index) out of a list of 'word' strings."""
    raw = "".join(words)
    codes = np.array([ord(c) for c in raw], dtype=np.int32)
    word_index = []
    pos = 0
    for i, w in enumerate(words):
        word_index.append((pos, Reference("Test", 1, i + 1, 1)))
        pos += len(w)
    return Corpus("test", raw, codes, codes, word_index=word_index)


def test_compactness_bounding_box():
    # width=4 grid:
    # row0: A B C D
    # row1: E F G H
    # match1 spans positions 0,4,8 -> col 0, rows 0,1,2 (skip=4 => width default 4)
    m1 = ELSMatch("m1", 0, 4, 3)
    # match2 spans positions 1,2 -> row0 cols1-2
    m2 = ELSMatch("m2", 1, 1, 2)
    result = compactness(m1, m2, width=4)
    # rows spanned by m1: 0,1,2 ; by m2: row 0 -> union rows 0-2 => span 3
    assert result.row_span == 3
    # cols spanned by m1: col 0 ; by m2: cols 1,2 -> union cols 0-2 => span 3
    assert result.col_span == 3
    assert result.area == 9


def test_shuffle_letters_preserves_multiset_and_length():
    words = ["ABAB", "CDCD", "EFEF", "GHGH"]
    corpus = make_word_corpus(words)
    shuffled = shuffle_letters(corpus, seed=0)
    assert len(shuffled) == len(corpus)
    assert sorted(shuffled.raw_letters) == sorted(corpus.raw_letters)
    assert shuffled.word_index is None


def test_shuffle_words_preserves_word_multiset():
    words = ["ABAB", "CDCD", "EFEF", "GHGH"]
    corpus = make_word_corpus(words)
    shuffled = shuffle_words(corpus, seed=0)
    assert len(shuffled) == len(corpus)
    # the same 4-letter chunks must reappear (possibly reordered)
    chunks = [shuffled.raw_letters[i : i + 4] for i in range(0, len(shuffled), 4)]
    assert sorted(chunks) == sorted(words)


def test_minimal_skip_significance_runs_and_bounds_p_value():
    rng = np.random.default_rng(0)
    # Alphabet of 6 letters spread over a few thousand positions so that
    # a 3-letter word has plausible ELS occurrences at small skips.
    raw = "".join(chr(ord("A") + int(x)) for x in rng.integers(0, 6, size=3000))
    codes = np.array([ord(c) for c in raw], dtype=np.int32)
    corpus = Corpus("synthetic", raw, codes, codes, word_index=None)

    # Find any word that's actually present so the test is deterministic-ish.
    from torahcodes.corpus import encode_word
    from torahcodes.search.els import find_minimal_skip_els

    word = "ABC"
    assert find_minimal_skip_els(corpus.match_codes, encode_word(word, hebrew=False), word) is not None

    result = minimal_skip_significance(corpus, word, n_permutations=15, permutation="letters", max_skip_cap=500, seed=1)
    assert 0.0 <= result.p_value <= 1.0
    assert result.n_permutations == 15
    assert result.observed is not None


def test_pair_compactness_significance_runs_and_bounds_p_value():
    rng = np.random.default_rng(1)
    raw = "".join(chr(ord("A") + int(x)) for x in rng.integers(0, 6, size=3000))
    codes = np.array([ord(c) for c in raw], dtype=np.int32)
    corpus = Corpus("synthetic", raw, codes, codes, word_index=None)

    result = pair_compactness_significance(corpus, "ABC", "DEF", n_permutations=15, permutation="letters", max_skip_cap=500, seed=2)
    assert 0.0 <= result.p_value <= 1.0
    assert result.observed is not None
    assert result.observed >= 1
