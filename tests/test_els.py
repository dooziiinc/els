import numpy as np
import pytest

from torahcodes.search.els import ELSMatch, els_search, find_minimal_skip_els, minimal_skip_match


def codes_of(s: str) -> np.ndarray:
    return np.array([ord(c) for c in s], dtype=np.int32)


def test_simple_forward_skip():
    # "ABCDEFGHIJ" -> looking for "ACE" should be found at start=0, skip=2
    text = codes_of("ABCDEFGHIJ")
    word = codes_of("ACE")
    matches = els_search(text, word, "ACE", min_skip=1, max_skip=5, directions=("forward",))
    assert ELSMatch("ACE", 0, 2, 3) in matches


def test_backward_skip():
    # reverse: "ECA" should match text at start=4 (E), skip=-2 -> E, C, A
    text = codes_of("ABCDEFGHIJ")
    word = codes_of("ECA")
    matches = els_search(text, word, "ECA", min_skip=1, max_skip=5, directions=("backward",))
    assert any(m.start == 4 and m.skip == -2 for m in matches)


def test_no_match():
    text = codes_of("ABCDEFGHIJ")
    word = codes_of("ZZZ")
    matches = els_search(text, word, "ZZZ", min_skip=1, max_skip=5)
    assert matches == []


def test_skip_1_is_plain_substring():
    text = codes_of("HELLOWORLD")
    word = codes_of("WORLD")
    matches = els_search(text, word, "WORLD", min_skip=1, max_skip=1, directions=("forward",))
    assert len(matches) == 1
    assert matches[0].start == 5
    assert matches[0].skip == 1


def test_positions_and_span():
    m = ELSMatch("ACE", 0, 2, 3)
    assert m.positions() == [0, 2, 4]
    assert m.end == 4
    assert m.span == 5


def test_minimal_skip_match_picks_smallest_abs_skip():
    matches = [
        ELSMatch("X", 10, 5, 1),
        ELSMatch("X", 3, -2, 1),
        ELSMatch("X", 7, 3, 1),
    ]
    best = minimal_skip_match(matches)
    assert best.skip == -2


def test_find_minimal_skip_els_matches_brute_force():
    rng = np.random.default_rng(42)
    text = np.array(rng.integers(0, 5, size=2000), dtype=np.int32)  # small alphabet -> lots of hits
    word = np.array([1, 2, 3], dtype=np.int32)

    brute = els_search(text, word, "w", min_skip=1, max_skip=200)
    best_brute = minimal_skip_match(brute)

    fast = find_minimal_skip_els(text, word, "w", max_skip_cap=200)

    assert best_brute is not None and fast is not None
    assert abs(fast.skip) == abs(best_brute.skip)


def test_out_of_bounds_skip_returns_nothing():
    text = codes_of("AB")
    word = codes_of("ABCDE")
    matches = els_search(text, word, "ABCDE", min_skip=1, max_skip=10)
    assert matches == []
