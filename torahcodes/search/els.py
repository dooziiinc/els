"""Core Equidistant Letter Sequence (ELS) search.

An ELS of a word w = w[0..L-1] in a letter array `codes` of length N is
a starting position `start` and a nonzero integer skip `d` such that

    codes[start + i*d] == w[i]   for i = 0 .. L-1

`d > 0` reads in the corpus's natural (forward) order; `d < 0` reads
backward. `abs(d) == 1` is an ordinary (unskipped) substring match.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class ELSMatch:
    word: str
    start: int          # index of the word's first letter
    skip: int            # signed skip distance (nonzero)
    length: int           # len(word) in letters

    @property
    def end(self) -> int:
        """Index of the word's last letter."""
        return self.start + (self.length - 1) * self.skip

    @property
    def span(self) -> int:
        """Total number of array positions the ELS occupies, start to end inclusive."""
        return abs(self.end - self.start) + 1

    def positions(self) -> List[int]:
        return [self.start + i * self.skip for i in range(self.length)]

    def min_index(self) -> int:
        return min(self.start, self.end)

    def max_index(self) -> int:
        return max(self.start, self.end)


def _search_one_skip(codes: np.ndarray, word_codes: np.ndarray, d: int) -> np.ndarray:
    """Return an array of valid start positions for skip `d` (d != 0)."""
    n = len(codes)
    L = len(word_codes)
    span = (L - 1) * abs(d)
    if span >= n:
        return np.empty(0, dtype=np.int64)

    if d > 0:
        max_start = n - span - 1
        starts = np.arange(0, max_start + 1, dtype=np.int64)
    else:
        # start must satisfy start + (L-1)*d >= 0  =>  start >= span
        starts = np.arange(span, n, dtype=np.int64)

    mask = np.ones(len(starts), dtype=bool)
    for i in range(L):
        idx = starts + i * d
        mask &= codes[idx] == word_codes[i]
        if not mask.any():
            break
    return starts[mask]


def els_search(
    codes: np.ndarray,
    word_codes: np.ndarray,
    word_str: str = "",
    min_skip: int = 1,
    max_skip: Optional[int] = None,
    directions: Sequence[str] = ("forward", "backward"),
) -> List[ELSMatch]:
    """Search `codes` for every ELS occurrence of `word_codes`.

    Args:
        codes: the corpus, as an int array (Corpus.match_codes).
        word_codes: the encoded search word (see corpus.encode_word).
        word_str: original word text, stored on results for display.
        min_skip: smallest |skip| to try (must be >= 1).
        max_skip: largest |skip| to try. Defaults to the largest skip
            for which the word can still fit at all (N // (L-1)).
        directions: any of "forward" (d > 0), "backward" (d < 0).

    Returns:
        A list of ELSMatch, sorted by abs(skip) then start position --
        i.e. the most notable (shortest-skip) occurrences first.
    """
    n = len(codes)
    L = len(word_codes)
    if L == 0:
        return []
    if L == 1:
        # A single letter has no meaningful "skip"; treat skip=1 as the
        # only sensible representation and just find every occurrence.
        hits = np.flatnonzero(codes == word_codes[0])
        return [ELSMatch(word_str, int(s), 1, 1) for s in hits]

    if max_skip is None:
        max_skip = max(1, n // (L - 1))
    if min_skip < 1:
        raise ValueError("min_skip must be >= 1")
    if max_skip < min_skip:
        return []

    results: List[ELSMatch] = []
    for d_abs in range(min_skip, max_skip + 1):
        if "forward" in directions:
            for s in _search_one_skip(codes, word_codes, d_abs):
                results.append(ELSMatch(word_str, int(s), d_abs, L))
        if "backward" in directions:
            for s in _search_one_skip(codes, word_codes, -d_abs):
                results.append(ELSMatch(word_str, int(s), -d_abs, L))

    results.sort(key=lambda m: (abs(m.skip), m.start))
    return results


def minimal_skip_match(matches: Iterable[ELSMatch]) -> Optional[ELSMatch]:
    """The WRR convention of picking a single 'most notable' occurrence
    of a word: the one with the smallest |skip| (ties broken by
    leftmost start position). Returns None if `matches` is empty."""
    best = None
    for m in matches:
        if best is None or (abs(m.skip), m.start) < (abs(best.skip), best.start):
            best = m
    return best


def find_minimal_skip_els(
    codes: np.ndarray,
    word_codes: np.ndarray,
    word_str: str = "",
    max_skip_cap: Optional[int] = None,
    directions: Sequence[str] = ("forward", "backward"),
) -> Optional[ELSMatch]:
    """Efficiently find the single ELS occurrence with the smallest
    |skip| (the WRR 'most notable occurrence' convention), without
    enumerating every occurrence at every skip first.

    Scans |skip| = 1, 2, 3, ... in order and stops at the first skip
    where a match is found (checking both directions at each skip, if
    requested), so this is much cheaper than els_search() when you
    only need the minimal-skip representative -- which is what the
    statistics module needs, potentially thousands of times per
    permutation test.

    Returns None if no occurrence is found up to `max_skip_cap` (which
    defaults to N // (L-1), the largest skip for which the word could
    possibly fit at all).
    """
    n = len(codes)
    L = len(word_codes)
    if L == 0:
        return None
    if L == 1:
        hits = np.flatnonzero(codes == word_codes[0])
        if len(hits) == 0:
            return None
        return ELSMatch(word_str, int(hits[0]), 1, 1)

    if max_skip_cap is None:
        max_skip_cap = max(1, n // (L - 1))

    for d_abs in range(1, max_skip_cap + 1):
        if "forward" in directions:
            starts = _search_one_skip(codes, word_codes, d_abs)
            if len(starts):
                return ELSMatch(word_str, int(starts.min()), d_abs, L)
        if "backward" in directions:
            starts = _search_one_skip(codes, word_codes, -d_abs)
            if len(starts):
                return ELSMatch(word_str, int(starts.min()), -d_abs, L)
    return None
