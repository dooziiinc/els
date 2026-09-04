"""2-D grid arrangement of a corpus and 'vicinity' search: given a word
already found as an ELS, search for other words that appear as ELS
near it once the flat letter stream is wrapped into rows of a chosen
width.

Two notions of "near" are supported (this is a deliberate design
choice -- the classic bible-code displays and the underlying WRR-style
literature both use variants of this, and there's no single agreed
definition):

  * grid_block  -- reshape the corpus into rows of `width` letters
    (conventionally width == the skip of the primary match, so the
    found word reads as a single straight line down the grid) and
    look for candidate words whose *entire* ELS falls inside a
    rectangular block of rows/columns around the primary match. This
    is the "does this word appear in the same displayed panel"
    definition used by the reference sites in the prompt.

  * radius -- purely positional: look for candidate ELS matches whose
    *starting* cell lies within a Chebyshev (row, col) radius of the
    primary match's starting cell, regardless of the display width
    chosen. This is cheaper and doesn't require every letter of the
    candidate word to land inside a fixed panel.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..corpus import Corpus, encode_word
from .els import ELSMatch, els_search


def _default_width(corpus: Corpus, primary_match: ELSMatch) -> int:
    """abs(skip) is the classic choice (draws the primary word as a
    straight column), but that's degenerate for skip==1 (an ordinary
    substring match) -- a 1-column or 1-row grid has no useful
    'vicinity'. In that case fall back to the roughly-square width
    convention (round(sqrt(N))) commonly used for ELS grid displays."""
    skip = abs(primary_match.skip)
    if skip > 1:
        return skip
    return max(2, round(math.sqrt(len(corpus))))


@dataclass
class Grid:
    """A read-only view of a Corpus reshaped into rows of `width`
    letters. Position 0 is the top-left cell; row-major, left-to-right
    (matching the order letters were stored in the corpus)."""

    corpus: Corpus
    width: int

    def __post_init__(self):
        if self.width < 1:
            raise ValueError("width must be >= 1")
        self.n = len(self.corpus)
        self.rows = (self.n + self.width - 1) // self.width

    def to_rowcol(self, pos: int) -> Tuple[int, int]:
        return divmod(pos, self.width)

    def to_pos(self, row: int, col: int) -> Optional[int]:
        if row < 0 or col < 0 or col >= self.width:
            return None
        pos = row * self.width + col
        return pos if 0 <= pos < self.n else None

    def rows_text(self) -> List[str]:
        """Render the grid as a list of strings, one per row (last row
        may be shorter than `width`)."""
        letters = self.corpus.raw_letters
        return [letters[r * self.width : r * self.width + self.width] for r in range(self.rows)]

    def slice_rowcols(self, row_lo: int, row_hi: int, col_lo: int, col_hi: int) -> List[str]:
        """A sub-block of the grid as a list of row strings, useful for
        rendering just the panel around a match."""
        rows_text = self.rows_text()
        row_lo = max(row_lo, 0)
        row_hi = min(row_hi, self.rows - 1)
        col_lo = max(col_lo, 0)
        col_hi = min(col_hi, self.width - 1)
        return [row[col_lo : col_hi + 1] for row in rows_text[row_lo : row_hi + 1]]


@dataclass
class VicinityHit:
    word: str
    match: ELSMatch
    grid_width: int


def _bounding_rowcols(grid: Grid, match: ELSMatch) -> Tuple[int, int, int, int]:
    rc = [grid.to_rowcol(p) for p in match.positions()]
    rows = [r for r, _ in rc]
    cols = [c for _, c in rc]
    return min(rows), max(rows), min(cols), max(cols)


def vicinity_search_grid_block(
    corpus: Corpus,
    primary_match: ELSMatch,
    candidate_words: Sequence[str],
    width: Optional[int] = None,
    row_margin: int = 0,
    full_width: bool = True,
    col_margin: int = 0,
    min_skip: int = 1,
    max_skip: Optional[int] = None,
    directions: Sequence[str] = ("forward", "backward"),
    limit: Optional[int] = 200,
) -> List[VicinityHit]:
    """Find candidate words whose ELS falls entirely inside the
    rectangular grid panel around `primary_match`.

    Args:
        width: grid row width. Defaults to abs(primary_match.skip),
            the classic convention (the primary word then reads as a
            single straight column).
        row_margin: extra rows of padding above/below the primary
            match's own row span.
        full_width: if True (default, matches the reference-site
            display), the panel spans the *entire* row width -- i.e.
            "same rows" is the only constraint. If False, the panel is
            also constrained in columns to the primary match's column
            span (+/- col_margin).
        limit: cap on the number of hits returned, keeping the
            smallest-|skip| (most "notable") ones first. Short
            candidate words in a wide skip range can match combinatorially
            often -- that's expected (short words carry little
            evidentiary weight on their own), so results are capped by
            default rather than silently returning tens of thousands of
            rows. Pass None for no cap.
    """
    if width is None:
        width = _default_width(corpus, primary_match)
    grid = Grid(corpus, width)

    row_lo, row_hi, col_lo, col_hi = _bounding_rowcols(grid, primary_match)
    row_lo -= row_margin
    row_hi += row_margin
    if full_width:
        col_lo, col_hi = 0, width - 1
    else:
        col_lo -= col_margin
        col_hi += col_margin
    row_lo = max(row_lo, 0)
    row_hi = min(row_hi, grid.rows - 1)

    # A candidate's *entire* ELS must fall inside [row_lo, row_hi], so
    # its start (and every other letter) must lie within that band's
    # flat position range. Search only that local slice instead of the
    # whole corpus -- for a typical panel (a handful to a few dozen
    # rows) this is orders of magnitude smaller than the full text, and
    # is what makes an unbounded max_skip tractable.
    slice_start = row_lo * width
    slice_end = min(len(corpus), (row_hi + 1) * width)
    local_codes = corpus.match_codes[slice_start:slice_end]
    local_max_skip = max_skip if max_skip is not None else len(local_codes)
    local_max_skip = min(local_max_skip, max(1, len(local_codes) - 1))

    hits: List[VicinityHit] = []
    for word in candidate_words:
        word_codes = encode_word(word)
        if len(word_codes) == 0:
            continue
        local_matches = els_search(
            local_codes,
            word_codes,
            word_str=word,
            min_skip=min_skip,
            max_skip=local_max_skip,
            directions=directions,
        )
        for m in local_matches:
            gm = ELSMatch(word, m.start + slice_start, m.skip, m.length)
            rc = [grid.to_rowcol(p) for p in gm.positions()]
            if all(row_lo <= r <= row_hi and col_lo <= c <= col_hi for r, c in rc):
                hits.append(VicinityHit(word, gm, width))
    hits.sort(key=lambda h: (abs(h.match.skip), h.match.start))
    return hits[:limit] if limit is not None else hits


def vicinity_search_radius(
    corpus: Corpus,
    primary_match: ELSMatch,
    candidate_words: Sequence[str],
    width: Optional[int] = None,
    radius: int = 5,
    min_skip: int = 1,
    max_skip: Optional[int] = None,
    directions: Sequence[str] = ("forward", "backward"),
    limit: Optional[int] = 200,
) -> List[VicinityHit]:
    """Find candidate ELS matches whose *starting cell* lies within a
    Chebyshev (row, col) radius of the primary match's starting cell,
    once the corpus is arranged in a grid of the given `width`. See
    vicinity_search_grid_block for the `limit` behavior.
    """
    if width is None:
        width = _default_width(corpus, primary_match)
    grid = Grid(corpus, width)
    r0, c0 = grid.to_rowcol(primary_match.start)

    # Only a candidate's *starting* cell needs to be within `radius`
    # rows, so only its start needs to fall in this row band -- search
    # that local slice rather than the whole corpus (see the same
    # reasoning in vicinity_search_grid_block).
    row_lo = max(r0 - radius, 0)
    row_hi = min(r0 + radius, grid.rows - 1)
    slice_start = row_lo * width
    slice_end = min(len(corpus), (row_hi + 1) * width)
    local_codes = corpus.match_codes[slice_start:slice_end]
    local_max_skip = max_skip if max_skip is not None else len(local_codes)
    local_max_skip = min(local_max_skip, max(1, len(local_codes) - 1))

    hits: List[VicinityHit] = []
    for word in candidate_words:
        word_codes = encode_word(word)
        if len(word_codes) == 0:
            continue
        local_matches = els_search(
            local_codes,
            word_codes,
            word_str=word,
            min_skip=min_skip,
            max_skip=local_max_skip,
            directions=directions,
        )
        for m in local_matches:
            gm = ELSMatch(word, m.start + slice_start, m.skip, m.length)
            r, c = grid.to_rowcol(gm.start)
            if max(abs(r - r0), abs(c - c0)) <= radius:
                hits.append(VicinityHit(word, gm, width))
    hits.sort(key=lambda h: (abs(h.match.skip), h.match.start))
    return hits[:limit] if limit is not None else hits


def vicinity_search(
    corpus: Corpus,
    primary_match: ELSMatch,
    candidate_words: Sequence[str],
    mode: str = "grid_block",
    **kwargs,
) -> List[VicinityHit]:
    """Dispatch to the requested vicinity-search mode ('grid_block' or
    'radius'); see vicinity_search_grid_block / vicinity_search_radius
    for the mode-specific keyword arguments."""
    if mode == "grid_block":
        return vicinity_search_grid_block(corpus, primary_match, candidate_words, **kwargs)
    if mode == "radius":
        return vicinity_search_radius(corpus, primary_match, candidate_words, **kwargs)
    raise ValueError(f"Unknown vicinity search mode {mode!r}; use 'grid_block' or 'radius'")
