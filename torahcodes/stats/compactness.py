"""'Compactness' between two ELS matches.

The original Witztum-Rips-Rosenberg (WRR) papers used a notion of
"compactness" to argue that related word pairs (e.g. a rabbi's name
and birth/death date) cluster more tightly in the text, as ELSs, than
unrelated pairs do. WRR never published their exact compactness
formula in a fully reproducible form, and independent reviewers
(McKay, Bar-Natan, Bar-Hillel & Gill, "Solving the Bible Code Puzzle",
Statistical Science 1999) found that the specific choices involved
(which of several spellings/forms to use, how to measure distance)
gave researchers enough latitude to produce the reported result even
without any real effect.

This module implements our own clearly-specified, deterministic
operationalization of "compactness" -- a bounding-box measure in the
2-D grid arrangement -- so results here are exactly reproducible from
the code, but should NOT be read as a certified reproduction of WRR's
original (undisclosed) statistic. Use pair_compactness_significance()
in significance.py to test whether an observed compactness is more
extreme than chance under an explicit, inspectable null model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..search.els import ELSMatch


@dataclass
class CompactnessResult:
    match1: ELSMatch
    match2: ELSMatch
    width: int
    row_span: int
    col_span: int

    @property
    def area(self) -> int:
        """Bounding-box area in grid cells. Smaller = more compact."""
        return self.row_span * self.col_span


def compactness(match1: ELSMatch, match2: ELSMatch, width: Optional[int] = None) -> CompactnessResult:
    """Bounding-box compactness of two ELS matches once the corpus is
    laid out in a grid of `width` columns.

    Defaults to width = abs(match1.skip), the usual convention of
    drawing the first (anchor) word as a single straight column.
    """
    if width is None:
        width = max(1, abs(match1.skip))

    def rowcols(m: ELSMatch):
        return [divmod(p, width) for p in m.positions()]

    rc = rowcols(match1) + rowcols(match2)
    rows = [r for r, _ in rc]
    cols = [c for _, c in rc]
    row_span = max(rows) - min(rows) + 1
    col_span = max(cols) - min(cols) + 1
    return CompactnessResult(match1, match2, width, row_span, col_span)
