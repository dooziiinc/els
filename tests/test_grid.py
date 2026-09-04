import numpy as np

from torahcodes.corpus import Corpus, Reference
from torahcodes.search.els import ELSMatch
from torahcodes.search.grid import Grid, vicinity_search_grid_block, vicinity_search_radius


def make_corpus(letters: str) -> Corpus:
    codes = np.array([ord(c) for c in letters], dtype=np.int32)
    word_index = [(0, Reference("Test", 1, 1, 1))]
    return Corpus("test", letters, codes, codes, word_index=word_index)


def test_grid_rowcol_roundtrip():
    corpus = make_corpus("ABCDEFGHIJKL")  # 12 letters
    grid = Grid(corpus, width=4)
    assert grid.rows == 3
    assert grid.to_rowcol(0) == (0, 0)
    assert grid.to_rowcol(5) == (1, 1)
    assert grid.to_pos(1, 1) == 5
    assert grid.rows_text() == ["ABCD", "EFGH", "IJKL"]


def test_grid_slice():
    corpus = make_corpus("ABCDEFGHIJKL")
    grid = Grid(corpus, width=4)
    block = grid.slice_rowcols(0, 1, 1, 2)
    assert block == ["BC", "FG"]


def test_vicinity_search_grid_block_finds_word_in_same_rows():
    # Build text so that at width=3, row1 = "XYZ" contains the word "YZX"?
    # Simpler: construct text where primary word reads as a column
    # (skip = width) and a second word sits in the same row band.
    letters = "A1XB2YC3Z"  # width 3 -> rows: "A1X","B2Y","C3Z"
    corpus = make_corpus(letters)
    # primary match: "ABC" at start=0 skip=3 (column of first letters)
    primary = ELSMatch("ABC", 0, 3, 3)
    hits = vicinity_search_grid_block(corpus, primary, ["XYZ"], width=3, min_skip=1, max_skip=5)
    assert any(h.word == "XYZ" for h in hits)


def test_vicinity_search_radius():
    letters = "ABCDEFGHIJKLMNOP"  # width 4
    corpus = make_corpus(letters)
    primary = ELSMatch("A", 0, 1, 1)
    # "F" is at position 5 -> row1,col1 ; within radius 1 of (0,0)
    hits = vicinity_search_radius(corpus, primary, ["F"], width=4, radius=1, min_skip=1, max_skip=1)
    assert any(h.word == "F" for h in hits)
    # "P" is at position 15 -> row3,col3 ; should NOT be within radius 1
    hits_far = vicinity_search_radius(corpus, primary, ["P"], width=4, radius=1, min_skip=1, max_skip=1)
    assert not any(h.word == "P" for h in hits_far)
