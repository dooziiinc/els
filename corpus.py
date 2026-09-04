"""Corpus loading: the Torah (the default, bundled corpus) and any
additional user-provided comparison texts, all reduced to the same
representation the search engine needs: a flat array of letter codes,
plus (optionally) an index that maps letter positions back to
human-readable references (book/chapter/verse/word, or line number for
a plain text corpus).
"""
from __future__ import annotations

import bisect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import normalize

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

BOOK_NAMES = {
    "Gen": "Genesis",
    "Exod": "Exodus",
    "Lev": "Leviticus",
    "Num": "Numbers",
    "Deut": "Deuteronomy",
}

BOOK_CHOICES = list(BOOK_NAMES.keys())


@dataclass
class Reference:
    book: str
    chapter: int
    verse: int
    word_num: int

    def __str__(self) -> str:
        name = BOOK_NAMES.get(self.book, self.book)
        return f"{name} {self.chapter}:{self.verse} (word {self.word_num})"


@dataclass
class Corpus:
    """A searchable text.

    Attributes:
        name: human-readable label.
        raw_letters: the letter stream as written (final forms intact).
        display_codes: np.int32 array of raw_letters, ordinal-coded --
            used when rendering the grid.
        match_codes: np.int32 array used for *matching* -- for Hebrew
            corpora this has final letter forms normalized to their
            regular form, per the standard ELS convention.
        word_index: optional list of (start_pos, Reference) giving the
            position where each source word begins, sorted by
            start_pos -- enables mapping a letter position back to a
            (book, chapter, verse, word) reference. None for corpora
            with no reference structure (e.g. an arbitrary text file).
    """

    name: str
    raw_letters: str
    display_codes: np.ndarray
    match_codes: np.ndarray
    word_index: Optional[List[Tuple[int, Reference]]] = field(default=None)
    _word_starts: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self):
        if self.word_index is not None:
            self._word_starts = np.array([p for p, _ in self.word_index])

    def __len__(self) -> int:
        return len(self.raw_letters)

    def reference_at(self, pos: int) -> Optional[Reference]:
        """Return the Reference for the word containing letter `pos`,
        or None if this corpus has no reference index."""
        if self.word_index is None or self._word_starts is None:
            return None
        if pos < 0 or pos >= len(self.raw_letters):
            return None
        i = bisect.bisect_right(self._word_starts, pos) - 1
        if i < 0:
            return None
        return self.word_index[i][1]

    def slice_text(self, start: int, end: int) -> str:
        return self.raw_letters[start:end]


def load_torah(normalize_final_forms: bool = True) -> Corpus:
    """Load the bundled Torah text (built from the WLC via
    scripts/build_torah_text.py)."""
    letters_path = DATA_DIR / "torah_letters.txt"
    index_path = DATA_DIR / "torah_index.json"
    if not letters_path.exists():
        raise FileNotFoundError(
            f"{letters_path} not found. Run scripts/build_torah_text.py first."
        )
    raw_letters = letters_path.read_text(encoding="utf-8")

    word_index: List[Tuple[int, Reference]] = []
    if index_path.exists():
        raw_index = json.loads(index_path.read_text(encoding="utf-8"))
        for start_pos, book, chapter, verse, word_num in raw_index:
            word_index.append((start_pos, Reference(book, chapter, verse, word_num)))

    display_codes = np.array([ord(c) for c in raw_letters], dtype=np.int32)
    if normalize_final_forms:
        match_text = normalize.normalize_finals(raw_letters)
    else:
        match_text = raw_letters
    match_codes = np.array([ord(c) for c in match_text], dtype=np.int32)

    return Corpus(
        name="Torah (WLC, Gen-Deut)",
        raw_letters=raw_letters,
        display_codes=display_codes,
        match_codes=match_codes,
        word_index=word_index,
    )


def load_book(book_code: str, normalize_final_forms: bool = True) -> Corpus:
    """Load a single Torah book (e.g. 'Gen') as its own Corpus, sliced
    out of the full Torah using data/book_boundaries.json."""
    boundaries_path = DATA_DIR / "book_boundaries.json"
    boundaries = json.loads(boundaries_path.read_text(encoding="utf-8"))
    if book_code not in boundaries:
        raise KeyError(f"Unknown book code {book_code!r}. Known: {list(boundaries)}")
    start, end = boundaries[book_code]

    full = load_torah(normalize_final_forms=normalize_final_forms)
    word_index = None
    if full.word_index is not None:
        word_index = [
            (pos - start, ref) for pos, ref in full.word_index if start <= pos < end
        ]
    return Corpus(
        name=f"Torah / {BOOK_NAMES.get(book_code, book_code)}",
        raw_letters=full.raw_letters[start:end],
        display_codes=full.display_codes[start:end],
        match_codes=full.match_codes[start:end],
        word_index=word_index,
    )


def load_text_file(
    path: str | Path,
    name: Optional[str] = None,
    hebrew: Optional[bool] = None,
    normalize_final_forms: bool = True,
) -> Corpus:
    """Load an arbitrary user-provided comparison corpus from a plain
    text file. If `hebrew` is None, script is auto-detected. Reference
    tracking for these corpora is by (1-indexed) line number rather
    than book/chapter/verse, stored as Reference(book=<file name>,
    chapter=<line>, verse=0, word_num=<word position in line>).
    """
    path = Path(path)
    raw_text = path.read_text(encoding="utf-8", errors="replace")
    is_heb = normalize.is_hebrew_text(raw_text) if hebrew is None else hebrew

    letters_chunks: List[str] = []
    word_index: List[Tuple[int, Reference]] = []
    pos = 0
    for line_num, line in enumerate(raw_text.splitlines(), start=1):
        words = line.split()
        for word_num, w in enumerate(words, start=1):
            letters = (
                normalize.strip_to_hebrew_letters(w)
                if is_heb
                else normalize.strip_to_generic_letters(w)
            )
            if not letters:
                continue
            word_index.append((pos, Reference(path.name, line_num, 0, word_num)))
            letters_chunks.append(letters)
            pos += len(letters)

    raw_letters = "".join(letters_chunks)
    display_codes = np.array([ord(c) for c in raw_letters], dtype=np.int32)
    if is_heb and normalize_final_forms:
        match_text = normalize.normalize_finals(raw_letters)
    else:
        match_text = raw_letters
    match_codes = np.array([ord(c) for c in match_text], dtype=np.int32)

    return Corpus(
        name=name or path.name,
        raw_letters=raw_letters,
        display_codes=display_codes,
        match_codes=match_codes,
        word_index=word_index,
    )


def encode_word(word: str, hebrew: Optional[bool] = None, normalize_final_forms: bool = True) -> np.ndarray:
    """Encode a search word into the same code space used by
    Corpus.match_codes, so it can be compared directly."""
    is_heb = normalize.is_hebrew_text(word) if hebrew is None else hebrew
    letters = (
        normalize.strip_to_hebrew_letters(word)
        if is_heb
        else normalize.strip_to_generic_letters(word)
    )
    if is_heb and normalize_final_forms:
        letters = normalize.normalize_finals(letters)
    return np.array([ord(c) for c in letters], dtype=np.int32)
