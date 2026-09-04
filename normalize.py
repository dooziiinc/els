"""Text normalization utilities.

The ELS search engine works on a bare stream of "letters" -- no spaces,
no vowels/cantillation, no punctuation. This module turns arbitrary raw
text (Hebrew or otherwise) into that stream, and provides a Hebrew
final-letter-form normalizer used for *matching* (final and regular
forms of the same letter are treated as equivalent when searching,
which is the standard convention in ELS/Bible-code literature, since
sofit forms are a purely graphical convention with no phonemic
difference) while the original form is preserved for display.
"""
from __future__ import annotations

import re
import unicodedata

# Base Hebrew consonants (alef..tav) plus the five final forms.
HEBREW_LETTER_RANGE = re.compile(r"[א-ת]")

# Final-form -> regular-form mapping, used only for matching purposes.
FINAL_TO_REGULAR = {
    "ך": "כ",  # ך -> כ
    "ם": "מ",  # ם -> מ
    "ן": "נ",  # ן -> נ
    "ף": "פ",  # ף -> פ
    "ץ": "צ",  # ץ -> צ
}


def strip_to_hebrew_letters(text: str) -> str:
    """Remove niqqud, cantillation marks, maqqef, punctuation, spaces,
    digits, and any morpheme-boundary markers (e.g. the '/' used by the
    OpenScriptures WLC edition), leaving only the 27 Hebrew consonant
    letter forms (22 base + 5 final)."""
    return "".join(HEBREW_LETTER_RANGE.findall(text))


def normalize_finals(text: str) -> str:
    """Map final letter forms to their regular form. Used to build the
    *matching* representation of a corpus/search word; the raw text
    (with final forms intact) is kept separately for display."""
    return "".join(FINAL_TO_REGULAR.get(ch, ch) for ch in text)


def strip_to_generic_letters(text: str) -> str:
    """A generic (non-Hebrew) normalizer for comparison corpora in other
    scripts: keep only letters (any Unicode letter category), uppercase
    Latin-script text so case doesn't affect matching."""
    out = []
    for ch in text:
        if unicodedata.category(ch).startswith("L"):
            out.append(ch.upper() if ch.isascii() else ch)
    return "".join(out)


def is_hebrew_text(text: str, sample_size: int = 2000) -> bool:
    """Heuristic: does this text look like it's predominantly Hebrew?"""
    sample = text[:sample_size]
    letters = [c for c in sample if c.isalpha()]
    if not letters:
        return False
    hebrew = sum(1 for c in letters if "֐" <= c <= "׿")
    return hebrew / len(letters) > 0.5
