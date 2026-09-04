#!/usr/bin/env python3
"""
Build a clean, continuous consonantal Hebrew letter stream for the Torah
from the OpenScriptures Hebrew Bible (WLC) OSIS XML files, plus a compact
index that lets any letter position be mapped back to (book, chapter,
verse, word number).

Source: https://github.com/openscriptures/morphhb  (WLC text: Public
Domain; OSHB morphological tagging: CC BY 4.0 -- we only use the text,
not the tagging, but the source is preserved either way.)

Output (written to data/):
  torah_letters.txt   -- one line, the entire Torah as a bare consonant
                          string (Genesis..Deuteronomy), no spaces, no
                          niqqud/cantillation, final letter forms kept
                          as written in the source.
  torah_index.json    -- word-boundary index: a list of
                          [start_pos, book, chapter, verse, word_num]
                          entries, one per word, in reading order.
  book_boundaries.json-- {book: [start_pos, end_pos_exclusive]} for the
                          five books, for convenience.
"""
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

NS = {"osis": "http://www.bibletechnologies.net/2003/OSIS/namespace"}

BOOKS = ["Gen", "Exod", "Lev", "Num", "Deut"]

HEB_LETTER_RE = re.compile(r"[א-ת]")

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw_wlc"
OUT_DIR = Path(__file__).resolve().parent.parent / "data"


def extract_letters(word_text: str) -> str:
    """Strip niqqud, cantillation marks, the morpheme-boundary '/' used
    by OSHB, and any other non-consonant characters, leaving only the
    22 base Hebrew consonants + 5 final forms."""
    return "".join(HEB_LETTER_RE.findall(word_text))


def parse_book(book_code: str):
    """Return a list of (letters, book, chapter, verse, word_num) tuples,
    one per word, in reading order, for a single book."""
    path = RAW_DIR / f"{book_code}.xml"
    tree = ET.parse(path)
    root = tree.getroot()

    words = []
    for verse in root.iter("{http://www.bibletechnologies.net/2003/OSIS/namespace}verse"):
        osis_id = verse.get("osisID")
        if not osis_id:
            continue
        # osisID looks like "Gen.1.1"
        _, chapter, verse_num = osis_id.split(".")
        chapter = int(chapter)
        verse_num = int(verse_num)
        word_num = 0
        for w in verse.iter("{http://www.bibletechnologies.net/2003/OSIS/namespace}w"):
            text = w.text or ""
            letters = extract_letters(text)
            if not letters:
                continue
            word_num += 1
            words.append((letters, book_code, chapter, verse_num, word_num))
    return words


def main():
    all_letters = []
    index = []  # [start_pos, book, chapter, verse, word_num]
    book_boundaries = {}
    pos = 0

    for book_code in BOOKS:
        book_start = pos
        words = parse_book(book_code)
        for letters, book, chapter, verse_num, word_num in words:
            index.append([pos, book, chapter, verse_num, word_num])
            all_letters.append(letters)
            pos += len(letters)
        book_boundaries[book_code] = [book_start, pos]
        print(f"{book_code}: {pos - book_start} letters, {len(words)} words")

    full_text = "".join(all_letters)
    print(f"TOTAL: {len(full_text)} letters")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "torah_letters.txt").write_text(full_text, encoding="utf-8")
    (OUT_DIR / "torah_index.json").write_text(
        json.dumps(index, ensure_ascii=False), encoding="utf-8"
    )
    (OUT_DIR / "book_boundaries.json").write_text(
        json.dumps(book_boundaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
