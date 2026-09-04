"""A small Flask app for interactively searching and visualizing ELS
matches. Run with:

    python -m torahcodes.webapp.app

then open http://127.0.0.1:5000/
"""
from __future__ import annotations

from typing import Optional

from flask import Flask, render_template, request

from ..corpus import BOOK_CHOICES, BOOK_NAMES, Corpus, encode_word, load_book, load_torah
from ..search.els import find_minimal_skip_els
from ..search.grid import Grid, _default_width, vicinity_search

app = Flask(__name__)

_CORPUS_CACHE: dict[str, Corpus] = {}

COLORS = ["#e07a5f", "#3d9970", "#3a86ff", "#d62828", "#8338ec", "#f4a261", "#2a9d8f", "#ff006e"]


def get_corpus(book: Optional[str]) -> Corpus:
    key = book or "ALL"
    if key not in _CORPUS_CACHE:
        _CORPUS_CACHE[key] = load_book(book) if book else load_torah()
    return _CORPUS_CACHE[key]


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", books=BOOK_NAMES, result=None, error=None, form={})


@app.route("/search", methods=["POST"])
def search():
    form = request.form
    word = (form.get("word") or "").strip()
    book = form.get("book") or None
    mode = form.get("mode") or "grid_block"
    candidates_raw = (form.get("candidates") or "").strip()
    candidates = [c.strip() for c in candidates_raw.split(",") if c.strip()]

    def _int(name, default):
        raw = form.get(name)
        try:
            return int(raw) if raw not in (None, "") else default
        except ValueError:
            return default

    width_in = _int("width", None)
    row_margin = _int("row_margin", 5)
    radius = _int("radius", 10)
    max_skip = _int("max_skip", 200)

    form_state = {
        "word": word, "book": book or "", "mode": mode, "candidates": candidates_raw,
        "width": width_in or "", "row_margin": row_margin, "radius": radius, "max_skip": max_skip,
    }

    if not word:
        return render_template("index.html", books=BOOK_NAMES, result=None, error="Please enter a word to search for.", form=form_state)

    corpus = get_corpus(book)
    try:
        word_codes = encode_word(word)
    except Exception as exc:  # pragma: no cover - defensive
        return render_template("index.html", books=BOOK_NAMES, result=None, error=str(exc), form=form_state)

    if len(word_codes) == 0:
        return render_template("index.html", books=BOOK_NAMES, result=None, error="That word had no letters left after normalization.", form=form_state)

    primary = find_minimal_skip_els(corpus.match_codes, word_codes, word)
    if primary is None:
        return render_template(
            "index.html", books=BOOK_NAMES, result=None,
            error=f"'{word}' was not found as an ELS anywhere in {corpus.name}.", form=form_state,
        )

    width = width_in or _default_width(corpus, primary)
    grid = Grid(corpus, width)
    r0, c0 = grid.to_rowcol(primary.min_index())
    r1, c1 = grid.to_rowcol(primary.max_index())
    row_lo = max(0, min(r0, r1) - row_margin)
    row_hi = min(grid.rows - 1, max(r0, r1) + row_margin)
    # Keep the rendered panel from becoming enormous in a browser.
    if row_hi - row_lo > 120:
        row_hi = row_lo + 120

    hits = []
    if candidates:
        kwargs = dict(width=width, min_skip=1, max_skip=max_skip, limit=150)
        if mode == "grid_block":
            kwargs["row_margin"] = row_margin
        else:
            kwargs["radius"] = radius
        hits = vicinity_search(corpus, primary, candidates, mode=mode, **kwargs)

    color_for_word = {}
    for w in candidates:
        color_for_word[w] = COLORS[len(color_for_word) % len(COLORS)]

    highlight = {}
    for p in primary.positions():
        highlight[p] = (color_for_word.get(word, "#222"), word, True)
    for h in hits:
        color = color_for_word.get(h.word, "#999")
        for p in h.match.positions():
            highlight.setdefault(p, (color, h.word, False))

    rows = []
    for r in range(row_lo, row_hi + 1):
        row_cells = []
        for c in range(width):
            pos = grid.to_pos(r, c)
            if pos is None:
                row_cells.append(None)
                continue
            ch = corpus.raw_letters[pos]
            color, title, is_primary = highlight.get(pos, (None, None, False))
            row_cells.append({"char": ch, "color": color, "title": title, "primary": is_primary})
        rows.append(row_cells)

    ref = corpus.reference_at(primary.start)
    hit_rows = []
    for h in hits:
        href = corpus.reference_at(h.match.start)
        hit_rows.append({
            "word": h.word, "start": h.match.start, "skip": h.match.skip,
            "reference": str(href) if href else None, "color": color_for_word.get(h.word, "#999"),
        })

    result = {
        "word": word, "start": primary.start, "skip": primary.skip,
        "reference": str(ref) if ref else None,
        "width": width, "row_lo": row_lo, "row_hi": row_hi,
        "rows": rows, "hits": hit_rows, "corpus_name": corpus.name,
        "corpus_len": len(corpus), "mode": mode,
        "legend": [{"word": word, "color": color_for_word.get(word, "#222")}] + [
            {"word": w, "color": c} for w, c in color_for_word.items()
        ],
    }
    return render_template("index.html", books=BOOK_NAMES, result=result, error=None, form=form_state)


def main():
    app.run(debug=True, host="127.0.0.1", port=5000)


if __name__ == "__main__":
    main()
