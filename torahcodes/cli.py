"""Command-line interface for torahcodes.

    torahcodes search WORD              # find every ELS occurrence
    torahcodes minimal WORD             # find the single most-notable (smallest |skip|) occurrence
    torahcodes grid WORD                # render the grid panel around the minimal-skip occurrence
    torahcodes vicinity WORD -n W1 -n W2 # search for other words near WORD's ELS
    torahcodes stats minimal-skip WORD  # permutation significance test on minimal |skip|
    torahcodes stats pair WORD1 WORD2   # permutation significance test on pair compactness
    torahcodes stats control WORD --control-corpus other.txt

Run `torahcodes COMMAND --help` for the full option list of any command.
"""
from __future__ import annotations

import json as json_lib
import sys
from functools import reduce
from typing import Optional, Sequence, Tuple

import click

from .corpus import Corpus, encode_word, load_book, load_text_file, load_torah
from .search.els import els_search, find_minimal_skip_els, minimal_skip_match
from .search.grid import Grid, vicinity_search
from .stats.significance import (
    control_text_comparison,
    minimal_skip_significance,
    pair_compactness_significance,
)

BOOK_CHOICES = ["Gen", "Exod", "Lev", "Num", "Deut"]


def corpus_options(f):
    options = [
        click.option(
            "--corpus",
            "corpus_path",
            type=click.Path(exists=True),
            default=None,
            help="Path to a plain-text file to search instead of the Torah.",
        ),
        click.option(
            "--book",
            type=click.Choice(BOOK_CHOICES),
            default=None,
            help="Restrict the search to one Torah book instead of all five.",
        ),
        click.option(
            "--hebrew/--no-hebrew",
            "hebrew",
            default=None,
            help="Force Hebrew/generic-script normalization for --corpus (default: auto-detect).",
        ),
        click.option(
            "--no-normalize-finals",
            is_flag=True,
            default=False,
            help="Don't treat Hebrew final letter forms (ך ם ן ף ץ) as equivalent to their regular form.",
        ),
    ]
    return reduce(lambda g, opt: opt(g), reversed(options), f)


def _load_corpus(corpus_path, book, hebrew, no_normalize_finals) -> Corpus:
    normalize_final_forms = not no_normalize_finals
    if corpus_path:
        return load_text_file(corpus_path, hebrew=hebrew, normalize_final_forms=normalize_final_forms)
    if book:
        return load_book(book, normalize_final_forms=normalize_final_forms)
    return load_torah(normalize_final_forms=normalize_final_forms)


def _directions(direction: str) -> Tuple[str, ...]:
    return {"forward": ("forward",), "backward": ("backward",), "both": ("forward", "backward")}[direction]


direction_option = click.option(
    "--direction",
    type=click.Choice(["forward", "backward", "both"]),
    default="both",
    show_default=True,
    help="Which skip directions to search.",
)


@click.group()
def cli():
    """ELS (Equidistant Letter Sequence) search and statistical testing
    over the Hebrew Torah, with support for user-provided comparison
    corpora."""


@cli.command()
@click.argument("word")
@corpus_options
@click.option("--min-skip", default=1, show_default=True, type=int)
@click.option("--max-skip", default=None, type=int, help="Default: as large as the word can still fit.")
@direction_option
@click.option("--limit", default=25, show_default=True, type=int, help="Max rows to print (sorted by smallest |skip| first). 0 = no limit.")
@click.option("--json", "as_json", is_flag=True, help="Output JSON instead of a table.")
def search(word, corpus_path, book, hebrew, no_normalize_finals, min_skip, max_skip, direction, limit, as_json):
    """Find every ELS occurrence of WORD."""
    corpus = _load_corpus(corpus_path, book, hebrew, no_normalize_finals)
    word_codes = encode_word(word, hebrew=hebrew, normalize_final_forms=not no_normalize_finals)
    matches = els_search(
        corpus.match_codes, word_codes, word_str=word,
        min_skip=min_skip, max_skip=max_skip, directions=_directions(direction),
    )
    total = len(matches)
    shown = matches if limit == 0 else matches[:limit]

    if as_json:
        rows = [_match_row(corpus, m) for m in shown]
        click.echo(json_lib.dumps({"corpus": corpus.name, "word": word, "total_matches": total, "matches": rows}, ensure_ascii=False, indent=2))
        return

    click.echo(f"Corpus: {corpus.name} ({len(corpus):,} letters)")
    click.echo(f"'{word}': {total:,} ELS occurrence(s) with {min_skip} <= |skip| <= {max_skip or 'max'}")
    if total == 0:
        return
    click.echo(f"Showing {len(shown)} (smallest |skip| first):\n")
    for m in shown:
        ref = corpus.reference_at(m.start)
        ref_str = f"  [{ref}]" if ref else ""
        click.echo(f"  start={m.start:<8} skip={m.skip:<6}{ref_str}")


@cli.command()
@click.argument("word")
@corpus_options
@click.option("--max-skip-cap", default=None, type=int, help="Give up searching beyond this |skip|. Default: as large as the word can fit.")
@direction_option
def minimal(word, corpus_path, book, hebrew, no_normalize_finals, max_skip_cap, direction):
    """Find WORD's single most-notable ELS occurrence (smallest |skip|)."""
    corpus = _load_corpus(corpus_path, book, hebrew, no_normalize_finals)
    word_codes = encode_word(word, hebrew=hebrew, normalize_final_forms=not no_normalize_finals)
    m = find_minimal_skip_els(corpus.match_codes, word_codes, word, max_skip_cap=max_skip_cap, directions=_directions(direction))
    if m is None:
        click.echo(f"'{word}' was not found as an ELS in {corpus.name} (within the skip cap).")
        sys.exit(1)
    ref = corpus.reference_at(m.start)
    click.echo(f"Corpus: {corpus.name}")
    click.echo(f"'{word}': minimal |skip| = {abs(m.skip)}  (start={m.start}, skip={m.skip})")
    if ref:
        click.echo(f"Reference: {ref}")


@cli.command()
@click.argument("word")
@corpus_options
@click.option("--start", default=None, type=int, help="Use this start position instead of searching for WORD's minimal-skip occurrence.")
@click.option("--skip", default=None, type=int, help="Use this skip instead of searching for WORD's minimal-skip occurrence.")
@click.option("--width", default=None, type=int, help="Grid row width. Default: |skip| of the match (falls back to a square-ish width if |skip| <= 1).")
@click.option("--row-margin", default=5, show_default=True, type=int, help="Extra rows of context above/below the match.")
def grid(word, corpus_path, book, hebrew, no_normalize_finals, start, skip, width, row_margin):
    """Render the 2-D grid panel around WORD's ELS occurrence, with the
    word highlighted (marked with []) column-by-column."""
    corpus = _load_corpus(corpus_path, book, hebrew, no_normalize_finals)
    if start is not None and skip is not None:
        from .search.els import ELSMatch
        m = ELSMatch(word, start, skip, len(encode_word(word)))
    else:
        word_codes = encode_word(word, hebrew=hebrew, normalize_final_forms=not no_normalize_finals)
        m = find_minimal_skip_els(corpus.match_codes, word_codes, word)
        if m is None:
            click.echo(f"'{word}' was not found as an ELS in {corpus.name}.")
            sys.exit(1)

    from .search.grid import _default_width
    w = width or _default_width(corpus, m)
    g = Grid(corpus, w)
    positions = set(m.positions())
    r0, c0 = g.to_rowcol(m.min_index())
    r1, c1 = g.to_rowcol(m.max_index())
    row_lo, row_hi = max(0, min(r0, r1) - row_margin), min(g.rows - 1, max(r0, r1) + row_margin)

    click.echo(f"Corpus: {corpus.name}  |  width={w}  rows {row_lo}-{row_hi}  |  '{word}' start={m.start} skip={m.skip}\n")
    for r in range(row_lo, row_hi + 1):
        line = []
        for c in range(w):
            pos = g.to_pos(r, c)
            if pos is None:
                line.append(" ")
                continue
            ch = corpus.raw_letters[pos]
            line.append(f"[{ch}]" if pos in positions else f" {ch} ")
        click.echo("".join(line))


@cli.command()
@click.argument("word")
@corpus_options
@click.option("--near", "-n", "candidates", multiple=True, required=True, help="A candidate word to search for nearby. Repeat -n for multiple.")
@click.option("--mode", type=click.Choice(["grid_block", "radius"]), default="grid_block", show_default=True)
@click.option("--width", default=None, type=int, help="Grid row width. Default: |skip| of WORD's minimal occurrence (or a square-ish fallback).")
@click.option("--row-margin", default=0, show_default=True, type=int, help="[grid_block] extra rows of padding.")
@click.option("--radius", default=10, show_default=True, type=int, help="[radius] Chebyshev row/col radius.")
@click.option("--min-skip", default=1, show_default=True, type=int)
@click.option("--max-skip", default=200, show_default=True, type=int, help="Cap on candidate skip; short words match very often at large skips, so this is bounded by default.")
@direction_option
@click.option("--limit", default=25, show_default=True, type=int)
@click.option("--json", "as_json", is_flag=True)
def vicinity(word, corpus_path, book, hebrew, no_normalize_finals, candidates, mode, width, row_margin, radius, min_skip, max_skip, direction, limit, as_json):
    """Search for CANDIDATE words appearing near WORD's ELS occurrence,
    once the text is arranged as a 2-D grid."""
    corpus = _load_corpus(corpus_path, book, hebrew, no_normalize_finals)
    word_codes = encode_word(word, hebrew=hebrew, normalize_final_forms=not no_normalize_finals)
    primary = find_minimal_skip_els(corpus.match_codes, word_codes, word)
    if primary is None:
        click.echo(f"'{word}' was not found as an ELS in {corpus.name}.")
        sys.exit(1)

    kwargs = dict(width=width, min_skip=min_skip, max_skip=max_skip, directions=_directions(direction), limit=limit)
    if mode == "grid_block":
        kwargs["row_margin"] = row_margin
    else:
        kwargs["radius"] = radius
    hits = vicinity_search(corpus, primary, list(candidates), mode=mode, **kwargs)

    if as_json:
        payload = {
            "corpus": corpus.name,
            "word": word,
            "primary_match": _match_row(corpus, primary),
            "mode": mode,
            "hits": [{"word": h.word, **_match_row(corpus, h.match), "grid_width": h.grid_width} for h in hits],
        }
        click.echo(json_lib.dumps(payload, ensure_ascii=False, indent=2))
        return

    click.echo(f"Corpus: {corpus.name}")
    ref = corpus.reference_at(primary.start)
    click.echo(f"Primary: '{word}' start={primary.start} skip={primary.skip}" + (f"  [{ref}]" if ref else ""))
    click.echo(f"Mode: {mode}  |  {len(hits)} hit(s) among {list(candidates)}\n")
    for h in hits:
        r = corpus.reference_at(h.match.start)
        click.echo(f"  {h.word:<12} start={h.match.start:<8} skip={h.match.skip:<6}" + (f"  [{r}]" if r else ""))


@cli.group()
def stats():
    """Permutation-based statistical significance testing."""


@stats.command("minimal-skip")
@click.argument("word")
@corpus_options
@click.option("--n-permutations", default=200, show_default=True, type=int)
@click.option("--permutation", type=click.Choice(["letters", "words"]), default="letters", show_default=True)
@click.option("--max-skip-cap", default=None, type=int)
@direction_option
@click.option("--seed", default=None, type=int)
def stats_minimal_skip(word, corpus_path, book, hebrew, no_normalize_finals, n_permutations, permutation, max_skip_cap, direction, seed):
    """Is WORD's minimal |skip| smaller than chance would predict?"""
    corpus = _load_corpus(corpus_path, book, hebrew, no_normalize_finals)
    result = minimal_skip_significance(
        corpus, word, n_permutations=n_permutations, permutation=permutation,
        max_skip_cap=max_skip_cap, directions=_directions(direction), seed=seed,
    )
    click.echo(result.summary())
    click.echo(result.note)


@stats.command("pair")
@click.argument("word1")
@click.argument("word2")
@corpus_options
@click.option("--n-permutations", default=200, show_default=True, type=int)
@click.option("--permutation", type=click.Choice(["letters", "words"]), default="letters", show_default=True)
@click.option("--max-skip-cap", default=None, type=int)
@direction_option
@click.option("--seed", default=None, type=int)
def stats_pair(word1, word2, corpus_path, book, hebrew, no_normalize_finals, n_permutations, permutation, max_skip_cap, direction, seed):
    """Are WORD1 and WORD2's minimal-skip ELS occurrences more compact
    (closer together in the grid) than chance would predict?"""
    corpus = _load_corpus(corpus_path, book, hebrew, no_normalize_finals)
    result = pair_compactness_significance(
        corpus, word1, word2, n_permutations=n_permutations, permutation=permutation,
        max_skip_cap=max_skip_cap, directions=_directions(direction), seed=seed,
    )
    click.echo(result.summary())
    click.echo(result.note)


@stats.command("control")
@click.argument("word")
@click.option("--control-corpus", type=click.Path(exists=True), required=True, help="A comparison text file.")
@corpus_options
@click.option("--max-skip-cap", default=None, type=int)
@direction_option
def stats_control(word, control_corpus, corpus_path, book, hebrew, no_normalize_finals, max_skip_cap, direction):
    """Compare WORD's minimal |skip| in the Torah against a
    user-supplied control text (descriptive, not a formal p-value)."""
    corpus = _load_corpus(corpus_path, book, hebrew, no_normalize_finals)
    control = load_text_file(control_corpus, hebrew=hebrew, normalize_final_forms=not no_normalize_finals)
    result = control_text_comparison(corpus, control, word, max_skip_cap=max_skip_cap, directions=_directions(direction))
    click.echo(json_lib.dumps(result, ensure_ascii=False, indent=2))


def _match_row(corpus: Corpus, m) -> dict:
    ref = corpus.reference_at(m.start)
    return {
        "word": m.word,
        "start": m.start,
        "skip": m.skip,
        "length": m.length,
        "reference": str(ref) if ref else None,
    }


if __name__ == "__main__":
    cli()
