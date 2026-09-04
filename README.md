# torahcodes

A self-contained toolkit for searching the Hebrew Torah for **Equidistant
Letter Sequences (ELS)** -- the "Bible code" technique associated with
Dr. Eliyahu Rips and the 1994 Witztum-Rips-Rosenberg (WRR) paper -- with
tools to visualize hits in a 2-D letter grid, search for other words
nearby, and run permutation-based statistical significance tests.

It ships with the full Torah (Genesis-Deuteronomy) as clean Hebrew
consonantal text, and lets you load any additional plain-text corpus
(Hebrew or otherwise) for comparison.

References this implementation is informed by:
- Witztum, Rips & Rosenberg, "Equidistant Letter Sequences in the Book
  of Genesis", *Statistical Science* 9(3), 1994.
- McKay, Bar-Natan, Bar-Hillel & Gill, "Solving the Bible Code
  Puzzle", *Statistical Science* 14(2), 1999 -- the definitive
  rebuttal, showing the original result depended on unprincipled
  choices in word-form selection.
- https://www.math.toronto.edu/drorbn/Codes/Nations/WRR2/index.html
- https://users.cecs.anu.edu.au/~bdm/dilugim/Nations/WRR2/index.html
- https://torahbiblecodes.com/

**This is a research/exploration tool, not a verdict.** See
"Honest limitations" below before drawing conclusions from any single
result.

## What's here

```
torahcodes/
  normalize.py        Hebrew/generic text normalization
  corpus.py            Corpus loading (Torah, single books, user text files)
  search/
    els.py              Core ELS search (skip search, minimal-skip search)
    grid.py              2-D grid reshaping + "vicinity" search
  stats/
    compactness.py       WRR-style bounding-box compactness measure
    permutations.py       Randomized-control corpus generation
    significance.py        Monte Carlo significance tests
  cli.py                Command-line interface
  webapp/               Flask web UI (form + visual grid)
scripts/
  build_torah_text.py   Rebuilds data/ from the raw WLC XML
data/
  raw_wlc/               Source XML (Westminster Leningrad Codex, via OpenScriptures)
  torah_letters.txt        Prebuilt: the whole Torah as one clean letter stream
  torah_index.json          Prebuilt: letter-position -> (book, chapter, verse, word) index
  book_boundaries.json       Prebuilt: byte ranges for each of the 5 books
tests/                  pytest suite
```

## Install

```bash
pip install -e .
# or: pip install -r requirements.txt
```

Requires Python 3.9+. The bundled Torah text is already built (see
`data/`), so no network access is needed to start searching. To rebuild
it from source (e.g. after editing `scripts/build_torah_text.py`):

```bash
python scripts/build_torah_text.py
```

## Quickstart: command line

```bash
# Every ELS occurrence of a word (skip 1-500, both directions)
torahcodes search "משה" --max-skip 500

# Just the single most notable occurrence (smallest |skip|)
torahcodes minimal "תורה"

# Render the 2-D grid panel around a word's minimal-skip occurrence
torahcodes grid "משה" --row-margin 8

# Look for other words nearby, once the text is laid out as a grid
torahcodes vicinity "משה" -n "אלהים" -n "תורה" --mode radius --radius 15

# Statistical significance (permutation test)
torahcodes stats minimal-skip "משה" --n-permutations 200
torahcodes stats pair "משה" "תורה" --n-permutations 200
torahcodes stats control "משה" --control-corpus path/to/other_hebrew_text.txt
```

Run `torahcodes --help` or `torahcodes COMMAND --help` for the full
option list (skip ranges, direction, which book, a user-supplied
`--corpus` file, JSON output, etc).

## Quickstart: web UI

```bash
python -m torahcodes.webapp.app
```

Then open http://127.0.0.1:5000/ -- enter a word, optionally a
comma-separated list of "vicinity" words to search for nearby, and see
the highlighted grid.

## How it works

**The corpus.** The Torah text comes from the Westminster Leningrad
Codex (WLC), via the [OpenScriptures Hebrew
Bible](https://github.com/openscriptures/morphhb) project (WLC text:
public domain; OSHB tagging: CC BY 4.0 -- only the bare text is used
here). `scripts/build_torah_text.py` strips niqqud (vowel points),
cantillation marks, spaces, and verse punctuation, leaving a single
continuous string of 305,167 Hebrew consonant letters (Genesis through
Deuteronomy) -- close to, but not identical to, the traditional count
of 304,805, since letter counts vary slightly across Masoretic source
editions. Every letter's position is indexed back to its
(book, chapter, verse, word) reference.

By default, final letter forms (ך ם ן ף ץ) are treated as equivalent
to their regular form (כ מ נ פ צ) for *matching* purposes (this is the
standard ELS convention, since the forms are a purely graphical
convention with no phonemic difference); the original form is always
preserved for display. Use `--no-normalize-finals` / `normalize_final_forms=False`
to disable this.

**ELS search** (`search/els.py`). An ELS of a word is a start position
and a nonzero integer skip `d` such that reading every `d`-th letter
from the start spells the word. Positive skip reads forward, negative
backward; `|d| = 1` is an ordinary substring. The search is vectorized
with numpy. `find_minimal_skip_els` efficiently finds just the
smallest-|skip| occurrence (the "most notable" one, by WRR's own
convention) without enumerating every skip first -- this is what makes
permutation testing (below) practical.

**The grid and vicinity search** (`search/grid.py`). Reshape the flat
letter stream into rows of a chosen width (conventionally the primary
word's skip, so it displays as a straight column) and search for other
words nearby. Two modes, both configurable:
- `grid_block`: a candidate word's *entire* ELS must fall inside the
  displayed rows (optionally also inside a column range) -- "does this
  word appear in the same panel", matching the visual displays on the
  reference sites.
- `radius`: only a candidate's *starting cell* needs to be within a
  given (row, column) distance of the primary word's start -- cheaper,
  and doesn't require the whole word to stay on-panel.

Short candidate words (2-3 letters) will often turn up "nearby" by
chance alone, especially with a generous skip range -- that's an
expected property of the method, not a bug, and is exactly the kind of
thing that makes short words weak evidence on their own.

**Statistics** (`stats/`). WRR argued that *meaningfully related* word
pairs (e.g. a rabbi's name and birth/death date) sit closer together
("more compact") as ELSs than chance would predict. WRR's own
compactness formula was never published in a fully reproducible form,
and the McKay et al. rebuttal showed the original result hinged on
after-the-fact flexibility in which spelling/date-form to use for each
person. This toolkit does **not** try to reproduce that specific,
disputed statistic. Instead it gives you a clearly-specified,
fully-reproducible alternative:

- `compactness()`: a deterministic bounding-box measure between two
  ELS matches, in the 2-D grid.
- `minimal_skip_significance()` / `pair_compactness_significance()`:
  Monte Carlo permutation tests. The real statistic (minimal |skip|,
  or bounding-box area for a pair) is compared against the same
  statistic computed on many randomized versions of the corpus --
  either a full letter-frequency-preserving shuffle (`permutation="letters"`,
  the strongest null), or a word-order shuffle that keeps every word
  intact (`permutation="words"`, more conservative). The result is an
  empirical p-value: the fraction of random trials at least as
  "notable" as the real text.
- `control_text_comparison()`: descriptive-only comparison of a word's
  minimal skip in the Torah versus in some other real corpus you
  supply (not a formal test, since two different real texts aren't
  interchangeable the way random permutations are -- but a useful
  sanity check, since an effect that's "special" to one particular
  text is much less convincing than one that doesn't show up anywhere
  else).

### Honest limitations

- A single low p-value on one word or one pair, chosen *after* looking
  at results, proves very little -- with enough candidate words,
  spellings, and skip ranges to choose from, some will look "notable"
  by pure chance. If you want to genuinely test the ELS hypothesis
  rather than explore the text, decide your word list and exact
  spellings *before* running anything, test many words/pairs (not just
  the ones that "work"), and correct for multiple comparisons.
- The bounded default `--max-skip` in `vicinity` (200) and the result
  `limit` are there so short/common words don't produce unusable
  amounts of output; raising them doesn't make a result more
  meaningful, and often does the opposite.
- This is one particular manuscript edition (WLC) of one particular
  Masoretic tradition. Other editions have small letter-count
  differences, and WRR-style critics have shown some published results
  are sensitive to exactly this.

## Adding a comparison corpus

```python
from torahcodes.corpus import load_text_file
corpus = load_text_file("my_text.txt", name="War and Peace (Hebrew)")
```

or from the CLI, pass `--corpus path/to/file.txt` to any command. Hebrew
vs. other scripts is auto-detected (override with `--hebrew`/`--no-hebrew`).
Non-Hebrew text is normalized by keeping letters only and uppercasing
ASCII, so the same ELS engine works on it unchanged.

## Tests

```bash
pip install -e ".[dev]"
pytest
```
