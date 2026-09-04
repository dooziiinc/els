"""Randomized controls for significance testing.

Two null models are provided, corresponding to two different (and both
defensible) questions:

  * shuffle_letters -- destroys ALL structure, keeping only the
    corpus's letter-frequency table. Answers "is this more compact/
    lower-skip than a bag of the same letters in random order?" This
    is the strongest, least text-like null.

  * shuffle_words -- keeps every word exactly as it was written (so
    real Hebrew morphology, letter adjacency *within* words, and
    overall word-length distribution are preserved) but randomizes the
    order the words appear in. Answers the more conservative question
    "is this more compact/lower-skip than the same words in a random
    order?"

Both return a new Corpus with the same length and letter-frequency
profile as the original, but no valid book/chapter/verse references
(word_index=None), since those references only make sense for the
real, ordered text.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..corpus import Corpus


def shuffle_letters(corpus: Corpus, seed: Optional[int] = None) -> Corpus:
    rng = np.random.default_rng(seed)
    n = len(corpus)
    perm = rng.permutation(n)
    display_codes = corpus.display_codes[perm]
    match_codes = corpus.match_codes[perm]
    raw_letters = "".join(chr(c) for c in display_codes)
    return Corpus(
        name=f"{corpus.name} [letter-shuffled]",
        raw_letters=raw_letters,
        display_codes=display_codes,
        match_codes=match_codes,
        word_index=None,
    )


def shuffle_words(corpus: Corpus, seed: Optional[int] = None) -> Corpus:
    if corpus.word_index is None:
        raise ValueError(
            "shuffle_words requires a corpus with a word index "
            "(load_torah()/load_text_file() results have one; a "
            "previously shuffled corpus does not)."
        )
    rng = np.random.default_rng(seed)

    starts = [p for p, _ in corpus.word_index] + [len(corpus)]
    words = [corpus.raw_letters[starts[i] : starts[i + 1]] for i in range(len(corpus.word_index))]
    order = rng.permutation(len(words))
    shuffled_words = [words[i] for i in order]

    raw_letters = "".join(shuffled_words)
    display_codes = np.array([ord(c) for c in raw_letters], dtype=np.int32)

    # Re-derive match_codes the same way the original corpus did, by
    # comparing whether the original match_codes differed from
    # display_codes per-letter is unsafe across a reorder; instead just
    # re-normalize using the same final-forms rule the source used. We
    # detect that rule by checking whether corpus already had any
    # final-form letter mapped differently in match vs raw.
    from .. import normalize

    normalized_originally = not np.array_equal(corpus.display_codes, corpus.match_codes)
    if normalized_originally:
        match_text = normalize.normalize_finals(raw_letters)
    else:
        match_text = raw_letters
    match_codes = np.array([ord(c) for c in match_text], dtype=np.int32)

    return Corpus(
        name=f"{corpus.name} [word-shuffled]",
        raw_letters=raw_letters,
        display_codes=display_codes,
        match_codes=match_codes,
        word_index=None,
    )
