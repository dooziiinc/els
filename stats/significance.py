"""Monte Carlo significance testing for ELS results.

Every test here follows the same logic: compute a statistic on the
real corpus, recompute the same statistic on many randomized versions
of the corpus (see permutations.py), and report the empirical p-value
-- the fraction of randomized trials that were as extreme or more
extreme than the real observation. This is the same permutation-test
logic WRR relied on, made fully explicit and reproducible here rather
than depending on undisclosed word-form selection.

IMPORTANT CONTEXT: even a well-run permutation test on one word or one
pair, chosen after the fact, is not strong evidence of anything -- see
McKay, Bar-Natan, Bar-Hillel & Gill, "Solving the Bible Code Puzzle"
(Statistical Science, 1999) for the definitive critique of the original
WRR "Great Rabbis" experiment, which turned on exactly this kind of
after-the-fact flexibility (multiple valid spellings/date-forms to
choose from). If you're using this module to genuinely evaluate the
ELS hypothesis rather than just explore the text, pre-register your
word list and forms *before* looking at results, run many
words/pairs (not just the one that "worked"), and correct for multiple
comparisons.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Sequence

import numpy as np

from ..corpus import Corpus, encode_word
from ..search.els import find_minimal_skip_els
from .compactness import compactness
from .permutations import shuffle_letters, shuffle_words

PermutationKind = Literal["letters", "words"]

_PERM_FUNCS = {"letters": shuffle_letters, "words": shuffle_words}


@dataclass
class SignificanceResult:
    statistic_name: str
    observed: Optional[float]
    null_values: List[float] = field(repr=False)
    n_permutations: int = 0
    p_value: float = 1.0
    permutation_kind: str = "letters"
    note: str = ""

    def summary(self) -> str:
        finite = [v for v in self.null_values if np.isfinite(v)]
        mean = float(np.mean(finite)) if finite else float("nan")
        return (
            f"{self.statistic_name}: observed={self.observed}, "
            f"null mean={mean:.2f} (n={self.n_permutations}, "
            f"{self.permutation_kind}-shuffled), p={self.p_value:.4f}"
        )


def minimal_skip_significance(
    corpus: Corpus,
    word: str,
    n_permutations: int = 200,
    permutation: PermutationKind = "letters",
    max_skip_cap: Optional[int] = None,
    directions: Sequence[str] = ("forward", "backward"),
    seed: Optional[int] = None,
) -> SignificanceResult:
    """Is `word`'s minimal-skip ELS occurrence more 'notable' (smaller
    |skip|) than would be expected by chance?

    Null hypothesis: the minimal |skip| at which `word` appears as an
    ELS is no smaller than in a randomized version of the same corpus.
    """
    word_codes = encode_word(word)
    observed_match = find_minimal_skip_els(
        corpus.match_codes, word_codes, word, max_skip_cap=max_skip_cap, directions=directions
    )
    if observed_match is None:
        raise ValueError(f"{word!r} was not found as an ELS in the corpus within the skip cap.")
    observed = abs(observed_match.skip)

    perm_fn = _PERM_FUNCS[permutation]
    rng = np.random.default_rng(seed)
    null_values: List[float] = []
    for _ in range(n_permutations):
        trial_seed = int(rng.integers(0, 2**31 - 1))
        trial_corpus = perm_fn(corpus, seed=trial_seed)
        m = find_minimal_skip_els(
            trial_corpus.match_codes, word_codes, word, max_skip_cap=max_skip_cap, directions=directions
        )
        null_values.append(abs(m.skip) if m is not None else float("inf"))

    p_value = (sum(1 for v in null_values if v <= observed) + 1) / (n_permutations + 1)

    return SignificanceResult(
        statistic_name=f"minimal |skip| for {word!r}",
        observed=observed,
        null_values=null_values,
        n_permutations=n_permutations,
        p_value=p_value,
        permutation_kind=permutation,
        note=(
            "Lower minimal |skip| is considered more 'notable' under the WRR "
            "convention. p = (# permutations with skip <= observed + 1) / (n + 1)."
        ),
    )


def pair_compactness_significance(
    corpus: Corpus,
    word1: str,
    word2: str,
    n_permutations: int = 200,
    permutation: PermutationKind = "letters",
    max_skip_cap: Optional[int] = None,
    directions: Sequence[str] = ("forward", "backward"),
    seed: Optional[int] = None,
) -> SignificanceResult:
    """Do the minimal-skip ELS occurrences of `word1` and `word2` sit in
    a smaller bounding box (see stats.compactness) than would be
    expected by chance?

    Null hypothesis: the bounding-box area enclosing the two words'
    minimal-skip ELS occurrences is no smaller than in a randomized
    version of the same corpus.
    """
    w1_codes = encode_word(word1)
    w2_codes = encode_word(word2)

    m1 = find_minimal_skip_els(corpus.match_codes, w1_codes, word1, max_skip_cap=max_skip_cap, directions=directions)
    m2 = find_minimal_skip_els(corpus.match_codes, w2_codes, word2, max_skip_cap=max_skip_cap, directions=directions)
    if m1 is None or m2 is None:
        missing = word1 if m1 is None else word2
        raise ValueError(f"{missing!r} was not found as an ELS in the corpus within the skip cap.")
    observed = compactness(m1, m2).area

    perm_fn = _PERM_FUNCS[permutation]
    rng = np.random.default_rng(seed)
    null_values: List[float] = []
    for _ in range(n_permutations):
        trial_seed = int(rng.integers(0, 2**31 - 1))
        trial_corpus = perm_fn(corpus, seed=trial_seed)
        pm1 = find_minimal_skip_els(
            trial_corpus.match_codes, w1_codes, word1, max_skip_cap=max_skip_cap, directions=directions
        )
        pm2 = find_minimal_skip_els(
            trial_corpus.match_codes, w2_codes, word2, max_skip_cap=max_skip_cap, directions=directions
        )
        if pm1 is None or pm2 is None:
            null_values.append(float("inf"))
            continue
        null_values.append(compactness(pm1, pm2).area)

    p_value = (sum(1 for v in null_values if v <= observed) + 1) / (n_permutations + 1)

    return SignificanceResult(
        statistic_name=f"bounding-box area for ({word1!r}, {word2!r})",
        observed=observed,
        null_values=null_values,
        n_permutations=n_permutations,
        p_value=p_value,
        permutation_kind=permutation,
        note=(
            "Smaller bounding-box area = more 'compact' under the WRR convention. "
            "p = (# permutations with area <= observed + 1) / (n + 1)."
        ),
    )


def control_text_comparison(
    corpus: Corpus,
    control_corpus: Corpus,
    word: str,
    max_skip_cap: Optional[int] = None,
    directions: Sequence[str] = ("forward", "backward"),
) -> dict:
    """Descriptive (non-p-value) comparison: does `word` appear with an
    unusually small minimal skip in `corpus` compared to some other
    real-world text supplied as a control (e.g. a different large
    Hebrew corpus)? Because the two texts differ in content, this is
    not a valid randomization test -- it's context, in the spirit of
    the check McKay et al. recommend: results that are 'special' to
    one particular text and don't show up anywhere else are less
    convincing than robust, text-independent effects.
    """
    word_codes = encode_word(word)
    main_match = find_minimal_skip_els(
        corpus.match_codes, word_codes, word, max_skip_cap=max_skip_cap, directions=directions
    )
    control_match = find_minimal_skip_els(
        control_corpus.match_codes, word_codes, word, max_skip_cap=max_skip_cap, directions=directions
    )
    return {
        "word": word,
        "corpus_name": corpus.name,
        "corpus_length": len(corpus),
        "corpus_minimal_skip": abs(main_match.skip) if main_match else None,
        "control_name": control_corpus.name,
        "control_length": len(control_corpus),
        "control_minimal_skip": abs(control_match.skip) if control_match else None,
    }
