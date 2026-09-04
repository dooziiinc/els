from torahcodes.corpus import encode_word, load_book, load_torah


def test_load_torah_starts_with_genesis_1_1():
    corpus = load_torah()
    # Genesis 1:1-2 with spaces/niqqud/cantillation removed:
    # "בראשית ברא אלהים את השמים ואת הארץ והארץ..."
    assert corpus.raw_letters.startswith("בראשיתבראאלהיםאתהשמיםואתהארץוהארץהיתהתהו")
    assert len(corpus) > 300_000  # traditionally ~304,805 letters; exact count
    # depends on source-text edition (WLC here).


def test_reference_lookup_first_word():
    corpus = load_torah()
    ref = corpus.reference_at(0)
    assert ref is not None
    assert ref.book == "Gen"
    assert ref.chapter == 1
    assert ref.verse == 1
    assert ref.word_num == 1


def test_load_book_matches_slice_of_full_torah():
    full = load_torah()
    genesis = load_book("Gen")
    assert genesis.raw_letters == full.raw_letters[: len(genesis)]


def test_encode_word_strips_niqqud_and_normalizes_finals():
    # "מֶלֶך" with niqqud should encode the same as "מלך"; final kaf (ך)
    # normalizes to regular kaf (כ) by default.
    plain = encode_word("מלך")
    pointed = encode_word("מֶלֶךְ")
    assert list(plain) == list(pointed)
    assert list(encode_word("מלכ")) == list(plain)  # final/regular kaf equivalence
