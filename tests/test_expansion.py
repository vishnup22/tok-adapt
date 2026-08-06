"""Tests for tok_adapt.expansion.VocabularyExpander."""

from __future__ import annotations

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from tok_adapt.expansion import VocabularyExpander


@pytest.fixture(scope="module")
def base_tokenizer():
    try:
        return AutoTokenizer.from_pretrained("gpt2")
    except Exception as exc:  # pragma: no cover - network-dependent
        pytest.skip(f"Could not download base tokenizer 'gpt2': {exc}")


@pytest.fixture(scope="module")
def dummy_corpus(tmp_path_factory):
    corpus_dir = tmp_path_factory.mktemp("corpus")
    corpus_file = corpus_dir / "corpus.txt"
    lines = [
        "quirklex florbin trantak zestimo",
        "quirklex florbin appears frequently in this tiny domain corpus",
        "trantak zestimo trantak zestimo trantak zestimo",
        "florbin quirklex zestimo trantak repeated many times over",
    ] * 25
    corpus_file.write_text("\n".join(lines), encoding="utf-8")
    return corpus_file


def test_train_sub_tokenizer_creates_file(base_tokenizer, dummy_corpus):
    expander = VocabularyExpander(base_tokenizer)
    out_path = expander.train_sub_tokenizer(str(dummy_corpus), vocab_size=300, algorithm="bpe")
    assert out_path.exists()
    assert out_path.suffix == ".json"


def test_train_sub_tokenizer_rejects_unknown_algorithm(base_tokenizer, dummy_corpus):
    expander = VocabularyExpander(base_tokenizer)
    with pytest.raises(ValueError):
        expander.train_sub_tokenizer(str(dummy_corpus), vocab_size=300, algorithm="wordpiece")


def test_merge_vocabularies_extends_without_mutating_base(base_tokenizer, dummy_corpus):
    expander = VocabularyExpander(base_tokenizer)
    sub_tok_path = expander.train_sub_tokenizer(str(dummy_corpus), vocab_size=300, algorithm="bpe")
    original_size = len(base_tokenizer)

    extended = expander.merge_vocabularies(str(sub_tok_path))

    assert len(extended) >= original_size
    assert len(base_tokenizer) == original_size  # base tokenizer must be untouched
    assert extended.original_vocab_size == original_size


def test_merge_vocabularies_no_duplicate_tokens(base_tokenizer, dummy_corpus):
    expander = VocabularyExpander(base_tokenizer)
    sub_tok_path = expander.train_sub_tokenizer(str(dummy_corpus), vocab_size=300, algorithm="bpe")

    extended = expander.merge_vocabularies(str(sub_tok_path))

    vocab = extended.get_vocab()
    assert len(vocab) == len(set(vocab.keys()))
    assert len(vocab) == len(set(vocab.values()))


# --- Regression coverage for the byte-level vocab-remapping bug ---
#
# gpt2 (the base_tokenizer fixture above) is itself a *byte-level* BPE
# tokenizer, but the dummy corpus above is pure ASCII ("quirklex", "florbin",
# ...). GPT-2's byte-to-unicode remapping is the identity function for
# printable ASCII, so a bug where the sub-tokenizer's byte-remapped vocab
# strings get handed to add_tokens() verbatim (instead of decoded back to
# literal text) is invisible on ASCII-only corpora: none of the tests above
# would fail even if merge_vocabularies() were silently adding tokens that
# can never match real (non-ASCII) input text. The tests below use non-ASCII
# text specifically to exercise that path.


@pytest.fixture(scope="module")
def non_ascii_corpus(tmp_path_factory):
    corpus_dir = tmp_path_factory.mktemp("corpus_non_ascii")
    corpus_file = corpus_dir / "corpus.txt"
    # Real Hindi sentences, repeated to give the trainer enough signal to
    # learn multi-character merges (UTF-8 multi-byte sequences under a
    # ByteLevel pre-tokenizer, which is exactly the case that broke).
    lines = [
        "देहरादून भारत के उत्तराखण्ड राज्य का एक प्रमुख नगर है।",
        "देहरादून अपनी प्राकृतिक सुंदरता के लिए प्रसिद्ध है।",
        "यह नगर पर्यटन और शिक्षा के लिए भी जाना जाता है।",
    ] * 30
    corpus_file.write_text("\n".join(lines), encoding="utf-8")
    return corpus_file


def test_merge_vocabularies_added_tokens_are_literal_substrings(base_tokenizer, non_ascii_corpus):
    """Added tokens must be usable literal text, not byte-remapped vocab keys."""
    expander = VocabularyExpander(base_tokenizer)
    sub_tok_path = expander.train_sub_tokenizer(str(non_ascii_corpus), vocab_size=350, algorithm="bpe")
    extended = expander.merge_vocabularies(str(sub_tok_path))

    original_vocab = set(base_tokenizer.get_vocab().keys())
    added_tokens = [t for t in extended.get_vocab() if t not in original_vocab]
    assert added_tokens, "expected at least one genuinely new token to be added"

    corpus_text = non_ascii_corpus.read_text(encoding="utf-8")
    for token in added_tokens:
        assert token in corpus_text, (
            f"added token {token!r} is not a literal substring of the training "
            "corpus -- it looks like a byte-remapped vocab key leaked through "
            "instead of being decoded back to real text"
        )


def test_merge_vocabularies_actually_reduces_token_count(base_tokenizer, non_ascii_corpus):
    """The whole point of expansion: fertility must measurably improve on non-ASCII text."""
    expander = VocabularyExpander(base_tokenizer)
    sub_tok_path = expander.train_sub_tokenizer(str(non_ascii_corpus), vocab_size=350, algorithm="bpe")
    extended = expander.merge_vocabularies(str(sub_tok_path))

    sample = "देहरादून भारत के उत्तराखण्ड राज्य का एक प्रमुख नगर है।"
    base_len = len(base_tokenizer.encode(sample, add_special_tokens=False))
    extended_len = len(extended.encode(sample, add_special_tokens=False))
    assert extended_len < base_len


def test_merge_vocabularies_round_trip_decodes_correctly(base_tokenizer, non_ascii_corpus):
    expander = VocabularyExpander(base_tokenizer)
    sub_tok_path = expander.train_sub_tokenizer(str(non_ascii_corpus), vocab_size=350, algorithm="bpe")
    extended = expander.merge_vocabularies(str(sub_tok_path))

    sample = "देहरादून अपनी प्राकृतिक सुंदरता के लिए प्रसिद्ध है।"
    ids = extended.encode(sample, add_special_tokens=False)
    assert extended.decode(ids).strip() == sample.strip()


def test_expanded_tokenizer_serializes_and_reloads_without_dropping_merges(
    base_tokenizer, non_ascii_corpus, tmp_path
):
    """save_pretrained() must write both tokenizer.json and tokenizer_config.json,
    and reloading from disk must reproduce identical tokenization -- i.e. no
    merge rules or added tokens were silently dropped in the round trip.
    """
    expander = VocabularyExpander(base_tokenizer)
    sub_tok_path = expander.train_sub_tokenizer(str(non_ascii_corpus), vocab_size=350, algorithm="bpe")
    extended = expander.merge_vocabularies(str(sub_tok_path))

    out_dir = tmp_path / "expanded_tokenizer"
    extended.save_pretrained(out_dir)

    assert (out_dir / "tokenizer.json").exists()
    assert (out_dir / "tokenizer_config.json").exists()

    reloaded = AutoTokenizer.from_pretrained(out_dir)
    assert isinstance(reloaded, PreTrainedTokenizerFast)
    assert len(reloaded) == len(extended)
    assert reloaded.get_vocab() == extended.get_vocab()

    sample = "देहरादून भारत के उत्तराखण्ड राज्य का एक प्रमुख नगर है।"
    assert reloaded.encode(sample, add_special_tokens=False) == extended.encode(
        sample, add_special_tokens=False
    )
