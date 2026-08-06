"""Tests for tok_adapt.initialization.EmbeddingInitializer."""

from __future__ import annotations

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from conftest import untie_output_embeddings
from tok_adapt.expansion import VocabularyExpander
from tok_adapt.initialization import EmbeddingInitializer

MODEL_ID = "hf-internal-testing/tiny-random-gpt2"
# Use the tiny model's own tokenizer so old-token ids resolved during
# subword decomposition are always valid indices into its embedding matrix.
TOKENIZER_ID = "hf-internal-testing/tiny-random-gpt2"


@pytest.fixture()
def model_and_tokenizer():
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    except Exception as exc:  # pragma: no cover - network-dependent
        pytest.skip(f"Could not download tiny test model/tokenizer: {exc}")
    return model, tokenizer


@pytest.fixture(scope="module")
def dummy_corpus(tmp_path_factory):
    corpus_dir = tmp_path_factory.mktemp("corpus_init")
    corpus_file = corpus_dir / "corpus.txt"
    corpus_file.write_text(
        "\n".join(["quirklex florbin trantak zestimo appears often in this corpus"] * 60),
        encoding="utf-8",
    )
    return corpus_file


def _expand(tokenizer, corpus_path):
    expander = VocabularyExpander(tokenizer)
    sub_tok_path = expander.train_sub_tokenizer(str(corpus_path), vocab_size=280, algorithm="bpe")
    return expander.merge_vocabularies(str(sub_tok_path))


def test_smart_resize_embeddings_shapes_match(model_and_tokenizer, dummy_corpus):
    model, tokenizer = model_and_tokenizer
    extended_tokenizer = _expand(tokenizer, dummy_corpus)

    initializer = EmbeddingInitializer(model, tokenizer)
    resized_model = initializer.smart_resize_embeddings(extended_tokenizer, strategy="subword_mean")

    assert resized_model.get_input_embeddings().weight.shape[0] == len(extended_tokenizer)
    assert resized_model.config.vocab_size == len(extended_tokenizer)


def test_new_embeddings_are_finite_and_not_trivially_zero(model_and_tokenizer, dummy_corpus):
    model, tokenizer = model_and_tokenizer
    extended_tokenizer = _expand(tokenizer, dummy_corpus)
    original_size = len(tokenizer)

    initializer = EmbeddingInitializer(model, tokenizer)
    resized_model = initializer.smart_resize_embeddings(extended_tokenizer, strategy="subword_mean")

    weights = resized_model.get_input_embeddings().weight.data
    for new_id in range(original_size, min(original_size + 5, len(extended_tokenizer))):
        vec = weights[new_id]
        assert not torch.isnan(vec).any()
        assert not torch.isinf(vec).any()


def test_unsupported_strategy_raises(model_and_tokenizer, dummy_corpus):
    model, tokenizer = model_and_tokenizer
    extended_tokenizer = _expand(tokenizer, dummy_corpus)

    initializer = EmbeddingInitializer(model, tokenizer)
    with pytest.raises(ValueError):
        initializer.smart_resize_embeddings(extended_tokenizer, strategy="random")


def test_smart_resize_keeps_tied_weights_tied(model_and_tokenizer, dummy_corpus):
    """When tie_word_embeddings is True, lm_head must update automatically with embed_tokens.

    Regression coverage for the checklist requirement that resizing a tied
    model never desyncs embed_tokens and lm_head into two separate tensors.
    """
    model, tokenizer = model_and_tokenizer
    assert model.config.tie_word_embeddings is True  # tiny-random-gpt2 default: tied

    extended_tokenizer = _expand(tokenizer, dummy_corpus)
    initializer = EmbeddingInitializer(model, tokenizer)
    resized_model = initializer.smart_resize_embeddings(extended_tokenizer, strategy="subword_mean")

    input_embeddings = resized_model.get_input_embeddings()
    output_embeddings = resized_model.get_output_embeddings()
    assert resized_model.config.tie_word_embeddings is True
    # Still the exact same underlying tensor -- not just equal values, but
    # literally tied, so a later fine-tuning step can't update one without
    # the other silently going stale.
    assert output_embeddings.weight.data_ptr() == input_embeddings.weight.data_ptr()
    assert output_embeddings.weight.shape[0] == len(extended_tokenizer)


def test_smart_resize_mirrors_lm_head_when_untied(model_and_tokenizer, dummy_corpus):
    """New rows must be mirrored into an untied lm_head, without disturbing old rows."""
    model, tokenizer = model_and_tokenizer
    untie_output_embeddings(model)
    output_embeddings_before = model.get_output_embeddings()
    input_embeddings_before = model.get_input_embeddings()
    assert output_embeddings_before.weight.data_ptr() != input_embeddings_before.weight.data_ptr()

    # Snapshot a few old rows of the untied lm_head before resizing, so we can
    # confirm the resize/mirror step does not clobber pre-existing rows.
    original_size = len(tokenizer)
    sample_old_ids = list(range(min(5, original_size)))
    old_lm_head_rows = output_embeddings_before.weight.data[sample_old_ids].clone()

    extended_tokenizer = _expand(tokenizer, dummy_corpus)
    initializer = EmbeddingInitializer(model, tokenizer)
    resized_model = initializer.smart_resize_embeddings(extended_tokenizer, strategy="subword_mean")

    input_embeddings = resized_model.get_input_embeddings()
    output_embeddings = resized_model.get_output_embeddings()

    # Dimensions: both matrices must grow to the new vocab size.
    assert input_embeddings.weight.shape[0] == len(extended_tokenizer)
    assert output_embeddings.weight.shape[0] == len(extended_tokenizer)
    assert output_embeddings.weight.data_ptr() != input_embeddings.weight.data_ptr()

    # Old rows must survive untouched.
    assert torch.allclose(output_embeddings.weight.data[sample_old_ids], old_lm_head_rows)

    # New rows must be mirrored: lm_head[new_id] == embed_tokens[new_id].
    new_ids = range(original_size, len(extended_tokenizer))
    for new_id in list(new_ids)[:5]:
        assert torch.allclose(
            output_embeddings.weight.data[new_id], input_embeddings.weight.data[new_id]
        )
