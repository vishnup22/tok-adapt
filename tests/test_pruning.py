"""Tests for tok_adapt.pruning.VocabularyPruner."""

from __future__ import annotations

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from conftest import untie_output_embeddings
from tok_adapt.pruning import VocabularyPruner

MODEL_ID = "hf-internal-testing/tiny-random-gpt2"
# Use the tiny model's own tokenizer (not the full "gpt2" tokenizer): pruning
# slices the model's actual embedding matrix by token id, so the tokenizer's
# vocab must match the model's embedding row count or ids go out of range.
TOKENIZER_ID = "hf-internal-testing/tiny-random-gpt2"


@pytest.fixture()
def model_and_tokenizer():
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    except Exception as exc:  # pragma: no cover - network-dependent
        pytest.skip(f"Could not download tiny test model/tokenizer: {exc}")
    return model, tokenizer


@pytest.fixture()
def small_corpus(tmp_path):
    corpus_file = tmp_path / "corpus.txt"
    corpus_file.write_text("the quick brown fox jumps over the lazy dog\n" * 20, encoding="utf-8")
    return corpus_file


def test_prune_unused_tokens_shrinks_vocab(model_and_tokenizer, small_corpus):
    model, tokenizer = model_and_tokenizer
    pruner = VocabularyPruner(model, tokenizer)

    pruned_model, pruned_tokenizer = pruner.prune_unused_tokens(
        str(small_corpus), min_frequency=1, keep_special_tokens=True
    )

    assert len(pruned_tokenizer) < len(tokenizer)
    assert pruned_model.get_input_embeddings().weight.shape[0] == len(pruned_tokenizer)
    assert pruned_model.config.vocab_size == len(pruned_tokenizer)


def test_prune_keeps_special_tokens(model_and_tokenizer, small_corpus):
    model, tokenizer = model_and_tokenizer
    pruner = VocabularyPruner(model, tokenizer)

    _, pruned_tokenizer = pruner.prune_unused_tokens(
        str(small_corpus), min_frequency=1, keep_special_tokens=True
    )

    pruned_vocab = pruned_tokenizer.get_vocab()
    for special_id in tokenizer.all_special_ids:
        special_str = tokenizer.convert_ids_to_tokens(special_id)
        assert special_str in pruned_vocab


def test_prune_output_embeddings_shape_matches_tokenizer(model_and_tokenizer, small_corpus):
    model, tokenizer = model_and_tokenizer
    pruner = VocabularyPruner(model, tokenizer)

    pruned_model, pruned_tokenizer = pruner.prune_unused_tokens(str(small_corpus), min_frequency=1)

    output_embeddings = pruned_model.get_output_embeddings()
    if output_embeddings is not None:
        assert output_embeddings.weight.shape[0] == len(pruned_tokenizer)


def test_prune_raises_when_corpus_missing(model_and_tokenizer, tmp_path):
    model, tokenizer = model_and_tokenizer
    pruner = VocabularyPruner(model, tokenizer)

    with pytest.raises(FileNotFoundError):
        pruner.prune_unused_tokens(str(tmp_path / "does_not_exist.txt"))


def test_prune_slices_embed_tokens_and_lm_head_correctly(model_and_tokenizer, small_corpus):
    """Every retained row must land at the right new index in BOTH matrices.

    Verifies actual tensor values (not just shape) survive the slice/reorder
    for both embed_tokens and an independently-weighted (untied) lm_head.
    """
    model, tokenizer = model_and_tokenizer
    untie_output_embeddings(model)

    original_input_weight = model.get_input_embeddings().weight.data.clone()
    original_output_weight = model.get_output_embeddings().weight.data.clone()
    # Sanity check that the fixture actually produced independent matrices.
    assert not torch.allclose(original_input_weight, original_output_weight)

    pruner = VocabularyPruner(model, tokenizer)
    pruned_model, pruned_tokenizer = pruner.prune_unused_tokens(
        str(small_corpus), min_frequency=1, keep_special_tokens=True
    )

    old_id_to_new_id = pruner.old_id_to_new_id
    pruned_input_weight = pruned_model.get_input_embeddings().weight.data
    pruned_output_weight = pruned_model.get_output_embeddings().weight.data

    assert pruned_input_weight.shape[0] == len(pruned_tokenizer)
    assert pruned_output_weight.shape[0] == len(pruned_tokenizer)
    assert pruned_output_weight.data_ptr() != pruned_input_weight.data_ptr()

    for old_id, new_id in old_id_to_new_id.items():
        assert torch.allclose(pruned_input_weight[new_id], original_input_weight[old_id]), (
            f"embed_tokens row mismatch: old_id={old_id} new_id={new_id}"
        )
        assert torch.allclose(pruned_output_weight[new_id], original_output_weight[old_id]), (
            f"lm_head row mismatch: old_id={old_id} new_id={new_id}"
        )
