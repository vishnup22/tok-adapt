"""Tests for tok_adapt.evaluation (perplexity, translation, multiple-choice)."""

from __future__ import annotations

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from tok_adapt.evaluation.downstream import evaluate_multiple_choice, score_translations
from tok_adapt.evaluation.perplexity import compare_perplexity, compute_perplexity

MODEL_ID = "hf-internal-testing/tiny-random-gpt2"


@pytest.fixture(scope="module")
def tiny_model_and_tokenizer():
    try:
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    except Exception as exc:  # pragma: no cover - network-dependent
        pytest.skip(f"Could not download '{MODEL_ID}': {exc}")
    return model, tok


def test_compute_perplexity_returns_finite_positive_value(tiny_model_and_tokenizer):
    model, tok = tiny_model_and_tokenizer
    lines = ["the quick brown fox jumps over the lazy dog", "a second unrelated held-out sentence"]
    result = compute_perplexity(model, tok, lines, block_size=16)
    assert result.num_tokens > 0
    assert result.perplexity > 0
    assert result.loss == pytest.approx(result.loss)  # finite, no NaN


def test_compute_perplexity_rejects_empty_input(tiny_model_and_tokenizer):
    model, tok = tiny_model_and_tokenizer
    with pytest.raises(ValueError):
        compute_perplexity(model, tok, [""], block_size=16)


def test_compare_perplexity_same_model_gives_zero_delta(tiny_model_and_tokenizer):
    model, tok = tiny_model_and_tokenizer
    lines = ["the quick brown fox jumps over the lazy dog"]
    comparison = compare_perplexity(model, tok, model, tok, lines, block_size=16)
    assert comparison["perplexity_delta"] == pytest.approx(0.0, abs=1e-4)
    assert comparison["improved"] is False  # equal, not strictly better


def test_score_translations_perfect_match_scores_100():
    hyp = ["the cat sat on the mat", "hello world"]
    ref = ["the cat sat on the mat", "hello world"]
    score = score_translations(hyp, ref)
    assert score.bleu == pytest.approx(100.0, abs=0.01)
    assert score.chrf == pytest.approx(100.0, abs=0.01)


def test_score_translations_rejects_length_mismatch():
    with pytest.raises(ValueError):
        score_translations(["a"], ["a", "b"])


def test_evaluate_multiple_choice_returns_valid_accuracy(tiny_model_and_tokenizer):
    model, tok = tiny_model_and_tokenizer
    questions = [
        {"question": "What color is the sky?", "choices": ["blue", "purple", "green"], "answer_index": 0},
        {"question": "What is 2+2?", "choices": ["4", "5", "6"], "answer_index": 0},
    ]
    result = evaluate_multiple_choice(model, tok, questions)
    assert result.num_total == 2
    assert 0 <= result.num_correct <= 2
    assert result.accuracy == pytest.approx(result.num_correct / result.num_total)
