"""Smoke tests for tok_adapt.training (CPT, SFT, DPO).

These run one or two real optimizer steps against a tiny public checkpoint
to prove the wiring (dataset construction, PEFT freezing rules, TRL config
plumbing) is correct end to end. They are not a substitute for a real
training run on production-scale data/compute.
"""

from __future__ import annotations

import json

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from tok_adapt.training.cpt import ContinuedPretrainer
from tok_adapt.training.dpo import PreferenceAligner
from tok_adapt.training.sft import SupervisedFineTuner

MODEL_ID = "hf-internal-testing/tiny-random-gpt2"


@pytest.fixture()
def tiny_tokenizer():
    try:
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
    except Exception as exc:  # pragma: no cover - network-dependent
        pytest.skip(f"Could not download tokenizer '{MODEL_ID}': {exc}")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture()
def tiny_model():
    try:
        return AutoModelForCausalLM.from_pretrained(MODEL_ID)
    except Exception as exc:  # pragma: no cover - network-dependent
        pytest.skip(f"Could not download model '{MODEL_ID}': {exc}")


@pytest.fixture()
def cpt_corpus(tmp_path):
    lines = ["quirklex florbin trantak zestimo appears in this tiny domain corpus."] * 40
    path = tmp_path / "corpus_b.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


@pytest.fixture()
def sft_data(tmp_path):
    records = [{"prompt": "Translate: hello -> ", "response": "namaste"} for _ in range(8)]
    path = tmp_path / "sft.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return path


@pytest.fixture()
def dpo_data(tmp_path):
    records = [
        {"prompt": "How do I say hello?", "chosen": "You say namaste.", "rejected": "I don't know."}
        for _ in range(8)
    ]
    path = tmp_path / "dpo.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return path


def test_cpt_lora_leaves_embeddings_trainable(tiny_model, tiny_tokenizer, cpt_corpus, tmp_path):
    pretrainer = ContinuedPretrainer(tiny_model, tiny_tokenizer, use_lora=True, block_size=32)
    result = pretrainer.train(
        cpt_corpus, tmp_path / "out", max_steps=1, per_device_train_batch_size=2, fp16=False
    )
    assert result.output_dir.exists()
    assert (result.output_dir / "tokenizer_config.json").exists()
    assert 0 < result.trainable_params < result.total_params


def test_cpt_full_finetune_trains_everything(tiny_model, tiny_tokenizer, cpt_corpus, tmp_path):
    pretrainer = ContinuedPretrainer(tiny_model, tiny_tokenizer, use_lora=False, block_size=32)
    result = pretrainer.train(
        cpt_corpus, tmp_path / "out_full", max_steps=1, per_device_train_batch_size=2, fp16=False
    )
    assert result.trainable_params == result.total_params


def test_sft_trains_on_prompt_response_pairs(tiny_model, tiny_tokenizer, sft_data, tmp_path):
    tuner = SupervisedFineTuner(tiny_model, tiny_tokenizer, use_lora=True)
    result = tuner.train(sft_data, tmp_path / "sft_out", max_steps=1, per_device_train_batch_size=2, max_length=64)
    assert result.output_dir.exists()
    assert isinstance(result.train_loss, float)


def test_dpo_trains_on_preference_triples(tiny_model, tiny_tokenizer, dpo_data, tmp_path):
    aligner = PreferenceAligner(tiny_model, tiny_tokenizer, use_lora=True)
    result = aligner.train(dpo_data, tmp_path / "dpo_out", max_steps=1, per_device_train_batch_size=1, max_length=64)
    assert result.output_dir.exists()
    assert isinstance(result.train_loss, float)
