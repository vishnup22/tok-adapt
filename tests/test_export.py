"""Tests for tok_adapt.export (GGUF, ONNX, vLLM).

GGUF and ONNX tests are network-dependent (GGUF fetches llama.cpp's
conversion script on first use; ONNX export needs the model available
locally) and are skipped if that access isn't available. The vLLM test
asserts the documented native-Windows limitation rather than exercising
real inference, since vLLM has no Windows build.
"""

from __future__ import annotations

import importlib.util
import shutil

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from tok_adapt.export.gguf_export import export_to_gguf
from tok_adapt.export.onnx_export import export_to_onnx
from tok_adapt.export.vllm_serve import VLLMUnavailableError, generate_with_vllm
from tok_adapt.expansion import VocabularyExpander
from tok_adapt.initialization import EmbeddingInitializer

MODEL_ID = "hf-internal-testing/tiny-random-gpt2"


@pytest.fixture(scope="module")
def tiny_checkpoint_dir(tmp_path_factory):
    # Pair a tiny random model with the *real* gpt2 tokenizer rather than
    # tiny-random-gpt2's own tokenizer: llama.cpp's GGUF converter
    # fingerprints the tokenizer's pre-tokenizer regex against a table of
    # known models, and tiny-random-gpt2's synthetic tokenizer.json isn't
    # in that table. The real gpt2 tokenizer is what tok-adapt's own
    # expand/prune pipeline actually produces checkpoints with, so this
    # is also the more representative fixture.
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    except Exception as exc:  # pragma: no cover - network-dependent
        pytest.skip(f"Could not download model/tokenizer: {exc}")

    model.resize_token_embeddings(len(tokenizer))
    model.config.vocab_size = len(tokenizer)

    out_dir = tmp_path_factory.mktemp("tiny_ckpt")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return out_dir


def test_export_to_gguf_produces_valid_file(tiny_checkpoint_dir, tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git not available")

    output_path = tmp_path / "model.gguf"
    try:
        result = export_to_gguf(tiny_checkpoint_dir, output_path, outtype="f16")
    except RuntimeError as exc:  # pragma: no cover - network-dependent
        pytest.skip(f"Could not fetch/run GGUF conversion tooling: {exc}")

    assert result.exists()
    assert result.stat().st_size > 0
    with open(result, "rb") as f:
        magic = f.read(4)
    assert magic == b"GGUF"


def test_export_to_onnx_produces_graph(tiny_checkpoint_dir, tmp_path):
    output_dir = tmp_path / "onnx_out"
    try:
        result = export_to_onnx(tiny_checkpoint_dir, output_dir, task="text-generation-with-past")
    except RuntimeError as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"ONNX export unavailable in this environment: {exc}")

    onnx_files = list(result.glob("*.onnx"))
    assert onnx_files, f"No .onnx files produced in {result}"


def test_export_to_gguf_with_pre_tokenizer_hint_on_expanded_tokenizer(tmp_path):
    """Vocabulary-expanded tokenizers can fail llama.cpp's hash-based
    pre-tokenizer auto-detection (see the gguf_export module docstring),
    since it depends on the full learned vocabulary rather than just the
    pre-tokenizer regex -- whether a given expansion actually trips it
    depends on how much the new merges perturb llama.cpp's fixed probe
    string. This asserts the override always produces a valid GGUF file
    for an expanded tokenizer regardless of whether the hash happens to
    collide, since real (e.g. Hindi Wikipedia-scale) expansions have been
    observed to trip it even when small synthetic ones don't.
    """
    if shutil.which("git") is None:
        pytest.skip("git not available")

    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
        base_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    except Exception as exc:  # pragma: no cover - network-dependent
        pytest.skip(f"Could not download model/tokenizer: {exc}")

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("quirklex florbin trantak zestimo\n" * 50, encoding="utf-8")

    expander = VocabularyExpander(base_tokenizer)
    sub_tok_path = expander.train_sub_tokenizer(str(corpus), vocab_size=300, algorithm="bpe")
    expanded_tokenizer = expander.merge_vocabularies(str(sub_tok_path))

    initializer = EmbeddingInitializer(model, base_tokenizer)
    expanded_model = initializer.smart_resize_embeddings(expanded_tokenizer, strategy="subword_mean")

    ckpt_dir = tmp_path / "expanded_ckpt"
    expanded_model.save_pretrained(ckpt_dir)
    expanded_tokenizer.save_pretrained(ckpt_dir)

    try:
        result = export_to_gguf(ckpt_dir, tmp_path / "with_hint.gguf", outtype="f16", pre_tokenizer_hint="gpt2")
    except RuntimeError as exc:  # pragma: no cover - network-dependent
        pytest.skip(f"Could not fetch/run GGUF conversion tooling: {exc}")

    assert result.exists()
    with open(result, "rb") as f:
        assert f.read(4) == b"GGUF"


def test_vllm_raises_clear_error_when_unavailable():
    if importlib.util.find_spec("vllm") is not None:
        pytest.skip("vllm is actually installed in this environment; this test targets its absence.")
    with pytest.raises(VLLMUnavailableError):
        generate_with_vllm("gpt2", ["hello"])
