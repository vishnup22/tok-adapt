"""End-to-end demo: expand a tokenizer, align embeddings, and compare fertility.

Run with:

    python examples/quickstart.py

Requires network access on first run to download the tiny public models used
below (``gpt2`` tokenizer + ``hf-internal-testing/tiny-random-gpt2`` model).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

from tok_adapt.expansion import VocabularyExpander
from tok_adapt.initialization import EmbeddingInitializer
from tok_adapt.metrics import FertilityEvaluator

BASE_MODEL_ID = "hf-internal-testing/tiny-random-gpt2"
BASE_TOKENIZER_ID = "gpt2"


def main() -> None:
    # 1. Load a base tokenizer (and a matching tiny model for the demo).
    print(f"Loading base tokenizer '{BASE_TOKENIZER_ID}' and model '{BASE_MODEL_ID}'...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_TOKENIZER_ID)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID)

    dummy_text = (
        "quirklex florbin trantak zestimo appears often in this domain corpus.\n"
        "florbin quirklex trantak zestimo florbin quirklex trantak zestimo.\n"
    ) * 50
    corpus_path = Path(tempfile.mkdtemp()) / "domain_corpus.txt"
    corpus_path.write_text(dummy_text, encoding="utf-8")
    print(f"Wrote dummy domain corpus to {corpus_path}")

    # 2. Expand the vocabulary on the dummy domain text.
    print("\nTraining sub-tokenizer and merging new vocabulary...")
    expander = VocabularyExpander(tokenizer)
    sub_tok_path = expander.train_sub_tokenizer(str(corpus_path), vocab_size=300, algorithm="bpe")
    extended_tokenizer = expander.merge_vocabularies(str(sub_tok_path))
    print(f"Base vocab size: {len(tokenizer)} -> Extended vocab size: {len(extended_tokenizer)}")

    # 3. Align model embeddings with subword_mean initialization.
    print("\nAligning new embeddings with subword_mean initialization...")
    initializer = EmbeddingInitializer(model, tokenizer)
    model = initializer.smart_resize_embeddings(extended_tokenizer, strategy="subword_mean")
    print(f"Model embedding matrix shape: {tuple(model.get_input_embeddings().weight.shape)}")

    # 4. Print fertility score comparison.
    print("\nComparing fertility of base vs. extended tokenizer on the domain corpus...")
    sample_texts = [line for line in dummy_text.strip().split("\n") if line]
    base_eval = FertilityEvaluator(tokenizer)

    base_stats = base_eval.compute_fertility(sample_texts)
    extended_stats = FertilityEvaluator(extended_tokenizer).compute_fertility(sample_texts)
    compression = base_eval.compare_sequence_compression(extended_tokenizer, sample_texts)

    print(f"Base tokenizer fertility:     {base_stats}")
    print(f"Extended tokenizer fertility: {extended_stats}")
    print(f"Compression vs base:          {compression}")


if __name__ == "__main__":
    main()
