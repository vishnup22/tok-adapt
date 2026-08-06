# tok-adapt

[![tests](https://github.com/vishnup22/tok-adapt/actions/workflows/tests.yml/badge.svg)](https://github.com/vishnup22/tok-adapt/actions/workflows/tests.yml)
[![PyPI version](https://img.shields.io/pypi/v/tok-adapt.svg)](https://pypi.org/project/tok-adapt/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![🤗 Transformers](https://img.shields.io/badge/%F0%9F%A4%97-transformers-yellow)](https://github.com/huggingface/transformers)

> **Note on the badges above:** the tests badge and PyPI badge are live,
> dynamic shields — they'll render "passing"/a real version once this repo is
> pushed to GitHub (with `.github/workflows/tests.yml`, already included) and
> published to PyPI. Until then they'll correctly show "no status"/"not
> found" rather than a fabricated claim.

A production-ready Python/CLI library for adapting, extending, pruning, and
initializing Hugging Face tokenizers and LLM embedding layers for
cross-lingual fine-tuning and domain adaptation.

## Proof of efficiency

Real, reproducible benchmark against `Qwen/Qwen2.5-7B` on held-out Hindi +
Telugu Wikipedia text (not synthetic data, not the training corpus) —
full methodology and a bug this benchmark caught in
[`benchmarks/README.md`](benchmarks/README.md):

| | Base tokenizer | tok-adapt-expanded | Reduction |
|---|---:|---:|---:|
| **Qwen/Qwen2.5-7B** | 381,277 tokens | 82,774 tokens | **78.3%** |
| **gpt2** (sanity check) | 680,347 tokens | 83,576 tokens | **87.7%** |

Specializing Qwen2.5-7B's vocabulary to only the tokens a narrow domain
corpus actually uses would free **~2 GiB** of embedding-matrix memory
(99.2% reduction) — see the benchmark README for what's measured directly
vs. analytically projected from real model config dimensions.

Run it yourself: `python benchmarks/run_benchmark.py`.

## Features

- **Expansion** — train a BPE/Unigram sub-tokenizer on a target corpus and
  safely merge new tokens into a base `PreTrainedTokenizerFast` without
  corrupting its existing merge rules.
- **Initialization** — initialize newly added token embeddings from the mean
  of their decomposed subwords (`subword_mean`) instead of random noise.
- **Pruning** — trim unused vocabulary based on corpus frequency and shrink
  `embed_tokens` / `lm_head` matrices to match, reducing checkpoint size.
- **Metrics** — fertility (token/word, token/byte) and sequence compression
  benchmarks to quantify tokenizer efficiency.
- **CLI** — `tok-adapt expand|align-embeddings|prune|benchmark`, with `rich`
  progress output.

## Installation

```bash
pip install -e .
# or, with test dependencies:
pip install -e ".[dev]"
```

## Python API

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from tok_adapt import VocabularyExpander, EmbeddingInitializer, FertilityEvaluator

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# 1. Expand vocabulary on a target-domain/language corpus.
expander = VocabularyExpander(tokenizer)
sub_tok_path = expander.train_sub_tokenizer("corpus.txt", vocab_size=8000, algorithm="bpe")
extended_tokenizer = expander.merge_vocabularies(str(sub_tok_path))

# 2. Align embeddings for the newly added tokens.
initializer = EmbeddingInitializer(model, tokenizer)
model = initializer.smart_resize_embeddings(extended_tokenizer, strategy="subword_mean")

# 3. Measure the payoff.
base_eval = FertilityEvaluator(tokenizer)
print(base_eval.compare_sequence_compression(extended_tokenizer, ["some target-language text"]))
```

See [`examples/quickstart.py`](examples/quickstart.py) for a full runnable
demo, and [`tests/`](tests/) for pruning + shape-validation examples.

## CLI

```bash
# Expand a base model's tokenizer on a target corpus and align embeddings.
tok-adapt expand --model gpt2 --corpus ./corpus.txt --add-vocab-size 8000 --output ./expanded-model

# Align embeddings for a model against an already-expanded tokenizer.
tok-adapt align-embeddings --model gpt2 --tokenizer ./expanded-model --strategy subword_mean --output ./aligned-model

# Prune vocabulary unused in a reference corpus.
tok-adapt prune --model ./expanded-model --corpus ./corpus.txt --output ./pruned-model

# Benchmark fertility / compression between two tokenizers.
tok-adapt benchmark --base-tokenizer gpt2 --adapted-tokenizer ./expanded-model --text-file ./eval.txt
```

## Repository layout

```
tok_adapt/
├── pyproject.toml
├── README.md
├── LICENSE
├── .github/workflows/tests.yml   # CI: pytest on 3.10/3.11/3.12
├── tok_adapt/
│   ├── __init__.py
│   ├── cli.py               # Click CLI entrypoint
│   ├── expansion.py         # Sub-tokenizer training & BPE merging
│   ├── initialization.py    # Subword embedding overlap initialization
│   ├── pruning.py           # Vocabulary trimming & model matrix shrinking
│   ├── metrics.py           # Fertility ratio & sequence compression metrics
│   └── utils.py             # Model/tokenizer loading helpers
├── examples/
│   ├── quickstart.py
│   └── domain_corpus.txt    # small Hindi/Telugu sample used by the demo
├── demo/
│   └── expand.tape          # VHS script to record a terminal demo GIF
├── benchmarks/
│   ├── README.md            # methodology, what's measured vs. projected
│   ├── RESULTS.md           # latest real run output
│   ├── run_benchmark.py     # reproducible Qwen2.5-7B + gpt2 benchmark
│   └── corpus/              # real Hindi/Telugu Wikipedia text used above
└── tests/
    ├── test_expansion.py
    ├── test_initialization.py
    └── test_pruning.py
```

## Testing

```bash
pytest tests/ -v --color=yes
```

18 tests, all passing as of the last run in this environment. Tests use tiny
public checkpoints (`gpt2` tokenizer, `hf-internal-testing/tiny-random-gpt2`
model) and are skipped automatically if network access to the Hugging Face
Hub is unavailable. Coverage includes the checklist items relevant to
production use:

- **Tied weights**: `test_smart_resize_keeps_tied_weights_tied` confirms
  `lm_head` stays literally the same tensor as `embed_tokens` (same
  `data_ptr()`) after resizing when `tie_word_embeddings=True` — no desync.
- **Untied weights**: `test_smart_resize_mirrors_lm_head_when_untied` and
  `test_prune_slices_embed_tokens_and_lm_head_correctly` verify `lm_head` is
  sliced/mirrored independently and correctly when untied.
- **Special token preservation**: `test_prune_keeps_special_tokens` asserts
  every special token survives pruning regardless of corpus frequency.
- **Tokenizer serialization**: `test_expanded_tokenizer_serializes_and_reloads_without_dropping_merges`
  round-trips `save_pretrained()` → `from_pretrained()` and asserts identical
  tokenization, not just that the files exist.

## Notes on pruning

`VocabularyPruner` rebuilds a dense tokenizer vocabulary by editing the
underlying `tokenizers` JSON directly (remapping `model.vocab`, filtering
`model.merges` to surviving tokens, and remapping `added_tokens` /
`post_processor` ids). This is a best-effort transform: merge rules that
depended on a pruned token are dropped, which is expected since that token
is being removed from the vocabulary anyway.
