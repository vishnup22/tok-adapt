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
cross-lingual fine-tuning and domain adaptation — plus an end-to-end
pipeline (data prep → vocabulary adaptation → CPT/SFT/DPO → evaluation →
quantized export) built on top of it.

## Pipeline

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ 1. Data Prep    │ ──► │ 2. Vocabulary   │ ──► │ 3. Checkpoint   │
│ & Deduplication │     │ Adaptation      │     │ Alignment       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                         │
┌─────────────────┐     ┌─────────────────┐              ▼
│ 6. Inference    │ ◄── │ 5. Evaluation   │ ◄── ┌─────────────────┐
│ & Deployment    │     │ & Benchmarking  │     │ 4. Continued    │
└─────────────────┘     └─────────────────┘     │ Training & SFT  │
                                                 └─────────────────┘
```

Every phase is implemented and has a passing test; run the whole thing
end to end with one command against a YAML config (see
[Running the pipeline](#running-the-pipeline)) or drive each phase
individually via the CLI/Python API below.

| Phase | Module | Status |
|---|---|---|
| 1. Data prep & dedup | `tok_adapt.dedup` | MinHash/LSH near-dup removal + language-ID filtering + Corpus A/B split |
| 2-3. Vocabulary adaptation + checkpoint alignment | `tok_adapt.expansion`, `tok_adapt.initialization` | Sub-tokenizer training, safe vocab merge, `subword_mean` embedding init, weight-tying preserved |
| 4. Continued training & SFT | `tok_adapt.training` (`cpt.py`, `sft.py`, `dpo.py`) | HF `Trainer` CPT with unfrozen embeddings + LoRA/QLoRA backbone adapters; TRL `SFTTrainer`/`DPOTrainer` |
| 5. Evaluation & benchmarking | `tok_adapt.evaluation` | Perplexity (base vs. adapted), BLEU/chrF++, MMLU-style multiple-choice accuracy |
| 6. Export & deployment | `tok_adapt.export` | GGUF (llama.cpp), ONNX (optimum), vLLM serving |

**Honest scope note:** every phase above is real, tested code, and the
whole chain has been run end to end on an actual GPU (RTX 3060, 6GB VRAM)
— see [Verified end-to-end run](#verified-end-to-end-run) for the exact
numbers. What that run does *not* prove is production-scale training
quality: it's a few optimizer steps on a few KB of text to validate the
wiring (dataset construction, freeze/unfreeze rules, config plumbing,
checkpoint handoff between phases), not a converged model. Training to
convergence on gigabyte-scale corpora is a compute/time budget decision
for whoever runs it — the code doesn't change, only `--num-train-epochs`
and how long you're willing to wait. vLLM serving (part of Phase 6) has
no native Windows build; it's implemented and documented but only
testable under WSL2/Linux.

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
- **Dedup** — MinHash/LSH near-duplicate removal, language-ID filtering, and
  Corpus A (tokenizer sample) / Corpus B (full CPT corpus) splitting.
- **Training** — continued pre-training with unfrozen embeddings + LoRA/QLoRA
  backbone adapters, TRL-based supervised fine-tuning, and DPO alignment.
- **Evaluation** — held-out perplexity (base vs. adapted), BLEU/chrF++, and
  an MMLU-style multiple-choice accuracy harness.
- **Export** — GGUF (llama.cpp, with a pre-tokenizer override for expanded
  vocabularies — see [Exporting expanded checkpoints to GGUF](#exporting-expanded-checkpoints-to-gguf)),
  ONNX, and vLLM serving (Linux/WSL only).
- **CLI** — every phase as its own command, plus `tok-adapt pipeline` to run
  the whole chain from one YAML config. `rich` progress output throughout.

## Installation

```bash
pip install -e .
# Test dependencies:
pip install -e ".[dev]"
# Everything needed to run the full pipeline (dedup + train + eval + export):
pip install -e ".[pipeline]"
# Individually: dedup, train, eval, export, serve (vLLM, Linux/WSL only)
pip install -e ".[train]"
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
# --- Phases 2-3: vocabulary adaptation ---
# Expand a base model's tokenizer on a target corpus and align embeddings.
tok-adapt expand --model gpt2 --corpus ./corpus.txt --add-vocab-size 8000 --output ./expanded-model

# Align embeddings for a model against an already-expanded tokenizer.
tok-adapt align-embeddings --model gpt2 --tokenizer ./expanded-model --strategy subword_mean --output ./aligned-model

# Prune vocabulary unused in a reference corpus.
tok-adapt prune --model ./expanded-model --corpus ./corpus.txt --output ./pruned-model

# Benchmark fertility / compression between two tokenizers.
tok-adapt benchmark --base-tokenizer gpt2 --adapted-tokenizer ./expanded-model --text-file ./eval.txt

# --- Phase 1: data prep ---
tok-adapt dedup --input ./raw_corpus --output ./clean --languages hi,te

# --- Phase 4: training ---
tok-adapt train-cpt --model ./expanded-model --corpus ./clean/corpus_b.txt --output ./cpt-model
tok-adapt train-sft --model ./cpt-model --data ./sft_data.jsonl --output ./sft-model
tok-adapt train-dpo --model ./sft-model --data ./dpo_data.jsonl --output ./dpo-model

# --- Phase 5: evaluation ---
tok-adapt evaluate-perplexity --model ./dpo-model --base-model gpt2 --text-file ./eval.txt
tok-adapt evaluate-downstream --model ./dpo-model --translations ./translations.jsonl --mcq ./mcq.jsonl

# --- Phase 6: export ---
tok-adapt export-gguf --model ./dpo-model --output ./model.gguf --pre-tokenizer-hint gpt2
tok-adapt export-onnx --model ./dpo-model --output ./onnx-model
tok-adapt serve-vllm --model ./dpo-model --port 8000   # Linux/WSL only

# --- Run every enabled phase from one config ---
tok-adapt pipeline --config ./pipeline.yaml
```

### Running the pipeline

`tok-adapt pipeline --config pipeline.yaml` chains whichever phases you
enable, threading each stage's output into the next (disable a prefix of
stages and point `model:` at an existing checkpoint to resume mid-pipeline):

```yaml
model: gpt2
output_root: ./pipeline_out

dedup:
  enabled: true
  input_paths: ["./raw_corpus"]
  languages: ["hi", "te"]
  corpus_a_max_bytes: 30000000

expand:
  enabled: true
  add_vocab_size: 8000
  algorithm: bpe          # uses dedup's corpus_a.txt unless corpus_path is set

cpt:
  enabled: true
  use_lora: true
  max_steps: -1           # -1 runs num_train_epochs in full
  num_train_epochs: 1     # uses dedup's corpus_b.txt unless corpus_path is set

sft:
  enabled: true
  data_path: ./sft_data.jsonl

dpo:
  enabled: true
  data_path: ./dpo_data.jsonl

evaluate:
  enabled: true
  perplexity_text_file: ./eval.txt

export:
  enabled: true
  gguf: true
  gguf_pre_tokenizer_hint: gpt2   # required whenever expand ran -- see below
  onnx: true
```

Every stage is optional (`enabled: false` skips it) and writes to its own
numbered subdirectory of `output_root`; `summary.json` records what ran,
where, and the final model path.

### Exporting expanded checkpoints to GGUF

llama.cpp identifies a tokenizer's pre-tokenizer by hashing the encoded
output of a fixed probe string and matching it against a table of known
models. That hash depends on the *entire learned vocabulary*, not just the
pre-tokenizer regex — so a `tok-adapt expand`-ed tokenizer's hash generally
won't match anything in the table, even though its underlying pre-tokenizer
is unchanged from its base model. Pass `--pre-tokenizer-hint` (CLI) /
`gguf_pre_tokenizer_hint` (pipeline config) with the base tokenizer's id
(`gpt2`, `llama-bpe`, `qwen2`, ...) to declare it directly and skip the
broken auto-detection. Unmodified stock checkpoints don't need this.

## Repository layout

```
tok_adapt/
├── pyproject.toml
├── README.md
├── LICENSE
├── .github/workflows/tests.yml   # CI: pytest on 3.10/3.11/3.12
├── tok_adapt/
│   ├── __init__.py
│   ├── cli.py               # Click CLI entrypoint (all phases + pipeline)
│   ├── pipeline.py          # Pydantic-validated YAML config + orchestrator
│   ├── expansion.py         # Sub-tokenizer training & BPE merging
│   ├── initialization.py    # Subword embedding overlap initialization
│   ├── pruning.py           # Vocabulary trimming & model matrix shrinking
│   ├── metrics.py           # Fertility ratio & sequence compression metrics
│   ├── dedup.py             # Phase 1: MinHash/LSH dedup + language-ID filter
│   ├── training/            # Phase 4: CPT / SFT / DPO
│   │   ├── cpt.py
│   │   ├── sft.py
│   │   └── dpo.py
│   ├── evaluation/          # Phase 5: perplexity + downstream accuracy
│   │   ├── perplexity.py
│   │   └── downstream.py
│   ├── export/              # Phase 6: GGUF / ONNX / vLLM
│   │   ├── gguf_export.py
│   │   ├── onnx_export.py
│   │   └── vllm_serve.py
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
    ├── test_pruning.py
    ├── test_dedup.py
    ├── test_training.py
    ├── test_evaluation.py
    └── test_export.py
```

## Testing

```bash
pytest tests/ -v --color=yes
# Full pipeline test coverage needs the optional dependency groups installed:
pip install -e ".[pipeline]"
```

35 tests, all passing as of the last run in this environment. Tests use tiny
public checkpoints (`gpt2` tokenizer, `hf-internal-testing/tiny-random-gpt2`
model) and are skipped automatically if network access to the Hugging Face
Hub — or, for GGUF export, `git` — is unavailable. Coverage includes the
checklist items relevant to production use:

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
- **Freeze/unfreeze correctness**: `test_cpt_lora_leaves_embeddings_trainable`
  confirms LoRA-wrapped CPT keeps embeddings trainable while freezing the
  rest of the backbone; `test_cpt_full_finetune_trains_everything` confirms
  the non-LoRA path trains every parameter.
- **GGUF pre-tokenizer override**: `test_export_to_gguf_with_pre_tokenizer_hint_on_expanded_tokenizer`
  runs a real `VocabularyExpander` → GGUF conversion and asserts the
  `pre_tokenizer_hint` override produces a valid file.

## Verified end-to-end run

The full pipeline — dedup → expand → CPT (LoRA) → SFT → DPO → perplexity
eval → GGUF + ONNX export — was run start to finish on an RTX 3060 (6GB
VRAM) against `gpt2` and the real Hindi Wikipedia text in
`benchmarks/corpus/`. Actual output from that run:

| Stage | Result |
|---|---|
| Dedup | 32 lines in → 32 kept (0 exact/near-duplicates, 0 language-filtered at this scale) |
| Expand | vocab 50,257 → 50,667 (+410 tokens from an 8-line Corpus A sample) |
| CPT | 38.9M / 124.8M params trainable (LoRA adapters + unfrozen embeddings only) |
| SFT | trained on 6 prompt/response pairs |
| DPO | trained on 4 preference triples; real reward/logprob metrics reported (`rewards/accuracies`, `logps/chosen`, etc.) |
| Evaluate | perplexity 297.8 on held-out Hindi text (1,602 tokens) |
| Export | valid `.gguf` (253MB, f16) and `.onnx` (499MB) files produced |

This is a wiring proof at `max_steps: 2-3` on a few KB of text, not a
converged model — the perplexity number reflects a handful of gradient
steps, not real language-adaptation quality. It confirms every stage's
data contracts, checkpoint handoffs, and freeze rules work together
correctly on real GPU hardware; scaling it to gigabyte corpora and full
epochs is a compute-budget decision, not a code change.

## Notes on pruning

`VocabularyPruner` rebuilds a dense tokenizer vocabulary by editing the
underlying `tokenizers` JSON directly (remapping `model.vocab`, filtering
`model.merges` to surviving tokens, and remapping `added_tokens` /
`post_processor` ids). This is a best-effort transform: merge rules that
depended on a pruned token are dropped, which is expected since that token
is being removed from the vocabulary anyway.
