# tok-adapt real-world benchmark: Hindi + Telugu vs. Qwen2.5-7B

Everything in this directory is a **real, reproducible measurement**, not
illustrative numbers. Run it yourself:

```bash
pip install -e ".[dev]"
python benchmarks/run_benchmark.py
```

## Headline result

| | Base tokenizer | tok-adapt-expanded | Reduction |
|---|---:|---:|---:|
| **Qwen/Qwen2.5-7B** | 381,277 tokens | 82,774 tokens | **78.3%** |
| **gpt2** (English-centric, sanity check) | 680,347 tokens | 83,576 tokens | **87.7%** |

Measured on 173 held-out Hindi + Telugu Wikipedia paragraphs (~3,647
sentences) that were **not** part of the sub-tokenizer's training corpus.
Full run log: [`RESULTS.md`](RESULTS.md).

Pruning projection (analytical, see caveat below): specializing
Qwen2.5-7B's 152,064-token vocabulary down to only the tokens this Hindi+Telugu
domain corpus actually uses (1,237 tokens, 0.8% of the full vocab) would free
**2.01 GiB** of embedding-matrix memory — a 99.2% reduction in `embed_tokens` +
`lm_head` footprint.

## What was measured directly vs. projected

- **Sequence-length reduction** (78.3% / 87.7% above) is a **direct
  measurement**: both tokenizers actually ran `.encode()` over the identical
  held-out text, `tok_adapt.metrics.FertilityEvaluator` counted the resulting
  tokens.
- **Embedding memory savings** is an **analytical projection**, computed from
  the model's real `config.json` (`vocab_size=152064`, `hidden_size=3584`,
  `dtype=bfloat16`, `tie_word_embeddings=False` — all real Qwen2.5-7B values),
  *not* measured on a loaded checkpoint. Qwen2.5-7B's weights are ~15GB;
  downloading and loading them was out of scope for this environment. The
  `tok_adapt.pruning.VocabularyPruner` machinery that would perform this
  slicing on real weights is covered separately by `tests/test_pruning.py`,
  which does load a real (tiny) model and asserts the sliced tensors are
  numerically correct, not just correctly shaped.

## Model and data choices, and why

- **Qwen2.5-7B**, not Llama-3-8B: both were considered as "standard open
  models" per the original ask. Llama-3-8B's tokenizer *is* accessible in this
  environment, but Qwen2.5-7B is fully ungated on the Hub, so it's the
  lower-friction default for anyone re-running this script without a
  Hugging Face access token.
- **gpt2** is included as a second, independent check specifically because
  Qwen2.5-7B already ships broad multilingual/Indic subword coverage (its own
  152k vocab isn't naive). Confirming the same expand→measure procedure also
  produces a large, honest reduction against a genuinely English-centric
  tokenizer (0.98 tokens/byte at baseline — near one token per raw byte,
  i.e. close to byte-fallback) rules out "this only works because Qwen's
  tokenizer is already unusually good" as an alternative explanation.
- **Corpus**: real Hindi Wikipedia articles
  (`vibrantlabsai/hindi-wikipedia`, Apache-2.0) and real Telugu Wikipedia rows
  (`vengi-ai/telugu-wikipedia-clean`) pulled live from the Hugging Face Hub —
  not synthetic or hand-written text. Training and held-out eval text are
  disjoint articles (Telugu: disjoint row offsets; Hindi: disjoint files) — the
  reduction numbers above are out-of-sample, not measured on the training data
  itself.

## A bug this benchmark caught (and fixed)

The first version of this benchmark measured only a **0.3–0.6% token
reduction** — a strange result for a 5,000+ new-token vocabulary merge. That
led to finding a real correctness bug in `tok_adapt/expansion.py`:
`train_sub_tokenizer` used a `ByteLevel` pre-tokenizer, which stores vocabulary
entries in a remapped byte-alphabet (a literal space becomes `Ġ` / U+0120, and
non-ASCII UTF-8 bytes get remapped too). `merge_vocabularies` then handed those
byte-remapped strings straight to `add_tokens()`, which matches literal text —
so the newly "added" tokens could basically never match real Devanagari/Telugu
text, except by accidental pure-ASCII overlap. Existing tests didn't catch
this because they used ASCII-only dummy corpora, where GPT-2's byte remapping
happens to be the identity function.

The fix: `train_sub_tokenizer` now uses a `Metaspace` (SentencePiece-style)
pre-tokenizer, which keeps vocab entries as literal Unicode text, and
`merge_vocabularies` decodes each token id back to its literal surface form
before adding it. Regression tests using real non-ASCII text were added in
`tests/test_expansion.py` (`test_merge_vocabularies_added_tokens_are_literal_substrings`,
`test_merge_vocabularies_actually_reduces_token_count`,
`test_merge_vocabularies_round_trip_decodes_correctly`) so this class of bug
can't silently regress. Without the fix, this benchmark's real numbers were
~100x smaller (0.3-0.6% instead of 78-88%).
