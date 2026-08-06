# tok-adapt real benchmark — Qwen/Qwen2.5-7B

Run timestamp: 2026-08-06 17:06:19

See benchmarks/README.md for methodology notes, including a byte-level vocab-remapping bug this benchmark run surfaced and fixed in expansion.py (without which these numbers would be ~100x smaller).

## Setup

- Base tokenizer: `Qwen/Qwen2.5-7B` (vocab size 151665, loaded in 1.2s)
- Model config: hidden_size=3584, config.vocab_size=152064, tie_word_embeddings=False, dtype=torch.bfloat16
- NOTE: only the tokenizer and config.json were downloaded here (a few MB). The ~15GB of Qwen2.5-7B weights were not downloaded; embedding-memory figures below are analytical, computed from these real config dimensions.

Fetching real Telugu Wikipedia rows for training (verbatim, disjoint from eval)...
Fetching additional held-out Telugu Wikipedia rows (disjoint offset, not used in training)...

## Corpus

- Training corpus (Hindi + Telugu Wikipedia, real text, disjoint from eval): 79572 words, 1612120 bytes
- Held-out eval corpus (Hindi + Telugu Wikipedia, articles NOT seen during sub-tokenizer training): 173 paragraphs, ~3647 sentences, 34010 words, 691937 bytes

## Vocabulary expansion

- Trained a 32k-vocab BPE sub-tokenizer on the Hindi+Telugu training corpus in 3.4s
- Merged 28647 genuinely new tokens into the base tokenizer (151665 -> 180312)

## Sequence length reduction (measured on held-out text)

- Base `Qwen/Qwen2.5-7B` tokenizer: 381277 tokens over ~3647 sentences (11.211 tokens/word, 0.551 tokens/byte)
- tok-adapt-expanded tokenizer: 82774 tokens over the same text (2.434 tokens/word, 0.120 tokens/byte)
- **381277 tokens -> 82774 tokens (~78.3% token reduction, compression ratio 0.217) on 173 held-out Hindi+Telugu paragraphs (~3647 sentences).**

## Embedding memory (analytical, from real config dimensions)

- Full Qwen/Qwen2.5-7B embedding footprint: vocab_size=152064 x hidden_size=3584 x 2 bytes x 2 matrixes (tie_word_embeddings=False) = **2.030 GiB**
- Tokens actually touched by this Hindi+Telugu domain corpus (what `tok-adapt prune` would retain): 1237 / 152064 (0.8% of the full vocab)
- Projected pruned embedding footprint: **0.017 GiB** (saving **2.014 GiB**, 99.2% reduction) -- projected from real config dimensions, not measured on a loaded checkpoint.

## Same experiment against an English-centric tokenizer (gpt2)

Second, independent confirmation using `gpt2`'s tokenizer (English-centric, 50257 vocab, still the base tokenizer for many community fine-tunes), which starts from much worse baseline fertility on this text than Qwen2.5-7B:

- Base `gpt2` tokenizer: 680347 tokens over the same held-out text (0.983 tokens/byte -- note this is ~1.78x worse fertility than base Qwen/Qwen2.5-7B on the identical text, confirming gpt2's vocab is genuinely under-provisioned for Hindi/Telugu)
- tok-adapt-expanded `gpt2` (same 32k sub-tokenizer training recipe, same corpus): 83576 tokens (0.121 tokens/byte)
- **680347 tokens -> 83576 tokens (~87.7% token reduction, compression ratio 0.123) on the same 173 held-out paragraphs (~3647 sentences).**

