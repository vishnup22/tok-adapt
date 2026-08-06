"""Real, reproducible benchmark of tok_adapt against a standard 7B-class open model.

Uses:
  - Base tokenizer: Qwen/Qwen2.5-7B (ungated, publicly downloadable; Llama-3-8B
    was considered but requires accepting Meta's gated license on the Hub).
  - Target corpus: real Hindi + Telugu Wikipedia text pulled live from the
    Hugging Face Hub (vibrantlabsai/hindi-wikipedia, vengi-ai/telugu-wikipedia-clean).
    No synthetic/dummy text is used here.

This script only ever downloads the tokenizer + config.json for the base
model (a few MB), never the ~15GB of Qwen2.5-7B weights. Sequence-length /
fertility numbers come from actually running both tokenizers over held-out
text. The "embedding memory" figures are an ANALYTICAL projection computed
from the model's real config dimensions (vocab_size, hidden_size, dtype),
not a measured live VRAM delta from a loaded checkpoint -- loading the full
7B model was out of scope for this environment. That distinction is called
out explicitly in the output.

Run with: python benchmarks/run_benchmark.py
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoTokenizer

from tok_adapt.expansion import VocabularyExpander
from tok_adapt.metrics import FertilityEvaluator

BASE_MODEL_ID = "Qwen/Qwen2.5-7B"
CORPUS_DIR = Path(__file__).parent / "corpus"
RESULTS_PATH = Path(__file__).parent / "RESULTS.md"

SENTENCE_SPLIT_RE = re.compile(r"[।.!?]")


def count_sentences(text: str) -> int:
    return len([s for s in SENTENCE_SPLIT_RE.split(text) if len(s.strip()) >= 3])


def _load_telugu_candidates():
    """Loads the full real Telugu Wikipedia validation shard (cached after first call)."""
    import pandas as pd

    path = hf_hub_download(
        repo_id="vengi-ai/telugu-wikipedia-clean",
        filename="data/validation-00000-of-00001.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(path)
    df["len"] = df["text"].str.len()
    return df[(df["len"] > 800) & (df["len"] < 4000)].reset_index(drop=True)


def build_telugu_split(n_rows: int, offset: int) -> str:
    """Pulls a disjoint, purely verbatim slice of real Telugu Wikipedia rows."""
    df = _load_telugu_candidates()
    rows = df.iloc[offset : offset + n_rows]
    return "\n".join(t.strip().replace("\n", " ") for t in rows["text"])


def main() -> None:
    results = []

    def log(msg: str) -> None:
        print(msg)
        results.append(msg)

    log(f"# tok-adapt real benchmark — {BASE_MODEL_ID}\n")
    log(f"Run timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log("See benchmarks/README.md for methodology notes, including a byte-level "
        "vocab-remapping bug this benchmark run surfaced and fixed in expansion.py "
        "(without which these numbers would be ~100x smaller).\n")

    # --- 1. Load base tokenizer + config (small downloads, no model weights) ---
    log("## Setup\n")
    t0 = time.time()
    base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    config = AutoConfig.from_pretrained(BASE_MODEL_ID)
    log(f"- Base tokenizer: `{BASE_MODEL_ID}` (vocab size {len(base_tokenizer)}, "
        f"loaded in {time.time() - t0:.1f}s)")
    log(f"- Model config: hidden_size={config.hidden_size}, "
        f"config.vocab_size={config.vocab_size}, "
        f"tie_word_embeddings={config.tie_word_embeddings}, dtype={config.dtype}")
    log("- NOTE: only the tokenizer and config.json were downloaded here "
        "(a few MB). The ~15GB of Qwen2.5-7B weights were not downloaded; "
        "embedding-memory figures below are analytical, computed from these "
        "real config dimensions.\n")

    # --- 2. Assemble real training corpus (Hindi + Telugu Wikipedia) ---
    # Telugu volume is pulled directly and verbatim from the cached parquet
    # shard (cheap, no paraphrasing) to give the BPE trainer enough real text
    # to generalize past corpus-specific quirks; Hindi stays as the smaller,
    # hand-picked verbatim article set fetched earlier.
    log("Fetching real Telugu Wikipedia rows for training (verbatim, disjoint from eval)...")
    hindi_train = (CORPUS_DIR / "hindi_train.txt").read_text(encoding="utf-8")
    telugu_train = build_telugu_split(n_rows=350, offset=0)
    train_text = hindi_train + "\n" + telugu_train
    train_path = CORPUS_DIR / "train_combined.txt"
    train_path.write_text(train_text, encoding="utf-8")

    hindi_eval = (CORPUS_DIR / "hindi_eval.txt").read_text(encoding="utf-8")
    hindi_eval_extra = (CORPUS_DIR / "hindi_eval_extra.txt").read_text(encoding="utf-8")
    log("Fetching additional held-out Telugu Wikipedia rows (disjoint offset, not used in training)...")
    telugu_eval = build_telugu_split(n_rows=150, offset=350)
    eval_text = "\n".join([hindi_eval, hindi_eval_extra, telugu_eval])
    eval_path = CORPUS_DIR / "eval_combined.txt"
    eval_path.write_text(eval_text, encoding="utf-8")

    eval_lines = [l for l in eval_text.splitlines() if l.strip()]
    n_sentences = count_sentences(eval_text)
    log(f"\n## Corpus\n")
    log(f"- Training corpus (Hindi + Telugu Wikipedia, real text, disjoint from eval): "
        f"{len(train_text.split())} words, {len(train_text.encode('utf-8'))} bytes")
    log(f"- Held-out eval corpus (Hindi + Telugu Wikipedia, articles NOT seen during "
        f"sub-tokenizer training): {len(eval_lines)} paragraphs, ~{n_sentences} sentences, "
        f"{len(eval_text.split())} words, {len(eval_text.encode('utf-8'))} bytes\n")

    # --- 3. Expand vocabulary on the real training corpus ---
    log("## Vocabulary expansion\n")
    expander = VocabularyExpander(base_tokenizer)
    t0 = time.time()
    sub_tok_path = expander.train_sub_tokenizer(str(train_path), vocab_size=32000, algorithm="bpe")
    adapted_tokenizer = expander.merge_vocabularies(str(sub_tok_path))
    elapsed = time.time() - t0
    n_added = len(adapted_tokenizer) - len(base_tokenizer)
    log(f"- Trained a 32k-vocab BPE sub-tokenizer on the Hindi+Telugu training corpus "
        f"in {elapsed:.1f}s")
    log(f"- Merged {n_added} genuinely new tokens into the base tokenizer "
        f"({len(base_tokenizer)} -> {len(adapted_tokenizer)})\n")

    # --- 4. Measure real fertility / sequence-length reduction on held-out text ---
    log("## Sequence length reduction (measured on held-out text)\n")
    base_eval = FertilityEvaluator(base_tokenizer)
    adapted_eval = FertilityEvaluator(adapted_tokenizer)

    base_stats = base_eval.compute_fertility(eval_lines)
    adapted_stats = adapted_eval.compute_fertility(eval_lines)
    compression = base_eval.compare_sequence_compression(adapted_tokenizer, eval_lines)

    log(f"- Base `{BASE_MODEL_ID}` tokenizer: {base_stats['total_tokens']} tokens over "
        f"~{n_sentences} sentences ({base_stats['token_to_word_ratio']:.3f} tokens/word, "
        f"{base_stats['token_to_byte_ratio']:.3f} tokens/byte)")
    log(f"- tok-adapt-expanded tokenizer: {adapted_stats['total_tokens']} tokens over the "
        f"same text ({adapted_stats['token_to_word_ratio']:.3f} tokens/word, "
        f"{adapted_stats['token_to_byte_ratio']:.3f} tokens/byte)")
    log(f"- **{compression['base_tokens']} tokens -> {compression['adapted_tokens']} tokens "
        f"(~{compression['token_reduction_pct']:.1f}% token reduction, "
        f"compression ratio {compression['compression_ratio']:.3f}) on "
        f"{len(eval_lines)} held-out Hindi+Telugu paragraphs "
        f"(~{n_sentences} sentences).**\n")

    # --- 5. Analytical embedding memory (real config dims, no weights loaded) ---
    log("## Embedding memory (analytical, from real config dimensions)\n")
    bytes_per_param = 2  # bf16, per config.dtype above
    hidden = config.hidden_size
    full_vocab = config.vocab_size

    def matrix_gib(vocab_size: int, n_matrices: int) -> float:
        return (vocab_size * hidden * bytes_per_param * n_matrices) / (1024 ** 3)

    n_matrices = 1 if config.tie_word_embeddings else 2
    full_gib = matrix_gib(full_vocab, n_matrices)
    log(f"- Full {BASE_MODEL_ID} embedding footprint: vocab_size={full_vocab} x "
        f"hidden_size={hidden} x {bytes_per_param} bytes x {n_matrices} matrix"
        f"{'es' if n_matrices > 1 else ''} "
        f"(tie_word_embeddings={config.tie_word_embeddings}) = **{full_gib:.3f} GiB**")

    # Tokens actually used if we pruned the BASE tokenizer down to only what's
    # needed for this Hindi+Telugu domain (train+eval combined as a stand-in
    # target corpus), plus all special tokens -- this is exactly what
    # VocabularyPruner.prune_unused_tokens would retain.
    domain_text = train_text + "\n" + eval_text
    used_ids = set(base_tokenizer.encode(domain_text, add_special_tokens=False))
    used_ids |= set(base_tokenizer.all_special_ids)
    pruned_vocab = len(used_ids)
    pruned_gib = matrix_gib(pruned_vocab, n_matrices)
    log(f"- Tokens actually touched by this Hindi+Telugu domain corpus (what "
        f"`tok-adapt prune` would retain): {pruned_vocab} / {full_vocab} "
        f"({pruned_vocab / full_vocab * 100:.1f}% of the full vocab)")
    log(f"- Projected pruned embedding footprint: **{pruned_gib:.3f} GiB** "
        f"(saving **{full_gib - pruned_gib:.3f} GiB**, "
        f"{(1 - pruned_gib / full_gib) * 100:.1f}% reduction) -- projected from real "
        f"config dimensions, not measured on a loaded checkpoint.\n")

    # --- 6. Same expansion applied to an English-centric tokenizer (gpt2) ---
    # Repeats the identical expand+measure procedure against gpt2's
    # tokenizer -- a genuinely English-centric BPE vocab (50257 tokens,
    # near-byte-fallback on Devanagari/Telugu) still used as the base
    # tokenizer for many community fine-tunes -- as a second, independent
    # confirmation that the gains above aren't specific to Qwen's tokenizer
    # internals.
    log("## Same experiment against an English-centric tokenizer (gpt2)\n")
    log("Second, independent confirmation using `gpt2`'s tokenizer (English-centric, "
        "50257 vocab, still the base tokenizer for many community fine-tunes), which "
        "starts from much worse baseline fertility on this text than Qwen2.5-7B:\n")

    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    gpt2_expander = VocabularyExpander(gpt2_tokenizer)
    sub_tok_path_gpt2 = gpt2_expander.train_sub_tokenizer(str(train_path), vocab_size=32000, algorithm="bpe")
    gpt2_adapted = gpt2_expander.merge_vocabularies(str(sub_tok_path_gpt2))

    gpt2_base_eval = FertilityEvaluator(gpt2_tokenizer)
    gpt2_base_stats = gpt2_base_eval.compute_fertility(eval_lines)
    gpt2_adapted_stats = FertilityEvaluator(gpt2_adapted).compute_fertility(eval_lines)
    gpt2_compression = gpt2_base_eval.compare_sequence_compression(gpt2_adapted, eval_lines)

    log(f"- Base `gpt2` tokenizer: {gpt2_base_stats['total_tokens']} tokens over the same "
        f"held-out text ({gpt2_base_stats['token_to_byte_ratio']:.3f} tokens/byte -- "
        f"note this is ~{gpt2_base_stats['token_to_byte_ratio'] / base_stats['token_to_byte_ratio']:.2f}x "
        f"worse fertility than base {BASE_MODEL_ID} on the identical text, confirming gpt2's "
        f"vocab is genuinely under-provisioned for Hindi/Telugu)")
    log(f"- tok-adapt-expanded `gpt2` (same 32k sub-tokenizer training recipe, same corpus): "
        f"{gpt2_adapted_stats['total_tokens']} tokens "
        f"({gpt2_adapted_stats['token_to_byte_ratio']:.3f} tokens/byte)")
    log(f"- **{gpt2_compression['base_tokens']} tokens -> {gpt2_compression['adapted_tokens']} tokens "
        f"(~{gpt2_compression['token_reduction_pct']:.1f}% token reduction, "
        f"compression ratio {gpt2_compression['compression_ratio']:.3f}) on the same "
        f"{len(eval_lines)} held-out paragraphs (~{n_sentences} sentences).**\n")

    RESULTS_PATH.write_text("\n".join(results) + "\n", encoding="utf-8")
    print(f"\nFull results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
